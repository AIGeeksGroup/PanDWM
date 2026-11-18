"""tab:gs_recon on the 249-clip Test-S: merge the stored 213-clip per-image rows with the 36 new ones
and re-aggregate with the aggregation of dwm_pano/recon/score.py.

summarize / summarize_depth below are verbatim copies of the nested functions in score.py main() (md5 3bc20a25...),
which cannot be imported.  Step 1 re-aggregates the stored 213-clip rows alone and requires EXACT equality with the
stored score JSON ("image" and "depth" blocks); only then are the merged 249-clip numbers written.

    python merge_249.py            (run in ~/pandwm_runs/gs_recon_ext249)
"""
import json
import sys
from pathlib import Path

import numpy as np

H = Path.home()
X = H / "pandwm_runs/gs_recon_ext249"
R = H / "pandwm_runs/gs_recon"
OX = H / "pandwm_runs/gs_recon_extra"

# table row -> (stored 213-clip score JSON, new 36-clip score JSON)
ROWS = [
    ("Omni-Scene", "224x400", "omniscene_224x400", R / "scores/omniscene_224x400.json"),
    ("Ours (3D)", "224x400", "ours_3d_224x400", R / "scores/ours_3d_224x400.json"),
    ("Ours (4D)", "224x400", "ours_4d_224x400", R / "scores/ours_4d_224x400.json"),
    ("ReconDrive", "280x518", "recondrive_280x518", OX / "scores/recondrive_280x518.json"),
    ("Ours (3D)", "280x518", "ours_3d_280x518", OX / "scores/ours_3d_280x518.json"),
    ("Ours (4D)", "280x518", "ours_4d_280x518", OX / "scores/ours_4d_280x518.json"),
    ("DrivingForward", "352x640", "drivingforward_sf_352x640", R / "scores/drivingforward_sf_352x640.json"),
    ("Ours (3D)", "352x640", "ours_3d_352x640", R / "scores/ours_3d_352x640.json"),
    ("Ours (4D)", "352x640", "ours_4d_352x640", R / "scores/ours_4d_352x640.json"),
]
COLS = [("PSNR_cov", "image", "psnr_cov"), ("PSNR", "image", "psnr_fill"), ("SSIM", "image", "ssim_fill"),
        ("LPIPS", "image", "lpips_vgg_fill"), ("Cov.", "image", "coverage"), ("AbsRel", "depth", "abs_rel"),
        ("delta1", "depth", "delta1")]
# overridable by a wrapper (merge_249_a800.py): where the new 36-clip scores of a key live, output names
NEW_SCORES = {}                    # key -> score JSON path; default X / f"scores/{key}_ext36.json"
FINAL_NAME = "FINAL_249.json"
MERGED_DIR = "scores_249"
DIGITS = {"PSNR_cov": 2, "PSNR": 2, "SSIM": 3, "LPIPS": 3, "Cov.": 3, "AbsRel": 3, "delta1": 3}


# ---- verbatim from dwm_pano/recon/score.py main() ----
def summarize(rs, key_rows):
    out = {}
    for group in ["all"] + sorted({r["target"] for r in rs}):
        sel = [r for r in rs if group == "all" or r["target"] == group]
        agg = {"images": len(sel)}
        for k in [k for k in sel[0] if k not in ("test_index", "target", "camera")]:
            vals = [r[k] for r in sel if r[k] is not None]
            agg[k] = float(np.mean(vals)) if vals else None
            agg[f"{k}_n"] = len(vals)
        out[group] = agg
    return out


def summarize_depth(rs):
    out = {}
    for group in ["all"] + sorted({r["target"] for r in rs}):
        sel = [r for r in rs if group == "all" or r["target"] == group]
        agg = {}
        for tag in ("all", "static"):
            vals = [r[tag] for r in sel if r[tag] is not None]
            agg[tag] = {k: float(np.mean([v[k] for v in vals])) for k in
                        ("abs_rel", "rmse", "delta1", "abs_rel_med")} if vals else None
            if vals:
                agg[tag]["images"] = len(vals)
                agg[tag]["points"] = int(sum(v["points"] for v in vals))
        out[group] = agg
    return out
# ---- end verbatim ----


def table_values(res):
    a, d = res["image"]["all"], res["depth"]["all"]["static"]
    return {c: (a[k] if part == "image" else d[k]) for c, part, k in COLS}


def max_abs_diff(x, y, path=""):
    """max |x - y| over matching numeric leaves; raises on structural differences."""
    if isinstance(x, dict):
        if set(x) != set(y):
            raise ValueError(f"{path}: keys differ {sorted(set(x) ^ set(y))}")
        return max([max_abs_diff(x[k], y[k], f"{path}/{k}") for k in x] or [0.0])
    if x is None or y is None:
        if x is not y:
            raise ValueError(f"{path}: {x} vs {y}")
        return 0.0
    return abs(float(x) - float(y))


def verbatim_check():
    """The two functions above must be textually identical to the nested ones in score.py (after dedent)."""
    import inspect, textwrap
    score_src = (X / "snapshot/PanDWM-gs-aa-ft/src/dwm_pano/recon/score.py").read_text()
    block = score_src.split("    def summarize(rs, key_rows):", 1)[1].split("    result = {", 1)[0]
    block = textwrap.dedent("    def summarize(rs, key_rows):" + block).strip()
    mine = (inspect.getsource(summarize) + "\n" + inspect.getsource(summarize_depth)).strip()
    return block == mine


def repro_only():
    ok_all = verbatim_check()
    print("verbatim summarize copy identical to score.py:", ok_all)
    for label, res_txt, key, old_path in ROWS:
        old = json.loads(old_path.read_text())
        old_rows = json.loads(old_path.with_suffix(".rows.json").read_text())
        rep = {"image": summarize(old_rows["image"], old_rows["image"]), "depth": summarize_depth(old_rows["depth"])}
        exact = rep["image"] == old["image"] and rep["depth"] == old["depth"]
        diff = max(max_abs_diff(rep["image"], old["image"]), max_abs_diff(rep["depth"], old["depth"]))
        v = table_values(rep)
        ok_all &= exact
        print(f"{label:15s} {res_txt} n={old['samples']} exact={exact} maxdiff={diff:.1e}  " +
              " ".join(f"{v[c]:.{DIGITS[c]}f}" for c in DIGITS), flush=True)
    print("REPRO213_OK" if ok_all else "REPRO213_FAIL")
    return 0 if ok_all else 1


def main():
    if not verbatim_check():
        print("summarize copy differs from score.py")
        return 1
    out_dir = X / MERGED_DIR
    out_dir.mkdir(exist_ok=True)
    final = {"note": "tab:gs_recon on Test-S extended to 83 scenes / 249 clips (tests249_manifest.json). "
                     "Per-image rows of the stored 213-clip scores + the 36 new clips (test_index 213..248) scored "
                     "with the same scorer, targets construction, masks code and renderers; aggregation = "
                     "dwm_pano/recon/score.py (mean over images; depth: per-image metrics averaged over images "
                     "with points). Columns: PSNR_cov = image.all.psnr_cov, PSNR = image.all.psnr_fill, SSIM = "
                     "image.all.ssim_fill, LPIPS = image.all.lpips_vgg_fill, Cov. = image.all.coverage, AbsRel = "
                     "depth.all.static.abs_rel, delta1 = depth.all.static.delta1.",
             "columns": [c for c, _, _ in COLS], "rows": []}
    all_ok = True
    for label, res_txt, key, old_path in ROWS:
        old = json.loads(old_path.read_text())
        old_rows = json.loads(old_path.with_suffix(".rows.json").read_text())
        new_path = Path(NEW_SCORES.get(key, X / f"scores/{key}_ext36.json"))
        new = json.loads(new_path.read_text())
        new_rows = json.loads(new_path.with_suffix(".rows.json").read_text())
        # 1) reproduction of the stored 213-clip aggregate from its rows
        rep = {"image": summarize(old_rows["image"], old_rows["image"]), "depth": summarize_depth(old_rows["depth"])}
        exact = rep["image"] == old["image"] and rep["depth"] == old["depth"]
        diff = max(max_abs_diff(rep["image"], old["image"]), max_abs_diff(rep["depth"], old["depth"]))
        # the new file's own aggregate must equal the re-aggregation of its rows as well
        rep_new = {"image": summarize(new_rows["image"], new_rows["image"]),
                   "depth": summarize_depth(new_rows["depth"])}
        exact_new = rep_new["image"] == new["image"] and rep_new["depth"] == new["depth"]
        old_ti = sorted({r["test_index"] for r in old_rows["image"]})
        new_ti = sorted({r["test_index"] for r in new_rows["image"]})
        ok_sets = old_ti == list(range(213)) and new_ti == list(range(213, 249)) and old["samples"] == 213 \
            and new["samples"] == 36 and old["lidar"] == new["lidar"] == "lidar_sync" \
            and list(old["resolution"]) == list(new["resolution"])
        ok = exact and exact_new and ok_sets
        all_ok &= ok
        # 2) merged 249-clip aggregate
        rows = {"image": old_rows["image"] + new_rows["image"], "depth": old_rows["depth"] + new_rows["depth"]}
        merged = {"renders": [old["renders"], new["renders"]], "resolution": old["resolution"],
                  "samples": old["samples"] + new["samples"],
                  "image": summarize(rows["image"], rows["image"]), "depth": summarize_depth(rows["depth"]),
                  "lidar": old["lidar"], "protocol": old["protocol"],
                  "merged_from": [str(old_path), str(new_path)]}
        (out_dir / f"{key}.json").write_text(json.dumps(merged, indent=1) + "\n")
        (out_dir / f"{key}.rows.json").write_text(json.dumps(rows) + "\n")
        v213, v249, v36 = table_values(old), table_values(merged), table_values(new)
        img = merged["image"]
        final["rows"].append({
            "method": label, "resolution": res_txt, "score_key": key,
            "values_249": v249, "values_249_rounded": {c: round(v, DIGITS[c]) for c, v in v249.items()},
            "values_213": v213, "values_213_rounded": {c: round(v, DIGITS[c]) for c, v in v213.items()},
            "values_new36_only": v36,
            "n_samples_249": merged["samples"],
            "images_249": {g: img[g]["images"] for g in img},
            "depth_images_static_249": {g: (merged["depth"][g]["static"] or {}).get("images") for g in merged["depth"]},
            "images_new36": {g: new["image"][g]["images"] for g in new["image"]},
            "check_213_reaggregation_exact": exact, "check_213_max_abs_diff": diff,
            "check_new36_reaggregation_exact": exact_new, "check_test_index_sets": ok_sets,
            "stored_213_json": str(old_path), "new_36_json": str(new_path),
            "merged_249_json": str(out_dir / f"{key}.json")})
        print(f"{label:15s} {res_txt}  213 " + " ".join(f"{v213[c]:.{DIGITS[c]}f}" for c in DIGITS) +
              "  | 249 " + " ".join(f"{v249[c]:.{DIGITS[c]}f}" for c in DIGITS) +
              f"  | repro213 exact={exact} maxdiff={diff:.1e} new36 exact={exact_new} sets={ok_sets} "
              f"images {img['all']['images']}", flush=True)
    final["all_checks_ok"] = all_ok
    (X / FINAL_NAME).write_text(json.dumps(final, indent=1) + "\n")
    print("MERGE_OK" if all_ok else "MERGE_CHECK_FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(repro_only() if "--repro-only" in sys.argv else main())
