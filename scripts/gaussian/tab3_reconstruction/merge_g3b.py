"""2026-11-01 tab:gs_recon rows of the FT3 G3b decoder on the 249-clip Test-S (CONSIST_PLAN_20261101.md section 5).

The 213-clip and 36-clip per-image rows of each G3b (mode, resolution) are concatenated and aggregated with the
verbatim copies of score.py's summarize functions in gs_recon_ext249_20261024/scripts/merge_249.py (its own verbatim
check is run first).  The stored rows of FINAL_249.json are printed next to the new ones.

    python merge_g3b.py [tag]  (in ~/pandwm_runs/gs_recon_g3b_20261101; tag g3b (default) or g3bs; writes
                               FINAL_<TAG>_249.json)
"""
import json
import sys
from pathlib import Path

H = Path.home()
sys.path.insert(0, str(H / "pandwm_runs/gs_recon_ext249_20261024/scripts"))
import merge_249 as m  # noqa: E402

X2 = H / "pandwm_runs/gs_recon_g3b_20261101"


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "g3b"
    if not m.verbatim_check():
        print("summarize copy differs from score.py")
        return 1
    final = {"note": "tab:gs_recon, FT3 G3b decoder (step 2000, md5 18e655d8), 249 Test-S clips; 3d = plane offsets "
                     "1 px with per-frame DA3, 4d = G3b's V2 scene (clip DA3 over frames 0..20 + fusion, plane "
                     "offsets 1 px); renders on A800 at all resolutions; scorer and masks as FINAL_249.json.",
             "columns": [c for c, _, _ in m.COLS], "rows": []}
    old = json.loads((H / "pandwm_runs/gs_recon_ext249_20261024/FINAL_249.json").read_text())
    for mode in ("3d", "4d"):
        for res in ("224x400", "280x518", "352x640"):
            rows = {"image": [], "depth": []}
            for part in ("s213", "ext36"):
                p = X2 / f"scores/{tag}_{mode}_{res}_{part}.rows.json"
                r = json.loads(p.read_text())
                rows["image"] += r["image"]
                rows["depth"] += r["depth"]
            rep = {"image": m.summarize(rows["image"], rows["image"]), "depth": m.summarize_depth(rows["depth"])}
            v = m.table_values(rep)
            label = f"{tag} ({mode.upper()})"
            final["rows"].append({"row": label, "res": res, "images": rep["image"]["all"]["images"], "values": v,
                                  "values_rounded": {c: round(v[c], m.DIGITS[c]) for c in m.DIGITS}})
            print(f"{label:10s} {res} images={rep['image']['all']['images']:5d}  " +
                  " ".join(f"{c} {v[c]:.{m.DIGITS[c]}f}" for c in m.DIGITS), flush=True)
    print("--- stored FINAL_249.json")
    print(json.dumps(old.get("rows", old), indent=None)[:4000])
    final["tag"] = tag
    (X2 / f"FINAL_{tag.upper()}_249.json").write_text(json.dumps(final, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
