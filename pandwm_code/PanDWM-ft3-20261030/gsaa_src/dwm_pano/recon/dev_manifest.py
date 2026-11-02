"""GS-Dev manifest (2026-10-24, round-2 fine-tune, GS_FT2_PLAN_20261024.md section 3): the Dev-78 scenes (validation
scenes that are not test scenes, dev78_manifest_20261018.json) at the clip starting OFFSET s after the scene start,
from the same enumeration as Test-S (val pkl, 35-frame clips, fps_stride_tuples [[10, 5.0]], offsets on the 5 s grid
of build_tests_manifest.py).  Same row format as tests213_manifest_20261018.json, so targets.py reads it unchanged.

    python -m dwm_pano.recon.dev_manifest --dev78 dev78_manifest_20261018.json --data-pkl <val pkl>
        --meta-dir <v1.0-trainval> --offset 10 --out gsdev78_manifest_20261024.json
"""
import argparse
import json
import pickle
from pathlib import Path

from dwm_pano.datasets.nuscenes_pano import _enumerate_pano_clips, _group_infos_by_scene, _load_token_to_scene
from dwm_pano.recon.targets import BAD_TIMESTAMPS_US

STRIDE_S = 5.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dev78", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--meta-dir", required=True)
    p.add_argument("--offset", type=float, default=10.0)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    scenes = [r["scene_token"] for r in json.loads(a.dev78.read_text())["clips"]]
    if len(set(scenes)) != len(scenes):
        raise RuntimeError("duplicate dev scenes")
    pkl = pickle.load(open(a.data_pkl, "rb"))
    grouped = _group_infos_by_scene(pkl["infos"], token2scene=_load_token_to_scene(a.meta_dir), strict=True)
    pairs = list(_enumerate_pano_clips(grouped, 35, [[10, STRIDE_S]], bad_timestamps_us=BAD_TIMESTAMPS_US))
    first = {}
    for clip, audit in pairs:
        first.setdefault(audit["scene_token"], int(clip[0]["timestamp"]))
    rows, missing = [], []
    for token in scenes:
        hit = [(i, clip) for i, (clip, audit) in enumerate(pairs) if audit["scene_token"] == token and
               round((int(clip[0]["timestamp"]) - first[token]) / 1e6 / STRIDE_S) * STRIDE_S == a.offset]
        if not hit:
            missing.append(token)
            continue
        i, clip = hit[0]
        rows.append({"test_index": len(rows), "dataset_index": i, "scene_token": token, "offset_s": a.offset,
                     "anchor_timestamp_us": int(clip[0]["timestamp"])})
    out = {"note": f"GS-Dev: Dev-78 scenes at {a.offset:g} s (dev_manifest.py, 2026-10-24); dataset_index into the "
                   "Test-S enumeration (val pkl, 35 frames, [[10, 5.0]])", "dataset_fps_stride_tuples": [[10, STRIDE_S]],
           "dataset_len": len(pairs), "count": len(rows), "missing_scenes": missing, "clips": rows}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[dev-manifest] {len(rows)} clips, {len(missing)} scenes without a {a.offset:g} s clip, dataset_len {len(pairs)}")


if __name__ == "__main__":
    main()
