"""Build the Test-S manifest (2026-10-18; PAPER_COMPARISON_PLAN_20261012.md section 2.2).

Test-S = the 71 test scenes (review149_test71_long12_20261016.json, the same scenes as Test-L and as
dev149_nontest_split_20261013.json "test_scene_tokens"), 3 clips per scene starting 5, 10 and 15 s after the scene
start: 3 recorded frames + 32 frames, 10 Hz, 35 frames, exactly the Dev-149 protocol at later start times.

The clips are the dataset's own clips: the frozen eval dataset (cd_eval_frozen149_hpc3.json) with
fps_stride_tuples [[10, 5]] instead of [[10, 20]], so the anchor walks 0, 5, 10, 15 s and each clip is re-based at its
own first frame like a Dev-149 clip.  Offset 0 is the Dev-149 clip and is not part of Test-S.

  python3 build_tests_manifest.py --base-config cd_eval_frozen149_hpc3.json \
      --scenes review149_test71_long12_20261016.json --split dev149_nontest_split_20261013.json \
      --output tests213_manifest_20261018.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import dwm.common

STRIDE_S = 5.0
OFFSETS_S = (5.0, 10.0, 15.0)
CLIP = 35


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    scenes = [r["scene_token"] for r in json.load(open(a.scenes))["scenes"]]
    split_tokens = set(json.load(open(a.split))["test_scene_tokens"])
    if len(scenes) != 71 or set(scenes) != split_tokens:
        raise RuntimeError("the Test-L scene list and the dev/test split disagree")
    config = json.loads(Path(a.base_config).read_text())
    ds = config["validation_dataset"]["base_dataset"]
    if ds["sequence_length"] != CLIP or ds["fps_stride_tuples"] != [[10, 20]]:
        raise RuntimeError("expected the frozen 35-frame protocol config")
    ds["fps_stride_tuples"] = [[10, STRIDE_S]]
    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    base = dataset.base_dataset
    first_ts = {}
    for clip, audit in zip(base.clips, base.clip_audit):
        token = audit["scene_token"]
        first_ts.setdefault(token, int(clip[0]["timestamp"]))
    rows = []
    for token in scenes:
        found = {}
        for idx, (clip, audit) in enumerate(zip(base.clips, base.clip_audit)):
            if audit["scene_token"] != token:
                continue
            offset = round((int(clip[0]["timestamp"]) - first_ts[token]) / 1e6 / STRIDE_S) * STRIDE_S
            found[offset] = (idx, clip, audit)
        for off in OFFSETS_S:
            if off not in found:
                raise RuntimeError(f"scene {token} has no valid clip at {off} s (has {sorted(found)})")
            idx, clip, audit = found[off]
            rows.append({
                "test_index": len(rows), "dataset_index": idx, "scene_token": token,
                "scene_order": scenes.index(token), "offset_s": off,
                "anchor_timestamp_us": int(clip[0]["timestamp"]),
                "last_timestamp_us": int(clip[-1]["timestamp"]),
                "scene_first_timestamp_us": first_ts[token],
                "max_time_error_us": int(audit["max_time_error_us"]),
                "frames": CLIP, "context_frames": 3, "fps": 10,
            })
    out = {
        "note": "Test-S, 71 test scenes x clips at 5 / 10 / 15 s after the scene start; 3 recorded + 32 frames at "
                "10 Hz (35 frames); dataset = cd_eval_frozen149_hpc3.json validation dataset with fps_stride_tuples "
                "[[10, 5]]; dataset_index refers to that dataset; built 2026-10-18 by build_tests_manifest.py",
        "dataset_fps_stride_tuples": [[10, STRIDE_S]], "dataset_len": len(dataset),
        "count": len(rows), "clips": rows,
    }
    text = json.dumps(out, indent=1) + "\n"
    a.output.write_text(text)
    print(f"WROTE {a.output} clips={len(rows)} dataset_len={len(dataset)} "
          f"sha256={hashlib.sha256(text.encode()).hexdigest()}")
    err = max(r["max_time_error_us"] for r in rows)
    print(f"max_time_error_us over all clips: {err}")


if __name__ == "__main__":
    main()
