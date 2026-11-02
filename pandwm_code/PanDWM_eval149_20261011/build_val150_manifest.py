"""One-clip manifest for scene-0018's scene-start clip, completing Dev-149 to the 150 nuScenes val scenes (2026-10-19).

Built on the frozen eval dataset with relaxed_sync.install() (stride 20 s, one clip per scene).  Checks that the relaxed
dataset has exactly the 149 Dev-149 clips (same scene tokens and anchor timestamps) plus scene-0018 and nothing else.
The new clip gets test_index 149, so it never collides with a Dev-149 clip index; dataset_index refers to the relaxed
dataset (150 clips), so runs and feature extraction of this manifest must also go through relaxed_sync.py.
  python3 build_val150_manifest.py --base-config cd_eval_frozen149_hpc3.json --dev149 dev149_manifest_20261018.json \
      --output scene0018_manifest_20261019.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import dwm.common
import relaxed_sync

SCENE0018 = "b51869782c0e464b8021eb798609f35f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--dev149", required=True)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    relaxed_sync.install()
    config = json.loads(Path(a.base_config).read_text())
    ds = config["validation_dataset"]["base_dataset"]
    if ds["sequence_length"] != 35 or ds["fps_stride_tuples"] != [[10, 20]]:
        raise RuntimeError("expected the frozen 35-frame protocol config")
    base = dwm.common.create_instance_from_config(config["validation_dataset"]).base_dataset
    dev = json.load(open(a.dev149))["clips"]
    old = {(r["scene_token"], int(r["anchor_timestamp_us"])) for r in dev}
    new = [(i, a2["scene_token"], int(c[0]["timestamp"]), a2) for i, (c, a2) in enumerate(zip(base.clips, base.clip_audit))]
    extra = [n for n in new if (n[1], n[2]) not in old]
    missing = old - {(n[1], n[2]) for n in new}
    if len(new) != 150 or missing or len(extra) != 1 or extra[0][1] != SCENE0018:
        raise RuntimeError(f"unexpected relaxed dataset: {len(new)} clips, missing {len(missing)}, extra {[e[1] for e in extra]}")
    idx, token, anchor, audit = extra[0]
    row = {"test_index": 149, "dataset_index": idx, "scene_token": token, "offset_s": 0.0,
           "anchor_timestamp_us": anchor, "max_time_error_us": int(audit["max_time_error_us"]),
           "relaxed_tolerance_us": audit.get("relaxed_tolerance_us"), "frames": 35, "context_frames": 3, "fps": 10}
    out = {"note": "scene-0018 scene-start clip (val scene 150 of 150), admitted with relaxed_sync.py (50.49 ms > 50 ms "
                   "sync tolerance); dataset = cd_eval_frozen149_hpc3.json validation dataset + relaxed_sync, "
                   "fps_stride_tuples [[10, 20]]; with dev149_manifest_20261018.json it forms the 150-scene val set",
           "dataset_fps_stride_tuples": [[10, 20]], "dataset_len": 150, "count": 1, "clips": [row]}
    text = json.dumps(out, indent=1) + "\n"
    a.output.write_text(text)
    print(f"WROTE {a.output} sha256={hashlib.sha256(text.encode()).hexdigest()} row={row}")


if __name__ == "__main__":
    main()
