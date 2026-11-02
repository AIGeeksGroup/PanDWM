"""Write the non-test val scenes that hold a 163-frame (10-window) clip, in dataset order (2026-10-14)."""
import json
import sys

import dwm.common

cfg_path, split_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
test = set(json.load(open(split_path))["test_scene_tokens"])
config = json.load(open(cfg_path))
ds = config["validation_dataset"]["base_dataset"]
ds["sequence_length"] = 3 + 16 * 10
ds["fps_stride_tuples"] = [[10, 20]]
dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
rows = [{"dev76_index": None, "dataset_index": i, "scene_token": a["scene_token"]}
        for i, a in enumerate(dataset.base_dataset.clip_audit) if a["scene_token"] not in test]
for k, r in enumerate(rows):
    r["dev76_index"] = k
json.dump({"note": "non-test nuScenes val scenes with a valid 163-frame clip (3 GT + 10 windows x 16) at fps 10, "
                   "one clip per scene from the scene start; built 2026-10-14 by diag/dump_dev76.py",
           "windows": 10, "count": len(rows), "scenes": rows}, open(out, "w"), indent=1)
print(f"DEV76 {len(rows)} scenes -> {out}")
