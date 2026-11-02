"""CPU probe: can the 700-scene full train set feed the SRR pipeline?

Counts 67- and 195-frame clips, checks dev/heldout leakage, checks that every
frame of every clip has RGB/bbox/hdmap files, and loads one sample.
"""

import argparse
import gc
import json
import os
from collections import Counter
from pathlib import Path

from dwm_pano.datasets.nuscenes_pano import PanoNuScenesDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--smoke-pkl", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = json.loads(Path(args.config).read_text())["training_dataset"]["base_dataset"]
    kwargs = {k: v for k, v in base.items()
              if k not in ("_class_name", "stub_key_data_dict")}
    data = Path(args.data_dir)
    kwargs.update(pkl_path=args.pkl, rgb_dir=str(data / "rgb"),
                  bbox_dir=str(data / "bbox"), hdmap_dir=str(data / "hdmap"))
    manifest = json.loads(Path(args.scene_manifest).read_text())
    forbidden = {s["scene_token"] for key in ("dev", "heldout")
                 for s in manifest[key]}
    report = {"pkl": args.pkl, "data_dir": args.data_dir,
              "fps_stride_tuples": kwargs["fps_stride_tuples"]}
    smoke_scenes = None
    for length in (67, 195):
        kw = dict(kwargs, sequence_length=length)
        full = PanoNuScenesDataset(**kw)
        scenes = [a["scene_token"] for a in full.clip_audit]
        missing = Counter()
        for window in full.clips:
            for info in window:
                ts = int(info["timestamp"])
                for kind, ext in (("rgb", "jpg"), ("bbox", "png"), ("hdmap", "png")):
                    if not os.path.exists(data / kind / f"{ts}.{ext}"):
                        missing[kind] += 1
        if smoke_scenes is None:
            smoke = PanoNuScenesDataset(**dict(kw, pkl_path=args.smoke_pkl))
            smoke_scenes = {a["scene_token"] for a in smoke.clip_audit}
            del smoke
            gc.collect()
        report[f"len{length}"] = {
            "clips": len(full.clips),
            "unique_scenes": len(set(scenes)),
            "max_clips_per_scene": max(Counter(scenes).values()) if scenes else 0,
            "forbidden_overlap": sorted(set(scenes) & forbidden),
            "missing_files": dict(missing),
            "smoke67_scenes_contained": len(smoke_scenes & set(scenes)),
        }
        print(json.dumps({length: report[f"len{length}"]}), flush=True)
        if length == 67:
            sample = full[0]
            report["sample0"] = {
                "keys": sorted(sample.keys()),
                "n_frames": len(sample.get("images", [])),
                "scene_token": sample.get("scene_token"),
                "split": sample.get("split"),
            }
            print(json.dumps(report["sample0"]), flush=True)
            del sample
        # The login node kills user processes near 8 GB RSS.
        del full, scenes
        gc.collect()
    report["smoke67_unique_scenes"] = len(smoke_scenes)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
