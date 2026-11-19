#!/usr/bin/env python3
"""Export official OmniNWM six-view tensors into the common front metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--omni-manifest", required=True, type=Path)
    parser.add_argument("--common-manifest", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def load_rgb(path: Path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


def main():
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    omni = json.loads(args.omni_manifest.read_text(encoding="utf-8"))
    common = json.loads(args.common_manifest.read_text(encoding="utf-8"))
    if omni.get("clips") != 149 or common.get("clips") != 149:
        raise ValueError("manifests are not the frozen common149 protocol")
    view_order = omni["official_view_order"]
    front_index = view_order.index("CAM_FRONT")
    omni_records = {
        int(record["common_clip_index"]): record
        for record in omni["records"]
    }
    common_records = {
        int(record["clip_index"]): record
        for record in common["records"]
    }

    sources = {}
    for path in args.run_root.glob("shard_*/lossless/clip_*.pt"):
        clip_index = int(path.stem.rsplit("_", 1)[1])
        if clip_index in sources:
            raise RuntimeError(f"duplicate clip {clip_index}")
        sources[clip_index] = path
    if set(sources) != set(range(149)):
        raise RuntimeError(
            f"incomplete OmniNWM dump: {sorted(set(range(149)) - set(sources))}")

    real_dir = args.out / "real"
    fake_dir = args.out / "fake"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    for clip_index in range(149):
        tensor = torch.load(
            sources[clip_index], map_location="cpu", weights_only=True)
        if (tensor.shape != (6, 3, 33, 448, 800)
                or tensor.dtype != torch.uint8):
            raise ValueError(
                f"unexpected official output {sources[clip_index]}: "
                f"{tuple(tensor.shape)} {tensor.dtype}")
        generated_indices = omni_records[clip_index][
            "generated_indices_scored"]
        if len(generated_indices) != 16:
            raise ValueError(f"clip {clip_index} target mapping is not 16 frames")
        fake = tensor[front_index].permute(1, 0, 2, 3)[generated_indices]
        frame_paths = [
            Path(item["path"])
            for item in common_records[clip_index]["frames"][1:17]
        ]
        real = torch.stack([load_rgb(path) for path in frame_paths])
        name = f"clip_{clip_index:04d}.pt"
        torch.save(real, real_dir / name)
        torch.save(fake, fake_dir / name)
        if (clip_index + 1) % 10 == 0:
            print(f"exported {clip_index + 1}/149", flush=True)

    metadata = {
        "protocol": "PanDWM-common149-OmniNWM-official-front-band",
        "official_steps": 50,
        "official_output_frames": 33,
        "official_views": 6,
        "source_fps": 12,
        "target_fps": 10,
        "condition_frames_in_model": 1,
        "condition_frames_in_export": 0,
        "scored_frames": 16,
        "ground_truth_common_offsets": [10, 25],
        "generation_seed": args.seed,
        "front_view_index": front_index,
        "selection": "exact source-token positions; no temporal interpolation",
    }
    (args.out / "dump_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
