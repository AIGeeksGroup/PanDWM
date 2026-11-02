#!/usr/bin/env python3
"""Losslessly export official GEM arrays into the frozen common149 metric dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--common-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


def main() -> None:
    args = parse_args()
    common = json.loads(args.common_manifest.read_text(encoding="utf-8"))
    if common.get("clips") != 149 or common.get("frames_per_clip") != 25:
        raise ValueError("input is not the frozen 149x25 common manifest")
    common_records = {int(x["clip_index"]): x for x in common["records"]}

    real_dir = args.out / "real"
    fake_dir = args.out / "fake"
    if args.out.exists():
        raise FileExistsError(args.out)
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    exported = set()
    for shard in range(5):
        shard_root = args.run_root / f"shard_{shard}"
        order = json.loads(
            (shard_root / "input_order.json").read_text(encoding="utf-8")
        )
        npy_dirs = list(shard_root.glob("n_frames_25_steps_50_*/virtual/npy"))
        if len(npy_dirs) != 1:
            raise RuntimeError(f"expected one official GEM npy dir in {shard_root}")
        npy_dir = npy_dirs[0]
        for item in order["records"]:
            clip_index = int(item["common_clip_index"])
            local_index = int(item["gem_output_index"])
            if clip_index in exported:
                raise RuntimeError(f"duplicate common clip {clip_index}")
            source = npy_dir / f"ZeroShot_{local_index:06d}.npy"
            array = np.load(source, allow_pickle=False)
            if array.shape != (25, 576, 1024, 3) or array.dtype != np.uint8:
                raise ValueError(
                    f"unexpected official GEM output {source}: {array.shape} {array.dtype}"
                )
            fake = torch.from_numpy(array[:17].copy()).permute(0, 3, 1, 2)

            record = common_records[clip_index]
            frame_paths = [Path(x["path"]) for x in record["frames"][:17]]
            real = torch.stack([load_rgb(path) for path in frame_paths])
            if real.shape != fake.shape or real.dtype != torch.uint8:
                raise ValueError(
                    f"real/fake mismatch for clip {clip_index}: "
                    f"{tuple(real.shape)} vs {tuple(fake.shape)}"
                )
            name = f"clip_{clip_index:04d}.pt"
            torch.save(real, real_dir / name)
            torch.save(fake, fake_dir / name)
            exported.add(clip_index)
            if len(exported) % 10 == 0:
                print(f"exported {len(exported)}/149", flush=True)

    if exported != set(range(149)):
        missing = sorted(set(range(149)) - exported)
        raise RuntimeError(f"incomplete common clips: {missing}")
    metadata = {
        "protocol": "PanDWM-common149-GEM-official",
        "official_steps": 50,
        "official_output_frames": 25,
        "saved_frames": 17,
        "condition_frames": 1,
        "scored_frames": 16,
        "ground_truth_common_offsets": [10, 25],
        "generation_seed": args.seed,
    }
    (args.out / "dump_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
