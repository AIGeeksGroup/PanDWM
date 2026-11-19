#!/usr/bin/env python3
"""Run official Vista sampling on the frozen PanDWM common clips.

Only the data loop and lossless tensor dump are project-side.  Model loading,
conditioning, sampler construction, and denoising all call Vista's pinned
official implementation.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vista-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--low-vram", action="store_true")
    return parser.parse_args()


def quantize_unit(tensor):
    return tensor.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8)


def load_rgb_uint8(path: Path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


def atomic_torch_save(tensor, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.steps <= 0 or args.shards <= 0:
        raise ValueError("steps and shards must be positive")
    if not 0 <= args.shard < args.shards:
        raise ValueError("shard must be in [0, shards)")
    vista_root = args.vista_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    sys.path.insert(0, str(vista_root))
    os.chdir(vista_root)

    # Imported from the official pinned checkout, after fixing import/CWD.
    from sample import load_img
    from sample_utils import (
        do_sample,
        init_embedder_options,
        init_model,
        init_sampling,
        set_lowvram_mode,
    )

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["frames_per_clip"] != 25:
        raise ValueError("Vista official config requires exactly 25 frames")
    records = [
        record for record in manifest["records"]
        if record["clip_index"] % args.shards == args.shard
    ]
    if args.max_clips is not None:
        records = records[:args.max_clips]

    real_dir = args.out / "real"
    fake_dir = args.out / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "official_vista_root": str(vista_root),
        "checkpoint": str(checkpoint),
        "manifest": str(manifest_path),
        "seed": args.seed,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "frames": 25,
        "condition_indices": [0],
        "action": "free",
        "shards": args.shards,
        "shard": args.shard,
        "clip_indices": [record["clip_index"] for record in records],
        "quantization": "round(clamp(x,0,1)*255)",
    }
    with (args.out / "run_seed{}_shard{}.json".format(
            args.seed, args.shard)).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    set_lowvram_mode(args.low_vram)
    model = init_model({
        "config": str(vista_root / "configs/inference/vista.yaml"),
        "ckpt": str(checkpoint),
    })
    unique_keys = {embedder.input_key for embedder in model.conditioner.embedders}
    value_dict_base = init_embedder_options(unique_keys)
    sampler = init_sampling(
        guider="VanillaCFG",
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        num_frames=25)
    unconditional_keys = [
        "cond_frames", "cond_frames_without_noise", "command",
        "trajectory", "speed", "angle", "goal",
    ]

    for ordinal, record in enumerate(records):
        clip_index = int(record["clip_index"])
        name = "clip_{:04d}.pt".format(clip_index)
        real_path = real_dir / name
        fake_path = fake_dir / name
        if real_path.is_file() and fake_path.is_file():
            print("skip existing", name, flush=True)
            continue
        seed_everything(args.seed)
        paths = [Path(frame["path"]) for frame in record["frames"]]
        if len(paths) != 25 or not all(path.is_file() for path in paths):
            raise FileNotFoundError("incomplete 25-frame input for {}".format(name))
        # Match Vista's official IMG branch exactly: path_list contains the
        # same condition image 25 times.  Future GT is loaded separately only
        # for metric export, so it can never enter the model encoder.
        condition_frame = load_img(str(paths[0]), 576, 1024)
        images = condition_frame[None].repeat(25, 1, 1, 1)
        value_dict = dict(value_dict_base)
        condition = condition_frame[None]
        value_dict["cond_frames_without_noise"] = condition
        value_dict["cond_aug"] = 0.0
        value_dict["cond_frames"] = condition
        samples, samples_z, inputs = do_sample(
            images,
            model,
            sampler,
            value_dict,
            num_rounds=1,
            num_frames=25,
            force_uc_zero_embeddings=unconditional_keys,
            initial_cond_indices=[0])
        fake = quantize_unit(samples)
        real = torch.stack([load_rgb_uint8(path) for path in paths])
        if fake.shape != real.shape or tuple(fake.shape[:2]) != (25, 3):
            raise ValueError("unexpected real/fake shapes: {} {}".format(
                tuple(real.shape), tuple(fake.shape)))
        atomic_torch_save(real, real_path)
        atomic_torch_save(fake, fake_path)
        print("completed {}/{} {}".format(
            ordinal + 1, len(records), name), flush=True)
        del condition_frame, images, samples, samples_z, inputs, fake, real
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
