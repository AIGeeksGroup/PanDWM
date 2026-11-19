#!/usr/bin/env python3
"""Materialize Vista inputs/targets from the frozen 149-clip protocol.

This is a data adapter only.  It does not modify Vista's network, checkpoint,
sampler, or conditioning implementation.  For each PanDWM clip it extracts
the calibrated CAM_FRONT band for common-frame offsets 9..33.  Offset 9 is
Vista's single condition frame; generated offsets 1..16 are scored against
the common target frames 10..25.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tvf

from audit_common_alignment import load_common_clips
from dwm_pano.calibrated_stitch import camera_records_to_arrays
from dwm_pano.datasets.nuscenes_pano import _slice_full_pano_to_halves
from dwm_pano.nuscenes6_bridge import (
    PANO_CANVAS_OVERLAP_COLUMNS,
    apply_camera_sampling_grid,
    build_camera_sampling_grid,
    canvas_source_axes,
)
from dwm_pano.datasets.pano_projection import CAM_ORDER


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-pkl", required=True)
    parser.add_argument("--pano-rgb-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--start-offset", type=int, default=9)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max-clips", type=int)
    return parser.parse_args()


def calibration_key(camera_to_ego, intrinsics, image_size):
    return (
        tuple(np.round(camera_to_ego, 5).ravel()),
        tuple(np.round(intrinsics, 4).ravel()),
        tuple(int(value) for value in image_size),
    )


def load_canvas(path: Path):
    with Image.open(path) as full:
        left, right = _slice_full_pano_to_halves(
            full.convert("RGB"), 2304, 256)
    return torch.cat([
        tvf.to_tensor(left),
        tvf.to_tensor(right)[..., PANO_CANVAS_OVERLAP_COLUMNS:],
    ], dim=-1)


def main():
    args = parse_args()
    clips = load_common_clips(args.common_pkl)
    if args.max_clips is not None:
        clips = clips[:args.max_clips]
    if args.start_offset < 0 or args.frames <= 0:
        raise ValueError("start-offset must be non-negative and frames positive")
    if args.start_offset + args.frames > 35:
        raise ValueError("requested frames escape the frozen 35-frame clip")

    args.out.mkdir(parents=True, exist_ok=True)
    source_axes = canvas_source_axes(
        half_size=(256, 2304),
        right_trim_px=PANO_CANVAS_OVERLAP_COLUMNS)
    camera_index = CAM_ORDER.index("CAM_FRONT")
    grid_cache = {}
    manifest = []

    for clip_index, clip in enumerate(clips):
        selected = clip[args.start_offset:args.start_offset + args.frames]
        clip_dir = args.out / "clip_{:04d}".format(clip_index)
        clip_dir.mkdir(parents=True, exist_ok=True)
        frame_records = []
        for local_index, info in enumerate(selected):
            camera_to_ego, intrinsics, image_sizes = camera_records_to_arrays(
                info["cams"])
            c2e = camera_to_ego[camera_index]
            intrinsic = intrinsics[camera_index]
            image_size = image_sizes[camera_index]
            key = calibration_key(c2e, intrinsic, image_size)
            if key not in grid_cache:
                canvas_shape = (256, 4114)
                grid_cache[key] = build_camera_sampling_grid(
                    c2e,
                    intrinsic,
                    image_size,
                    source_size=canvas_shape,
                    output_size=(args.height, args.width),
                    band_limit=True,
                    camera_index=camera_index,
                    source_axes=source_axes,
                )
            grid, valid, report = grid_cache[key]
            pano_path = args.pano_rgb_dir / "{}.jpg".format(
                int(info["timestamp"]))
            if not pano_path.is_file():
                raise FileNotFoundError(pano_path)
            canvas = load_canvas(pano_path)
            band = apply_camera_sampling_grid(canvas[None], grid, valid)[0]
            target = clip_dir / "{:02d}.jpg".format(local_index)
            tvf.to_pil_image(band.clamp(0, 1)).save(
                target, format="JPEG", quality=95, subsampling=0)
            frame_records.append({
                "local_index": local_index,
                "common_offset": args.start_offset + local_index,
                "pano_timestamp_us": int(info["timestamp"]),
                "camera_timestamp_us": int(
                    info["cams"]["CAM_FRONT"]["timestamp"]),
                "path": str(target),
            })
        manifest.append({
            "clip_index": clip_index,
            "condition_common_offset": args.start_offset,
            "scored_common_offsets": [10, 25],
            "front_band_report": report,
            "frames": frame_records,
        })
        print("prepared {}/{}".format(clip_index + 1, len(clips)), flush=True)

    payload = {
        "protocol": "PanDWM-common149-Vista-single-front-band",
        "clips": len(clips),
        "frames_per_clip": args.frames,
        "condition_common_offset": args.start_offset,
        "generated_indices_scored": [1, 16],
        "ground_truth_common_offsets": [10, 25],
        "resolution": [args.height, args.width],
        "records": manifest,
    }
    with (args.out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("wrote", args.out / "manifest.json")


if __name__ == "__main__":
    main()
