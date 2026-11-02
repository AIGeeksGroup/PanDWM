#!/usr/bin/env python3
"""Prepare the frozen common149 clips for official OmniNWM inference.

This is a data adapter only. It keeps OmniNWM's model, checkpoint, sampler,
12 Hz timing, six-view layout, 448x800 resolution, and 33-frame input. The
source is the same calibrated PanDWM nuScenes panorama cache. Common frame 9
is the condition; the following native 12 Hz records provide OmniNWM's
trajectory/ray-map inputs. Exact common target tokens 10..25 are mapped back
to their native 12 Hz generated indices for later scoring.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tvf

from dwm_pano.calibrated_stitch import camera_records_to_arrays
from dwm_pano.datasets.nuscenes_pano import _slice_full_pano_to_halves
from dwm_pano.datasets.pano_projection import CAM_ORDER
from dwm_pano.nuscenes6_bridge import (
    PANO_CANVAS_OVERLAP_COLUMNS,
    apply_camera_sampling_grid,
    build_rig_sampling_grids,
    canvas_source_axes,
)


OMNI_VIEW_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-pkl", required=True, type=Path)
    parser.add_argument("--pano-rgb-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_canvas(path: Path):
    with Image.open(path) as full:
        left, right = _slice_full_pano_to_halves(
            full.convert("RGB"), 2304, 256)
    return torch.cat([
        tvf.to_tensor(left),
        tvf.to_tensor(right)[..., PANO_CANVAS_OVERLAP_COLUMNS:],
    ], dim=-1)


def calibration_key(camera_to_ego, intrinsics, image_sizes):
    return (
        tuple(np.round(camera_to_ego, 6).ravel()),
        tuple(np.round(intrinsics, 5).ravel()),
        tuple(int(value) for value in image_sizes.ravel()),
    )


def adjusted_band_intrinsic(intrinsic, native_size, report, output_size):
    output_height, output_width = output_size
    native_width, _native_height = [int(value) for value in native_size]
    row_low = float(report["row_low"])
    row_high = float(report["row_high"])
    result = np.asarray(intrinsic, dtype=np.float32).copy()
    scale_x = output_width / native_width
    scale_y = output_height / (row_high - row_low)
    result[0, 0] *= scale_x
    result[0, 2] *= scale_x
    result[1, 1] *= scale_y
    result[1, 2] = (result[1, 2] - row_low) * scale_y
    return result


def main():
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)

    with args.common_pkl.open("rb") as handle:
        source = pickle.load(handle)
    infos = {item["token"]: item for item in source["infos"]}
    token_to_scene = {
        token: scene_index
        for scene_index, tokens in enumerate(source["scene_tokens"])
        for token in tokens
    }

    # Reuse the already-audited common clip enumerator instead of reimplementing
    # its timestamp matching here.
    vista_script_dir = Path(__file__).resolve().parents[1] / "vista"
    sys.path.insert(0, str(vista_script_dir))
    from audit_common_alignment import load_common_clips

    common_clips = load_common_clips(str(args.common_pkl))
    if args.max_clips is not None:
        common_clips = common_clips[:args.max_clips]

    output_infos = []
    output_scenes = []
    manifest_records = []
    output_size = (args.height, args.width)
    source_axes = canvas_source_axes(
        half_size=(256, 2304),
        right_trim_px=PANO_CANVAS_OVERLAP_COLUMNS,
    )
    grid_cache = {}
    image_root = args.out / "images"

    for clip_index, common_clip in enumerate(common_clips):
        condition = common_clip[9]
        scene_index = token_to_scene[condition["token"]]
        scene_tokens = source["scene_tokens"][scene_index]
        condition_position = scene_tokens.index(condition["token"])
        native_tokens = scene_tokens[condition_position:condition_position + 33]
        if len(native_tokens) != 33:
            raise ValueError(
                f"clip {clip_index} has only {len(native_tokens)} native frames")
        native_infos = [infos[token] for token in native_tokens]

        common_target_tokens = [item["token"] for item in common_clip[10:26]]
        scored_indices = [native_tokens.index(token)
                          for token in common_target_tokens]
        if scored_indices[0] <= 0 or len(set(scored_indices)) != 16:
            raise ValueError(
                f"invalid target mapping for clip {clip_index}: {scored_indices}")

        synthetic_tokens = []
        native_manifest = []
        first_reports = None
        for frame_index, info in enumerate(native_infos):
            camera_to_ego, intrinsics, image_sizes = camera_records_to_arrays(
                info["cams"])
            key = calibration_key(camera_to_ego, intrinsics, image_sizes)
            if key not in grid_cache:
                grid_cache[key] = build_rig_sampling_grids(
                    camera_to_ego,
                    intrinsics,
                    image_sizes,
                    source_size=(256, 4114),
                    output_size=output_size,
                    band_limit=True,
                    source_axes=source_axes,
                )
            grids, valids, reports = grid_cache[key]
            if first_reports is None:
                first_reports = reports

            pano_path = args.pano_rgb_dir / f"{int(info['timestamp'])}.jpg"
            if not pano_path.is_file():
                raise FileNotFoundError(pano_path)
            canvas = load_canvas(pano_path).to(args.device)
            with torch.inference_mode():
                frame_images = torch.stack([
                    apply_camera_sampling_grid(
                        canvas[None], grids[camera_index],
                        valids[camera_index]
                    )[0]
                    for camera_index in range(len(CAM_ORDER))
                ]).cpu()

            cloned = copy.deepcopy(info)
            synthetic_token = f"omni_common_{clip_index:04d}_{frame_index:02d}"
            synthetic_tokens.append(synthetic_token)
            cloned["source_token"] = info["token"]
            cloned["source_timestamp"] = int(info["timestamp"])
            cloned["token"] = synthetic_token
            # Force globally sorted records to remain contiguous per clip;
            # camera poses and source timestamps retain the real timing.
            cloned["timestamp"] = (
                10_000_000_000_000_000
                + clip_index * 10_000_000
                + frame_index * 83_333
            )

            saved_paths = {}
            for camera_index, camera_name in enumerate(CAM_ORDER):
                target = (image_root / f"clip_{clip_index:04d}"
                          / camera_name / f"{frame_index:02d}.jpg")
                target.parent.mkdir(parents=True, exist_ok=True)
                tvf.to_pil_image(frame_images[camera_index].clamp(0, 1)).save(
                    target, format="JPEG", quality=95, subsampling=0)
                saved_paths[camera_name] = str(target)
                camera = cloned["cams"][camera_name]
                adjusted_k = adjusted_band_intrinsic(
                    intrinsics[camera_index], image_sizes[camera_index],
                    reports[camera_index], output_size)
                camera["data_path"] = str(target)
                camera["camera_intrinsics"] = adjusted_k
                if "cam_intrinsic" in camera:
                    camera["cam_intrinsic"] = adjusted_k
                camera["width"] = args.width
                camera["height"] = args.height
            output_infos.append(cloned)
            native_manifest.append({
                "native_index": frame_index,
                "source_token": info["token"],
                "source_timestamp_us": int(info["timestamp"]),
                "paths": saved_paths,
            })

        output_scenes.append(synthetic_tokens)
        manifest_records.append({
            "common_clip_index": clip_index,
            "source_scene_index": scene_index,
            "condition_common_offset": 9,
            "condition_source_token": condition["token"],
            "condition_source_timestamp_us": int(condition["timestamp"]),
            "common_target_offsets": [10, 25],
            "common_target_tokens": common_target_tokens,
            "generated_indices_scored": scored_indices,
            "native_frames": native_manifest,
            "band_reports": first_reports,
        })
        print(f"prepared {clip_index + 1}/{len(common_clips)}", flush=True)

    output_payload = {
        "infos": output_infos,
        "metadata": source["metadata"],
        "scene_tokens": output_scenes,
    }
    pkl_path = args.out / "omninwm_common149_12hz.pkl"
    with pkl_path.open("wb") as handle:
        pickle.dump(output_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "protocol": "PanDWM-common149-OmniNWM-sixview-band-12Hz",
        "clips": len(common_clips),
        "official_input_frames": 33,
        "official_resolution": [args.height, args.width],
        "official_view_order": OMNI_VIEW_ORDER,
        "materialization_order": CAM_ORDER,
        "condition_common_offset": 9,
        "ground_truth_common_offsets": [10, 25],
        "scored_frames": 16,
        "source": str(args.common_pkl),
        "records": manifest_records,
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote", pkl_path)
    print("wrote", manifest_path)


if __name__ == "__main__":
    main()
