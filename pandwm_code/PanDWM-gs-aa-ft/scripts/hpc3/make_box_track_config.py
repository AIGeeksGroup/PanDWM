#!/usr/bin/env python3
"""Build a paired HPC3 config for the 3D-box track consistency ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_config(
    base,
    arm,
    seed,
    student,
    teacher,
    train_pkl,
    train_rgb,
    bbox_dir,
    hdmap_dir,
    loss_weight,
    max_tracks,
    lr,
    bad_timestamps_us,
):
    if arm not in ("control", "track"):
        raise ValueError("arm must be control or track")
    config = json.loads(json.dumps(base))
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
    config["generator_seed"] = int(seed)
    config["optimizer"]["lr"] = float(lr)

    dataset = config["training_dataset"]["base_dataset"]
    dataset.update({
        "pkl_path": train_pkl,
        "rgb_dir": train_rgb,
        "bbox_dir": bbox_dir,
        "hdmap_dir": hdmap_dir,
        "split": "train",
        "sequence_length": 19,
        "fps_stride_tuples": [[10, 0.1]],
        # Keep data loading identical across paired arms.  The control emits
        # the same small tensors but applies zero track-loss weight.
        "emit_instance_tracks": True,
        "max_instance_tracks": int(max_tracks),
        "instance_track_min_pixels": 4,
        "instance_track_min_visibility": 2,
        "instance_track_excluded_classes": ["barrier", "traffic_cone"],
        "bad_timestamps_us": [int(i) for i in bad_timestamps_us],
    })

    pipeline = config["pipeline"]
    pipeline.update({
        "model_checkpoint_path": student,
        "generator_ckpt": student,
        "consistency_ema_checkpoint_path": student,
        "consistency_teacher_checkpoint_path": teacher,
        # A short fine-tune must update its EMA from the first step; otherwise
        # an early checkpoint can remain byte-identical to the warm start.
        "ema_start_step": 0,
    })
    training = pipeline["training_config"]
    training.update({
        # Match the deployment task: 3 clean history -> 16 future frames.
        "reference_frame_count": 3,
        "generation_task_ratio": 0.0,
        "image_generation_ratio": 0.0,
        "all_reference_visible_ratio": 1.0,
        "reference_visible_rate": 1.0,
        "disable_reference_frame_loss": True,
        "temporal_attention_mask": "causal",
        "consistency_prefix_corruption_std": 0.0,
        "box_track_loss_weight": (
            float(loss_weight) if arm == "track" else 0.0),
        "box_track_pool_size": 2,
        "box_track_min_latent_size": 0.75,
        "box_track_max_temporal_gap": 2,
        "box_track_temporal_weight": 1.0,
        "box_track_cross_view_weight": 1.0,
        "box_track_negative_weight": 0.25,
        "box_track_max_negative_pairs": 2048,
    })
    config.setdefault("informations", {}).update({
        "experiment": "MV-TAP-inspired 3D-box relational consistency",
        "arm": arm,
        "paired_seed": int(seed),
        "warm_start": student,
        "frozen_teacher": teacher,
        "task": "3 clean context frames -> 16 future frames",
        "excluded_corrupt_timestamps_us": [
            int(i) for i in bad_timestamps_us],
        "track_supervision": {
            "source": "nuScenes persistent gt_box_ids + projected 3D boxes",
            "space": "predicted-clean VAE latent ROI relations",
            "positive_relations": ["adjacent-time same ID",
                                   "same-time overlapping-view same ID"],
            "anti_collapse": "same-frame/view different-ID real relation",
            "weight": training["box_track_loss_weight"],
        },
    })
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", choices=("control", "track"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--train-pkl", required=True)
    parser.add_argument("--train-rgb", required=True)
    parser.add_argument("--bbox-dir", required=True)
    parser.add_argument("--hdmap-dir", required=True)
    parser.add_argument("--loss-weight", type=float, default=0.05)
    parser.add_argument("--max-tracks", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument(
        "--bad-timestamp-us", action="append", type=int, default=[],
        help="Exclude every clip containing this known-corrupt frame.")
    args = parser.parse_args()
    if args.loss_weight < 0 or args.max_tracks < 1 or args.lr <= 0:
        parser.error("loss-weight must be non-negative; max-tracks/lr positive")

    with args.base.open("r", encoding="utf-8") as handle:
        base = json.load(handle)
    config = build_config(
        base, args.arm, args.seed, args.student, args.teacher,
        args.train_pkl, args.train_rgb, args.bbox_dir, args.hdmap_dir,
        args.loss_weight, args.max_tracks, args.lr,
        args.bad_timestamp_us)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
