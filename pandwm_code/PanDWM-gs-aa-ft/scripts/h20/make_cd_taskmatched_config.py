#!/usr/bin/env python3
"""Create the auditable H20 config for task-matched causal distillation."""

import argparse
import json
from pathlib import Path


def build_config(base: dict, teacher: str, mesh_size: int, eval_items: int):
    config = json.loads(json.dumps(base))
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, mesh_size]

    train_base = config["training_dataset"]["base_dataset"]
    train_base.update({
        # h20's local rgb_dir only mirrors the 85-scene smoke subset (not the
        # full 700-scene train split); train_with_bid_caption.pkl indexes all
        # 700 and was verified (2026-08-29) to hit missing-file errors on
        # every batch. train_smoke.pkl's 19795 timestamps are 100% present.
        "pkl_path": (
            "/path/to/pandwm_root/data/pickle/"
            "nuscenes_interp_12Hz_infos_train_smoke.pkl"
        ),
        "rgb_dir": "/path/to/pandwm_root/data/train/rgb",
        "bbox_dir": "/path/to/pandwm_root/data/bbox",
        "hdmap_dir": "/path/to/pandwm_root/data/hdmap",
        "nuscenes_meta_dir": (
            "/path/to/pandwm_root/data/nuscenes_meta/v1.0-trainval"
        ),
        "split": "train",
        "sequence_length": 19,
        "fps_stride_tuples": [[10, 0.1]],
    })

    pipeline = config["pipeline"]
    for key in (
        "model_checkpoint_path",
        "generator_ckpt",
        "consistency_teacher_checkpoint_path",
        "consistency_ema_checkpoint_path",
    ):
        pipeline[key] = teacher

    training = pipeline["training_config"]
    training.update({
        "reference_frame_count": 3,
        # The deployment task is always 3 clean historical frames followed
        # by 16 predicted frames. Disable the original mixture of unconditional
        # generation, image generation and partial-prefix prediction.
        "generation_task_ratio": 0.0,
        "image_generation_ratio": 0.0,
        "all_reference_visible_ratio": 1.0,
        "reference_visible_rate": 1.0,
        "disable_reference_frame_loss": True,
        "temporal_attention_mask": "causal",
        # First task-matched run keeps this disabled. Follow-up robustness
        # arms may use a small value after the clean task-matched control.
        "consistency_prefix_corruption_std": 0.0,
        # Pin the consistency hyper-parameters instead of inheriting code
        # defaults, so the recipe is auditable from the config alone. Values
        # verified against the official reference
        # references/Causal-Forcing/configs/causal_cd_framewise.yaml:
        #   discrete_cd_N 48 / ema_weight 0.99 / ema_start_step 200.
        # ema_start_step is the one place the code default (0) diverged from
        # the reference (200); the others already matched by coincidence.
        "consistency_discrete_steps": 48,
        "consistency_ema_decay": 0.99,
        "consistency_ema_start_step": 200,
    })
    inference = pipeline["inference_config"]
    inference.update({
        "reference_frame_count": 3,
        "sequence_length_per_iteration": 19,
        "evaluation_item_count": eval_items,
        "temporal_attention_mask": "causal",
    })

    info = config.setdefault("informations", {})
    info.update({
        "stage": "task_matched_causal_consistency_distillation",
        "initial_causal_checkpoint": teacher,
        "task_definition": "3 clean history frames -> 16 future frames",
        "removed_training_mixtures": [
            "unconditional_generation",
            "single_image_generation",
            "partial_reference_visibility",
        ],
    })
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--mesh-size", type=int, default=4)
    parser.add_argument("--eval-items", type=int, default=5)
    args = parser.parse_args()
    if args.mesh_size < 1 or args.eval_items < 1:
        parser.error("mesh-size and eval-items must be positive")

    with args.base.open("r", encoding="utf-8") as handle:
        base = json.load(handle)
    config = build_config(
        base, args.teacher, args.mesh_size, args.eval_items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
