#!/usr/bin/env python3
"""Build auditable Stage-2 restart and paired on-policy CD configs."""

import argparse
import json
from pathlib import Path

from dwm_pano.pipelines.stage2_inert_keys import INERT_OFFICIAL_STAGE2_KEYS


def build_config(base, arm, seed, student, teacher, train_pkl, train_rgb,
                 bbox_dir, hdmap_dir, meta_dir, curriculum_steps=2000):
    if arm not in ("warmup", "control", "self"):
        raise ValueError("arm must be warmup, control, or self")
    config = json.loads(json.dumps(base))
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
    config["generator_seed"] = int(seed)

    dataset = config["training_dataset"]["base_dataset"]
    dataset.update({
        "pkl_path": train_pkl,
        "rgb_dir": train_rgb,
        "bbox_dir": bbox_dir,
        "hdmap_dir": hdmap_dir,
        "nuscenes_meta_dir": meta_dir,
        "split": "train",
        "sequence_length": 19 if arm == "warmup" else 67,
        "fps_stride_tuples": [[10, 0.1]],
        "bad_timestamps_us": [1531884295049111],
        "emit_instance_tracks": False,
    })
    config["training_dataloader"].update({
        "batch_size": 1, "num_workers": 2, "prefetch_factor": 2,
    })

    pipeline = config["pipeline"]
    # The inherited official namespace contains 35 accepted but unused keys,
    # including distribution_loss="dmd" and teacher_forcing=True. Remove
    # them so the saved config states the objective actually trained here.
    for key in INERT_OFFICIAL_STAGE2_KEYS:
        pipeline.pop(key, None)
    pipeline["_class_name"] = (
        "dwm_pano.pipelines.ctsd_pano_consistency_distillation."
        "PanoCrossviewTemporalConsistencyDistillation"
        if arm == "warmup" else
        "dwm_pano.pipelines.ctsd_pano_self_forcing."
        "PanoSelfForcingConsistencyDistillation")
    pipeline.update({
        "model_checkpoint_path": student,
        "generator_ckpt": student,
        "consistency_ema_checkpoint_path": student,
        "consistency_teacher_checkpoint_path": teacher,
        "denoising_step_list": [1000, 625, 250, 42],
        "ema_start_step": 200 if arm == "warmup" else 0,
    })
    training = pipeline["training_config"]
    training.update({
        "reference_frame_count": 3,
        "generation_task_ratio": 0.0,
        "image_generation_ratio": 0.0,
        "all_reference_visible_ratio": 1.0,
        "reference_visible_rate": 1.0,
        "disable_reference_frame_loss": True,
        "temporal_attention_mask": "causal",
        "consistency_prefix_corruption_std": 0.0,
        "box_track_loss_weight": 0.0,
    })
    if arm != "warmup":
        pipeline["self_forcing"] = {
            "enabled": arm == "self",
            "rollout_windows": 4,
            "window_length": 19,
            "reference_frame_count": 3,
            "window_stride": 16,
            "curriculum_steps": int(curriculum_steps),
            "sampler_mode": "stochastic",
        }
    pipeline["inference_config"].update({
        "reference_frame_count": 3,
        "sequence_length_per_iteration": 19,
        "temporal_attention_mask": "causal",
    })
    config.setdefault("informations", {}).update({
        "experiment": "Self-Forcing-inspired on-policy context CD",
        "arm": arm,
        "student_initialization": student,
        "frozen_teacher": teacher,
        "source_stage": "Stage-1 causal teacher" if arm == "warmup"
                        else "shared 2000-step Stage-2 warmup",
        "not_original_self_forcing": (
            "Uses CD objective, without holistic DMD/SiD/GAN or rolling KV"),
    })
    return config


def main():
    parser = argparse.ArgumentParser()
    for name in ("base", "output", "student", "teacher", "train-pkl",
                 "train-rgb", "bbox-dir", "hdmap-dir", "meta-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--arm", choices=("warmup", "control", "self"),
                        required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curriculum-steps", type=int, default=2000)
    args = parser.parse_args()
    with open(args.base, encoding="utf-8") as handle:
        base = json.load(handle)
    config = build_config(
        base, args.arm, args.seed, args.student, args.teacher,
        args.train_pkl, args.train_rgb, args.bbox_dir, args.hdmap_dir,
        args.meta_dir, args.curriculum_steps)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(out)


if __name__ == "__main__":
    main()
