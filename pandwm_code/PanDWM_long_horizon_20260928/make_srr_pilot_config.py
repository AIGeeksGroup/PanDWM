"""Freeze a paired Stage-1 teacher recovery/control pilot configuration."""

import argparse
import json
from pathlib import Path


def build_config(base, *, arm, checkpoint, cache_dir, data_root,
                 scene_manifest, max_target_window=1, lr=1e-6, seed=0):
    if arm not in ("control", "recovery"):
        raise ValueError("invalid arm")
    if max_target_window not in (1, 2, 3):
        raise ValueError("max target window must be 1, 2 or 3")
    config = json.loads(json.dumps(base))
    root = Path(data_root)
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
    config["generator_seed"] = int(seed)
    config["data_shuffle"] = True
    config["fix_training_data_order"] = True
    config["optimizer"]["lr"] = float(lr)
    train = config["training_dataset"]["base_dataset"]
    train.update({
        "pkl_path": str(root / "pandwm_data/pickle/nuscenes_interp_12Hz_infos_train_smoke.pkl"),
        "rgb_dir": str(root / "pandwm_data/train/rgb"),
        "bbox_dir": str(root / "pandwm_data/bbox"),
        "hdmap_dir": str(root / "pandwm_data/hdmap"),
        "nuscenes_meta_dir": str(root / "pandwm_data/nuscenes_meta/v1.0-trainval"),
        "split": "train", "sequence_length": 67,
        "fps_stride_tuples": [[10, 20]],
        "bad_timestamps_us": [1531884295049111],
        "emit_instance_tracks": False,
    })
    val = config["validation_dataset"]["base_dataset"]
    val.update({
        "pkl_path": str(root / "pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl"),
        "rgb_dir": str(root / "pandwm_data/rgb"),
        "bbox_dir": str(root / "pandwm_data/bbox"),
        "hdmap_dir": str(root / "pandwm_data/hdmap"),
        "nuscenes_meta_dir": str(root / "pandwm_data/nuscenes_meta/v1.0-trainval"),
        "split": "val", "sequence_length": 19,
        "fps_stride_tuples": [[10, 0.1]],
        "bad_timestamps_us": [1531884295049111],
        "emit_instance_tracks": False,
    })
    config["training_dataloader"].update({
        "batch_size": 1, "num_workers": 2, "prefetch_factor": 2,
    })
    pc = config["pipeline"]
    pc["_class_name"] = (
        "dwm_pano.pipelines.ctsd_pano_srr.PanoCrossviewTemporalSDRecovery")
    pc["model_checkpoint_path"] = str(checkpoint)
    pc["srr_config"] = {
        "arm": arm, "cache_dir": str(cache_dir),
        "checkpoint": str(checkpoint),
        "forbidden_scene_manifest": str(scene_manifest),
        "max_target_window": max_target_window,
        "history_end_blend": 0.0,
        "window_curriculum": None,
    }
    # Train temporal history handling and the existing structure adapter.
    # Freeze whole top-level backbone modules before FSDP wrapping. This keeps
    # the single-80GB pilot feasible and is identical across paired arms.
    pc["training_config"].update({
        "freezing_pattern": (
            r"^(?!$)(?!temporal_transformer_blocks(?:\.|$))"
            r"(?!time_pos_embeds(?:\.|$))(?!time_mixers(?:\.|$))"
            r"(?!condition_image_adapter(?:\.|$))[^.]+$"
        ),
        "reference_frame_count": 3,
        "generation_task_ratio": 0.0,
        "image_generation_ratio": 0.0,
        "all_reference_visible_ratio": 1.0,
        "reference_visible_rate": 1.0,
        "disable_reference_frame_loss": True,
        "temporal_attention_mask": "causal",
    })
    pc["inference_config"].update({
        "reference_frame_count": 3,
        "sequence_length_per_iteration": 19,
        "temporal_attention_mask": "causal",
    })
    config.setdefault("informations", {}).update({
        "experiment": "Stage-1 teacher SRR-inspired native flow matching pilot",
        "arm": arm,
        "paired_control": "same checkpoint, data order, window schedule, RNG and optimizer",
        "objective": "real future GT latent, native flow matching",
        "cache_source": "train-split 50-step teacher closed-loop",
        "trainable_modules": "temporal blocks, time embeddings/mixers, condition image adapter",
        "max_target_window": max_target_window,
    })
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "recovery"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--max-target-window", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = build_config(
        json.loads(args.base.read_text()), arm=args.arm,
        checkpoint=args.checkpoint, cache_dir=args.cache_dir,
        data_root=args.data_root, max_target_window=args.max_target_window,
        scene_manifest=args.scene_manifest, lr=args.lr, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
