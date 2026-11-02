"""Build strictly paired B0 continuation and conditioned history-noise arms.

The training objective remains consistency distillation. This experiment does
not claim to implement DMD or paper-faithful Self Forcing.
"""

import argparse
import json
from pathlib import Path

from dwm_pano.pipelines.stage2_inert_keys import INERT_OFFICIAL_STAGE2_KEYS


def build_config(base, arm, *, student, teacher, data_root, seed=0,
                 lr=2e-7, max_sigma=0.05, probability=0.5):
    if arm not in ("control", "noise"):
        raise ValueError("arm must be control or noise")
    if not 0 < lr <= 2e-6:
        raise ValueError("learning rate outside safe pilot range")
    if not 0 <= max_sigma <= 0.2 or not 0 <= probability <= 1:
        raise ValueError("noise pilot expects sigma <= 0.2 and probability <= 1")
    config = json.loads(json.dumps(base))
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
    config["generator_seed"] = int(seed)
    config["optimizer"]["lr"] = float(lr)
    config["data_shuffle"] = True
    config["fix_training_data_order"] = True

    dataset = config["training_dataset"]["base_dataset"]
    data_root = Path(data_root)
    dataset.update({
        "pkl_path": str(data_root / "pandwm_data/pickle/nuscenes_interp_12Hz_infos_train_smoke.pkl"),
        "rgb_dir": str(data_root / "pandwm_data/train/rgb"),
        "bbox_dir": str(data_root / "pandwm_data/bbox"),
        "hdmap_dir": str(data_root / "pandwm_data/hdmap"),
        "nuscenes_meta_dir": str(data_root / "pandwm_data/nuscenes_meta/v1.0-trainval"),
        "split": "train", "sequence_length": 19,
        "fps_stride_tuples": [[10, 0.1]],
        "bad_timestamps_us": [1531884295049111],
        "emit_instance_tracks": False,
    })
    config["training_dataloader"].update({
        "batch_size": 1, "num_workers": 2, "prefetch_factor": 2,
    })

    pc = config["pipeline"]
    for key in INERT_OFFICIAL_STAGE2_KEYS:
        pc.pop(key, None)
    pc["_class_name"] = (
        "dwm_pano.pipelines.ctsd_pano_consistency_distillation."
        "PanoCrossviewTemporalConsistencyDistillation")
    pc.update({
        "model_checkpoint_path": str(student),
        "generator_ckpt": str(student),
        "consistency_ema_checkpoint_path": str(student),
        "consistency_teacher_checkpoint_path": str(teacher),
        "denoising_step_list": [1000, 625, 250, 42],
        "ema_start_step": 0,
    })
    pc["training_config"].update({
        "reference_frame_count": 3,
        "generation_task_ratio": 0.0,
        "image_generation_ratio": 0.0,
        "all_reference_visible_ratio": 1.0,
        "reference_visible_rate": 1.0,
        "disable_reference_frame_loss": True,
        "temporal_attention_mask": "causal",
        "consistency_prefix_corruption_std": 0.0,
        "history_flow_noise_max_sigma": float(max_sigma if arm == "noise" else 0),
        "history_flow_noise_probability": float(probability if arm == "noise" else 0),
        "history_flow_noise_seed": int(seed) + 20260928,
        "box_track_loss_weight": 0.0,
    })
    pc["inference_config"].update({
        "reference_frame_count": 3,
        "sequence_length_per_iteration": 19,
        "temporal_attention_mask": "causal",
    })
    config.setdefault("informations", {}).update({
        "experiment": "conditioned flow-noise history CD pilot",
        "arm": arm, "base_student": str(student), "frozen_teacher": str(teacher),
        "history_noise_mechanism": (
            "reference=(1-sigma)*GT+sigma*epsilon; reference timestep=1000*sigma"
            if arm == "noise" else "clean GT reference"),
        "paired_control": "same data order, optimizer, initialization and CD target",
        "objective": "consistency distillation, not DMD/Self Forcing",
    })
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "noise"), required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-7)
    parser.add_argument("--max-sigma", type=float, default=0.05)
    parser.add_argument("--probability", type=float, default=0.5)
    args = parser.parse_args()
    base = json.loads(args.base.read_text())
    config = build_config(
        base, args.arm, student=args.student, teacher=args.teacher,
        data_root=args.data_root, seed=args.seed, lr=args.lr,
        max_sigma=args.max_sigma, probability=args.probability)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
