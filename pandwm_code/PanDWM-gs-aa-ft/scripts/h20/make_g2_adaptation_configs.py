#!/usr/bin/env python3
"""Derive the three matched G2 adaptation configs from the production G2."""

from __future__ import annotations

import copy
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs/ctsd/pandwm/h20"
BASE_PATH = CONFIG_DIR / "gs_decoder_nuscenes6_stitch_g2.json"
G2_CHECKPOINT = (
    "/path/to/pandwm_root/runs/gs_nuscenes6_stitch_g2/checkpoints/2500.pth"
)
CACHE_DIR = "/path/to/pandwm_root/cache/stage2_train_g2_seed0_offset2500_500"


ARMS = {
    "control": {
        "file": "gs_decoder_g2_ft_control_500.json",
        "weights": {
            "weight_rgb_l1": 1.0,
            "weight_perceptual": 0.05,
            "weight_render_depth": 0.01,
            "weight_range_l1": 0.01,
            "weight_seg": 0.1,
        },
        "adaptation": None,
        "note": (
            "Warm-start control from G2-2500 with the original loss weights; "
            "isolates the effect of another 500 optimizer steps."
        ),
    },
    "appearance": {
        "file": "gs_decoder_g2_ft_appearance_500.json",
        "weights": {
            "weight_rgb_l1": 1.0,
            "weight_perceptual": 0.1,
            "weight_render_depth": 0.002,
            "weight_range_l1": 0.002,
            "weight_seg": 0.05,
        },
        "adaptation": None,
        "note": (
            "Appearance-balanced G2 fine-tune: increase LPIPS and reduce the "
            "depth-dominated gradient budget while retaining geometry anchors."
        ),
    },
    "appearance_stage2mix": {
        "file": "gs_decoder_g2_ft_appearance_stage2mix_500.json",
        "weights": {
            "weight_rgb_l1": 1.0,
            "weight_perceptual": 0.1,
            "weight_render_depth": 0.002,
            "weight_range_l1": 0.002,
            "weight_seg": 0.05,
        },
        "adaptation": {
            "enabled": True,
            "cache_dir": CACHE_DIR,
            "probability": 0.5,
            "strict": True,
            "supervise_geometry": False,
            "vae_decode_batch": 2,
        },
        "note": (
            "Appearance-balanced loss plus a 50/50 clean/actual Stage-2 "
            "four-step latent mix. Generated samples use the exact latent's "
            "SD3.5 VAE RGB as the aligned appearance target and do not receive "
            "misaligned GT depth/semantic supervision."
        ),
    },
}


def main() -> None:
    with BASE_PATH.open("r", encoding="utf-8") as handle:
        base = json.load(handle)
    for name, arm in ARMS.items():
        config = copy.deepcopy(base)
        config["pipeline"]["initial_decoder_checkpoint_path"] = G2_CHECKPOINT
        config["pipeline"].pop("latent_refinement", None)
        if arm["adaptation"] is None:
            config["pipeline"].pop("latent_domain_adaptation", None)
        else:
            config["pipeline"]["latent_domain_adaptation"] = arm["adaptation"]
        config["pipeline"]["training_config"].update(arm["weights"])
        config["informations"] = {
            "stage": f"gs_decoder_g2_ft_{name}_500",
            "parent_checkpoint": G2_CHECKPOINT,
            "ablation_budget_steps": 500,
            "initial_data_offset": 2500,
            "notes": arm["note"],
        }
        output = CONFIG_DIR / arm["file"]
        with output.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=4, ensure_ascii=True)
            handle.write("\n")
        print(output)

    smoke = copy.deepcopy(base)
    smoke["pipeline"]["initial_decoder_checkpoint_path"] = G2_CHECKPOINT
    smoke["pipeline"].pop("latent_refinement", None)
    smoke["pipeline"]["training_config"].update(
        ARMS["appearance_stage2mix"]["weights"])
    smoke_adaptation = copy.deepcopy(
        ARMS["appearance_stage2mix"]["adaptation"])
    smoke_adaptation["cache_dir"] = \
        "/path/to/pandwm_root/cache/stage2_train_g2_smoke_20260825"
    smoke["pipeline"]["latent_domain_adaptation"] = smoke_adaptation
    smoke["training_dataloader"].update({
        "num_workers": 1,
        "prefetch_factor": 1,
        "persistent_workers": False,
    })
    smoke["informations"] = {
        "stage": "gs_decoder_g2_ft_appearance_stage2mix_smoke1",
        "parent_checkpoint": G2_CHECKPOINT,
        "notes": "One-step end-to-end cache/target/loss smoke only.",
    }
    smoke_path = CONFIG_DIR / \
        "gs_decoder_g2_ft_appearance_stage2mix_smoke1.json"
    with smoke_path.open("w", encoding="utf-8") as handle:
        json.dump(smoke, handle, indent=4, ensure_ascii=True)
        handle.write("\n")
    print(smoke_path)


if __name__ == "__main__":
    main()
