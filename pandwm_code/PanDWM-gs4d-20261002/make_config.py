#!/usr/bin/env python3
"""Build a 4D Gaussian-decoder training config from the G2 anti-aliased config.

  python3 make_config.py --base <gs-aa-ft>/configs/ctsd/pandwm/hpc3/gs_decoder_g2_aa_eps0to0p1_250.json \
      --data-root /path/to/pandwm_root --out configs/gs4d_hpc2.json [--depth] [--smoke]

Only paths, the pipeline / dataset classes and the 4D options change; every
G2 rendering and loss setting is inherited, and the decoder warm-starts from
the G2-AA checkpoint (the decoder used for every current Gaussian demo).
"""

import argparse
import copy
import json
from pathlib import Path

GS4D = {
    "frame_stride": [2, 3, 4],
    "composite_targets": 2,
    "weight_own": 1.0,
    "weight_composite": 1.0,
    "weight_composite_perceptual": 0.1,
    "weight_loo": 0.5,
    "weight_motion": 0.0,
    "box_margin_m": 0.3,
    "loo_min_alpha": 0.5,
    "dynamic_pixel_alpha": 0.3,
    "checkpoint_decoder": True,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--depth", action="store_true",
                        help="keep G2 depth losses (needs <data-root>/pandwm_data/train/depth)")
    parser.add_argument("--motion", type=float, default=0.0,
                        help="weight of the box-motion dynamic loss (phase 2)")
    parser.add_argument("--frames-per-step", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--random-background", action="store_true",
                        help="composite training renders over a random colour (fixes night collapse)")
    parser.add_argument("--random-background-prob", type=float, default=1.0,
                        help="fraction of steps with a random background (rest black)")
    parser.add_argument("--coverage-weight", type=float, default=0.0,
                        help="weight of mean(1 - alpha) on each frame's own render (black background)")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny dataloader settings for a GPU smoke test")
    args = parser.parse_args()
    root = args.data_root.rstrip("/")
    base = json.loads(args.base.read_text())
    config = copy.deepcopy(base)

    pipe = config["pipeline"]
    pipe["_class_name"] = "dwm_pano_4d.pipeline.PanoGaussianDecoder4DPipeline"
    pipe["pretrained_model_name_or_path"] = f"{root}/pandwm_data/stable-diffusion-3.5-medium"
    pipe["initial_decoder_checkpoint_path"] = (
        f"{root}/pandwm_gs_aa_experiment/runs/g2_mix75_250_aa_eps0to0p1/checkpoints/250.pth")
    pipe["latent_domain_adaptation"]["enabled"] = False
    pipe["gs4d"] = dict(GS4D, weight_motion=args.motion,
                        random_background=args.random_background,
                        random_background_prob=args.random_background_prob,
                        weight_coverage=args.coverage_weight)
    tc = pipe["training_config"]
    tc["frames_per_step"] = args.frames_per_step
    # The warm start is already AA-trained: render at its terminal eps2d from
    # step 0 instead of re-running the G2-AA 0 -> 0.1 ramp.
    tc["aa_eps2d_start"] = tc.get("aa_eps2d_end", 0.1)
    tc["aa_eps2d_ramp_steps"] = 0
    tc["frame_sampling"] = "random"
    tc["weight_cross_rgb_l1"] = 0.0
    tc["weight_cross_alpha_coverage"] = 0.0
    tc["range_dir"] = f"{root}/pandwm_data/train/depth"
    if not args.depth:
        # No depth mirror on HPC2: geometry is constrained by the multi-frame
        # composite / leave-one-out photometric losses instead.
        tc["weight_render_depth"] = 0.0
        tc["weight_range_l1"] = 0.0

    box_fields = {"_class_name": "dwm_pano_4d.dataset.PanoNuScenesBoxDataset",
                  "max_boxes": 128}
    train = config["training_dataset"]["base_dataset"]
    train.update(box_fields)
    train.update({
        "pkl_path": f"{root}/pandwm_recovered_metadata_20260929/"
                    "nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl",
        "rgb_dir": f"{root}/pandwm_full_train_700_20260929/rgb",
        "nuscenes_meta_dir": f"{root}/pandwm_data/nuscenes_meta/v1.0-trainval",
        "split": "train"})
    val = config["validation_dataset"]["base_dataset"]
    val.update(box_fields)
    val.update({
        "pkl_path": f"{root}/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl",
        "rgb_dir": f"{root}/pandwm_data/rgb",
        "nuscenes_meta_dir": f"{root}/pandwm_data/nuscenes_meta/v1.0-trainval"})
    config["optimizer"]["lr"] = args.lr
    config["train_epochs"] = 1000   # stop with --max-steps
    if args.smoke:
        config["training_dataloader"].update(
            {"num_workers": 2, "prefetch_factor": 2, "persistent_workers": False})
    config.setdefault("informations", {}).update({
        "experiment": "WorldSplat-style 4D training of the PanDWM Gaussian decoder",
        "base_config": str(args.base),
        "depth_losses": bool(args.depth),
        "init": "G2-AA 250.pth",
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(config, indent=2)
    # Every remaining base-config path is rooted at the HPC3 data root.
    text = text.replace("/path/to/pandwm_root", root)
    args.out.write_text(text + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
