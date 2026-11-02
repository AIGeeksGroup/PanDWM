"""Held-out own-view evaluation for either GS supervision geometry.

Loads a decoder checkpoint, runs `--clip-count` clips starting at
`--clip-offset` (pick an offset beyond the trained epoch position for a
held-out slice of the same split), renders the pipeline's target cameras for
one frame per clip, and reports PSNR against the canvas-resampled ground
truth. The first clip's comparison grid is saved to --out-dir.

Run (GPU node that holds /local data):
    PYTHONPATH=src python src/dwm_pano/eval_pano_gs_decoder.py \
        --config configs/ctsd/pandwm/..._gs_decoder.json \
        --checkpoint <run>/checkpoints/<step>.pth \
        --clip-offset 100000 --clip-count 20 --out-dir <dir>
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

import dwm.common
from dwm_pano.pipelines.ctsd_pano_gs_decoder import (
    PanoGaussianDecoderPipeline,
    masked_mean,
)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Center-view PSNR evaluation for a GS decoder checkpoint.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="The gs_decoder training config json.")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Decoder state_dict .pth to evaluate.")
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Directory for the comparison grid and the report.")
    parser.add_argument(
        "--clip-offset", type=int, default=100000,
        help="First clip index of the evaluated slice.")
    parser.add_argument(
        "--clip-count", type=int, default=20,
        help="Number of clips to evaluate.")
    parser.add_argument(
        "--frame-index", type=int, default=0,
        help="Frame of each clip to evaluate.")
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device for the evaluation.")
    return parser


def collate_single(item):
    batch = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = [value]
    return batch


def peak_signal_to_noise_ratio(prediction, target, valid):
    mean_squared_error = masked_mean(
        (prediction - target).square(), valid)
    return float(10.0 * torch.log10(1.0 / mean_squared_error))


def save_comparison_grid(target_images, rendered_rgb, path):
    truth_row = torch.cat(list(target_images), dim=-1)
    rendered_row = torch.cat(list(rendered_rgb), dim=-1)
    grid = torch.cat([truth_row, rendered_row], dim=-2)
    grid = grid.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((grid * 255.0).astype(np.uint8)).save(path)


@torch.no_grad()
def main():
    args = create_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    config = json.load(open(args.config))
    pipeline = PanoGaussianDecoderPipeline(
        output_path=args.out_dir,
        config=config,
        device=device,
        training_config=config["pipeline"]["training_config"],
        pretrained_model_name_or_path=config["pipeline"][
            "pretrained_model_name_or_path"],
        decoder=config["pipeline"]["decoder"])
    pipeline.decoder.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu"))
    pipeline.decoder.eval()

    dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    frame_indices = torch.tensor([args.frame_index], dtype=torch.long)

    psnr_values = []
    l1_values = []
    for clip_index in range(
        args.clip_offset, args.clip_offset + args.clip_count
    ):
        batch = collate_single(dataset[clip_index])
        pipeline._prepare_nuscenes6_supervision(batch)
        half_images = batch["vae_images"][:, frame_indices].flatten(0, 1)
        half_images = half_images.to(device)
        if pipeline.decoder.use_range_prior:
            decoder_prior = pipeline._load_frame_ranges(batch, frame_indices)
        else:
            decoder_prior = None
        latent_origins, pixel_origins = pipeline._frame_ray_origins(
            batch, frame_indices)

        latents = pipeline.encode_half_latents(half_images)
        gaussians, _, _ = pipeline.decoder(
            latents, pipeline.latent_directions.unsqueeze(0),
            pipeline.pixel_directions.unsqueeze(0), decoder_prior,
            latent_origins=latent_origins, pixel_origins=pixel_origins)
        rendered_rgb, _, _, render_valid = \
            pipeline._render_own_supervision(gaussians[0])
        target_images = pipeline._target_rgb_for_supervision(half_images)

        psnr = peak_signal_to_noise_ratio(
            rendered_rgb, target_images, render_valid)
        l1 = float(masked_mean(
            (rendered_rgb - target_images).abs(), render_valid))
        psnr_values.append(psnr)
        l1_values.append(l1)
        print(
            f"[eval] clip {clip_index} psnr {psnr:.2f} dB "
            f"l1 {l1:.5f}")
        if clip_index == args.clip_offset:
            display_render = torch.where(
                render_valid.unsqueeze(1), rendered_rgb, target_images)
            save_comparison_grid(
                target_images, display_render,
                os.path.join(args.out_dir, f"clip{clip_index}.png"))

    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "supervision_geometry": pipeline.supervision_geometry,
        "clip_offset": args.clip_offset,
        "clip_count": len(psnr_values),
        "frame_index": args.frame_index,
        "psnr": float(np.mean(psnr_values)),
        "rgb_l1": float(np.mean(l1_values)),
    }
    with open(
        os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"[eval] mean psnr {report['psnr']:.2f} dB "
        f"l1 {report['rgb_l1']:.5f} over {len(psnr_values)} clips "
        f"(checkpoint {os.path.basename(args.checkpoint)})")


if __name__ == "__main__":
    main()
