"""Trajectory inference for the pano GS decoder on one clip.

Loads a decoder checkpoint, decodes `--frame-count` frames of one clip into
Gaussians, and renders every target camera at each `--lateral-shifts-m`
offset (0 = the supervised center). Output: one grid png per frame per shift.
This exercises the exact novel-view path the full world model will use once
DiT-generated latents replace the VAE latents.

Run (GPU node that holds /local data):
    PYTHONPATH=src python src/dwm_pano/infer_pano_gs_decoder.py \
        --config configs/ctsd/pandwm/..._gs_decoder.json \
        --checkpoint <run>/checkpoints/<step>.pth \
        --clip-index 100000 --lateral-shifts-m 0.0 1.0 2.0 --out-dir <dir>
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

import dwm.common
from dwm_pano.gs_geometry import build_virtual_camera
from dwm_pano.pipelines.ctsd_pano_gs_decoder import PanoGaussianDecoderPipeline


def create_parser():
    parser = argparse.ArgumentParser(
        description="Render a clip's Gaussians along lateral camera shifts.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="The gs_decoder training config json.")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Decoder state_dict .pth to run.")
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Directory for the rendered grids.")
    parser.add_argument(
        "--clip-index", type=int, required=True,
        help="Clip index in the validation dataset.")
    parser.add_argument(
        "--frame-count", type=int, default=4,
        help="Number of leading frames of the clip to decode.")
    parser.add_argument(
        "--lateral-shifts-m", type=float, nargs="+", default=[0.0, 1.0, 2.0],
        help="Ego +Y camera offsets rendered for every frame.")
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device for the inference.")
    return parser


def collate_single(item):
    batch = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = [value]
    return batch


def shifted_camera_stack(pipeline, lateral_shift_m):
    training_config = pipeline.training_config
    c2w_list = []
    for yaw_deg in training_config["target_yaws_deg"]:
        c2w, _ = build_virtual_camera(
            yaw_deg, training_config["target_height"],
            training_config["target_width"],
            training_config["target_fov_deg"], lateral_shift_m)
        c2w_list.append(c2w.astype(np.float32))
    return torch.from_numpy(np.stack(c2w_list)).to(pipeline.device)


def save_view_row(rendered_rgb, path):
    row = torch.cat(list(rendered_rgb), dim=-1)
    row = row.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((row * 255.0).astype(np.uint8)).save(path)


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
    batch = collate_single(dataset[args.clip_index])
    pipeline._prepare_nuscenes6_supervision(batch)
    shift_cameras = {
        shift: shifted_camera_stack(pipeline, shift)
        for shift in args.lateral_shifts_m
    }

    previous_latents = None
    use_temporal_context = pipeline.training_config.get(
        "evaluation_use_temporal_context", True)
    for frame_index in range(args.frame_count):
        frame_tensor = torch.tensor([frame_index], dtype=torch.long)
        half_images = batch["vae_images"][:, frame_tensor].flatten(0, 1)
        half_images = half_images.to(device)
        if pipeline.decoder.use_range_prior:
            decoder_prior = pipeline._load_frame_ranges(batch, frame_tensor)
        else:
            decoder_prior = None
        latent_origins, pixel_origins = pipeline._frame_ray_origins(
            batch, frame_tensor)

        latents = pipeline.encode_half_latents(half_images)
        gaussians, _, _ = pipeline.decoder(
            latents, pipeline.latent_directions.unsqueeze(0),
            pipeline.pixel_directions.unsqueeze(0), decoder_prior,
            previous_latents=(
                previous_latents if use_temporal_context else None),
            temporal_context_mask=(
                1.0 if use_temporal_context
                and previous_latents is not None else 0.0),
            latent_origins=latent_origins, pixel_origins=pixel_origins)
        previous_latents = (
            latents.detach() if use_temporal_context else None)

        for shift, camera_stack in shift_cameras.items():
            rendered_rgb, _, _ = pipeline.render_targets(
                gaussians[0], target_c2w=camera_stack)
            name = f"frame{frame_index:02d}_shift{shift:+.1f}m.png"
            save_view_row(rendered_rgb, os.path.join(args.out_dir, name))
        print(f"[infer] frame {frame_index} rendered "
              f"{len(shift_cameras)} shifts")
    print(f"[infer] saved to {args.out_dir}")


if __name__ == "__main__":
    main()
