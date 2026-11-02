"""Render a 195-frame Gaussian demo from a saved closed-loop rollout.

Latents come from ``run_long_rollout.py`` (``window01.pt`` .. ``window12.pt``,
``future_latent`` = 16 generated frames each) with the three real context
frames VAE-encoded here, or, with ``--latent-source gt``, from VAE-encoding
every real frame (the decoder's upper bound on this clip).  Each frame is
decoded independently (the G2 protocol) and rendered three ways:

* the original six cameras, stitched exactly like the GS training target,
  stacked under the ground truth;
* a chase camera behind and above the ego vehicle;
* an orbit camera that circles the ego vehicle once over the clip.

Selected frames are also written as standard 3DGS ``.ply`` files (ego frame:
x forward, y left, z up, metres) for interactive viewers such as SuperSplat.
This is a qualitative demo, not a metric evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import dwm.common
from dwm.utils.preview import save_tensor_to_video
from dwm_pano.evaluate_gs import (
    build_dataset_config,
    ensure_single_rank_process_group,
    install_strict_pano_hardseam_stub,
    load_config,
)

FRAMES = 195
WINDOWS = 12
CONTEXT = 3
SH_C0 = 0.28209479177387814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--data-pkl", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--scene-token", required=True)
    parser.add_argument("--latent-source", choices=("rollout", "gt"),
                        default="rollout")
    parser.add_argument("--rollout-dir", type=Path,
                        help="run_long_rollout output dir (window01.pt ...)")
    parser.add_argument("--rasterize-mode", default="antialiased",
                        choices=("antialiased", "classic"))
    parser.add_argument("--eps2d", type=float, default=0.1)
    parser.add_argument("--ply-frames", default="3,50,100,150,194")
    parser.add_argument("--ply-min-opacity", type=float, default=0.02)
    parser.add_argument("--novel-width", type=int, default=1280)
    parser.add_argument("--novel-height", type=int, default=640)
    parser.add_argument("--novel-fov-deg", type=float, default=90.0)
    # Display-only filter for the novel views: Gaussians stretched along a
    # depth discontinuity (largest axis above this, metres) smear the frame.
    parser.add_argument("--novel-max-scale", type=float, default=1.0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.latent_source == "rollout" and args.rollout_dir is None:
        parser.error("--latent-source rollout needs --rollout-dir")
    return args


def look_at_c2w(position, target):
    """OpenCV camera (x right, y down, z forward) in the ego frame (z up)."""
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, position
    return c2w


def pinhole(width, height, fov_deg):
    focal = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[focal, 0.0, width / 2.0],
                     [0.0, focal, height / 2.0],
                     [0.0, 0.0, 1.0]])


NOVEL_VIEWS = ["chase: 4 m behind, 2.5 m up",
               "lateral sweep: +/-2 m sideways at 1.8 m, 4 s period",
               "orbit: radius 8 m, 4 m up, one turn per clip (shows limits)"]


def novel_cameras(frame_index):
    """Three ego-relative cameras; returns (3, 4, 4) c2w."""
    chase = look_at_c2w([-4.0, 0.0, 2.5], [10.0, 0.0, 1.0])
    y = 2.0 * math.sin(2.0 * math.pi * frame_index / 40.0)
    sweep = look_at_c2w([0.0, y, 1.8], [15.0, y, 1.5])
    angle = math.pi + 2.0 * math.pi * frame_index / FRAMES
    orbit = look_at_c2w([8.0 * math.cos(angle), 8.0 * math.sin(angle), 4.0],
                        [0.0, 0.0, 1.0])
    return np.stack([chase, sweep, orbit])


def write_ply(path, gaussians, keep):
    """Standard 3DGS binary PLY; gsplat quaternions are (w, x, y, z)."""
    g = gaussians[keep].detach().float().cpu().numpy()
    means, rgb, opacity = g[:, 0:3], g[:, 3:6], g[:, 6:7]
    quats, scales = g[:, 7:11], g[:, 11:14]
    opacity = np.clip(opacity, 1e-6, 1.0 - 1e-6)
    fields = [means, np.zeros_like(means), (rgb - 0.5) / SH_C0,
              np.log(opacity / (1.0 - opacity)), np.log(np.maximum(scales, 1e-8)),
              quats]
    data = np.concatenate(fields, axis=1).astype("<f4")
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
              "opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"])
    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {data.shape[0]}"]
    header += [f"property float {name}" for name in names]
    header.append("end_header")
    with open(path, "wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(data.tobytes())
    return int(data.shape[0])


def load_rollout_latents(rollout_dir, context_latents):
    windows = []
    for window in range(1, WINDOWS + 1):
        payload = torch.load(rollout_dir / f"window{window:02d}.pt",
                             map_location="cpu", weights_only=True)
        future = payload["future_latent"]
        if future.ndim != 6 or future.shape[:2] != (1, 16):
            raise ValueError(f"window {window}: unexpected {tuple(future.shape)}")
        windows.append(future[0])
    latents = torch.cat([context_latents.cpu().to(windows[0].dtype)] + windows)
    if latents.shape[0] != FRAMES:
        raise RuntimeError(f"expected {FRAMES} frames, got {latents.shape[0]}")
    return latents


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    config = load_config(args.config)
    torch.manual_seed(int(config["generator_seed"]))
    torch.cuda.set_device(0)
    process_group_started = ensure_single_rank_process_group(torch.device("cuda", 0))
    device = torch.device("cuda", 0)
    try:
        pipeline = dwm.common.create_instance_from_config(
            config["pipeline"], output_path=str(args.out), config=config,
            device=device, resume_from=None)
        pipeline.vae.eval()
        state = torch.load(args.checkpoint, map_location="cpu")
        pipeline.decoder.load_state_dict(state, strict=True)
        pipeline.decoder.eval()
        pipeline.decoder_wrapper.eval()
        pipeline.rasterize_mode = args.rasterize_mode
        pipeline.active_rasterize_eps2d = args.eps2d

        install_strict_pano_hardseam_stub()
        # No Stage-2 sampling happens here, so the plain GS dataset (no
        # bbox/hdmap conditioning images) is the right one.
        dataset_config = build_dataset_config(
            config["training_dataset"], pkl_path=args.data_pkl,
            fps_stride_tuples=[[10, 20]])
        dataset_config["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(dataset_config)
        matches = [i for i, a in enumerate(dataset.base_dataset.clip_audit)
                   if a["scene_token"] == args.scene_token]
        if len(matches) != 1:
            raise ValueError(f"scene token has {len(matches)} {FRAMES}-frame clips")
        from dwm.datasets.common import CollateFnIgnoring
        batch = next(iter(torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, matches), batch_size=1,
            shuffle=False, num_workers=0,
            # The plain GS dataset has no clip_text; default collation suffices.
            collate_fn=CollateFnIgnoring(keys=[]))))
        pipeline._prepare_nuscenes6_supervision(batch)
        images = batch["vae_images"]  # (1, T, 2, 3, H, W) in [0, 1]

        def encode(frames):
            return torch.cat([pipeline.encode_half_latents(
                images[0, start:start + 2].to(device))
                for start in frames], dim=0)

        if args.latent_source == "gt":
            latents = encode(range(0, FRAMES, 2))[:FRAMES].cpu()
        else:
            context = encode(range(0, CONTEXT, 2))[:CONTEXT]
            latents = load_rollout_latents(args.rollout_dir, context)

        intrinsics = pinhole(args.novel_width, args.novel_height, args.novel_fov_deg)
        K = torch.tensor(np.stack([intrinsics] * 3), device=device,
                         dtype=pipeline.pixel_directions.dtype)
        ply_frames = {int(x) for x in args.ply_frames.split(",") if x.strip()}
        content_valid = getattr(pipeline, "pixel_content_ray_valid", None)
        original_rows, novel_rows, ply_records = [], [], []
        for t in range(FRAMES):
            selector = torch.tensor([t], dtype=torch.long)
            latent_origins, pixel_origins = pipeline._frame_ray_origins(batch, selector)
            gaussians, _, _ = pipeline.decoder_wrapper(
                latents[t:t + 1].to(device, dtype=pipeline.latent_directions.dtype),
                pipeline.latent_directions.unsqueeze(0),
                pipeline.pixel_directions.unsqueeze(0),
                None, previous_latents=None, temporal_context_mask=0.0,
                latent_origins=latent_origins, pixel_origins=pixel_origins,
                camera_origin_scale=1.0)
            g = gaussians[0]
            rendered_rgb, _, _, _ = pipeline._render_own_supervision(g)
            rendered = torch.cat(list(
                pipeline.mask_supervision_for_display(rendered_rgb)), dim=-1)
            target_rgb = pipeline._target_rgb_for_supervision(
                images[:, selector].flatten(0, 1).to(device))
            target = torch.cat(list(
                pipeline.mask_supervision_for_display(target_rgb)), dim=-1)
            original_rows.append(torch.cat((target, rendered), dim=-2)
                                 .clamp(0.0, 1.0).cpu())
            c2w = torch.tensor(novel_cameras(t), device=device, dtype=K.dtype)
            compact = g[g[:, 11:14].amax(dim=-1) <= args.novel_max_scale]
            novel, _, _ = pipeline.render_targets(
                compact, target_c2w=c2w, target_intrinsics=K,
                width=args.novel_width, height=args.novel_height)
            novel_rows.append(torch.cat(list(novel), dim=-1).clamp(0.0, 1.0).cpu())
            if t in ply_frames:
                keep = g[:, 6] >= args.ply_min_opacity
                if content_valid is not None and content_valid.numel() == g.shape[0]:
                    keep &= content_valid.reshape(-1).to(keep.device)
                count = write_ply(args.out / f"frame{t:03d}.ply", g, keep)
                ply_records.append({"frame": t, "gaussians": count,
                                    "time_s": round(t / 10.0, 1),
                                    "generated": t >= CONTEXT and
                                    args.latent_source == "rollout"})
            if t % 25 == 0:
                print(f"[gs-demo] frame {t}/{FRAMES}", flush=True)

        fps = float(batch["fps"][0])
        save_tensor_to_video(str(args.out / "gs_original_cameras.mp4"),
                             "libx264", fps, original_rows)
        save_tensor_to_video(str(args.out / "gs_novel_views.mp4"),
                             "libx264", fps, novel_rows)
        record = {
            "purpose": "qualitative 195-frame Gaussian demo, not a metric",
            "scene_token": args.scene_token,
            "latent_source": args.latent_source,
            "rollout_dir": str(args.rollout_dir) if args.rollout_dir else None,
            "context_frames": CONTEXT,
            "checkpoint": str(args.checkpoint),
            "rasterize_mode": args.rasterize_mode, "eps2d": args.eps2d,
            "decoder_protocol": "independent-frame G2, temporal context off",
            "original_cameras_rows": ["ground truth", "Gaussian render"],
            "novel_views": NOVEL_VIEWS,
            "novel_view_filter": f"largest Gaussian axis <= {args.novel_max_scale} m",
            "novel_camera": {"width": args.novel_width, "height": args.novel_height,
                             "fov_deg": args.novel_fov_deg},
            "ply_frame_coordinates": "ego frame of that frame: x fwd, y left, z up (m)",
            "ply": ply_records,
        }
        (args.out / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)
    finally:
        if process_group_started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
