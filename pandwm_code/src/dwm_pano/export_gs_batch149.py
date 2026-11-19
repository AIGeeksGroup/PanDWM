"""Gaussian renders for the 149-clip protocol videos, one model load per job.

Reads ``clipNNN/latents.pt`` written by ``run_eval149_videos.py`` (35 latent
frames: 3 real context + 32 generated) and, for every clip, writes
``gs_original_cameras.mp4`` (GT top / Gaussian render bottom, original six
cameras stitched) and ``gs_novel_views.mp4`` (chase | lateral sweep).  Clips
are matched by scene token and anchor timestamp, not by index.
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
from dwm_pano.export_gs_long_demo import look_at_c2w, pinhole

FRAMES = 35


def cameras(t):
    chase = look_at_c2w([-4.0, 0.0, 2.5], [10.0, 0.0, 1.0])
    y = 2.0 * math.sin(2.0 * math.pi * t / 40.0)
    sweep = look_at_c2w([0.0, y, 1.8], [15.0, y, 1.5])
    return np.stack([chase, sweep])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--data-pkl", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--clips-root", required=True, type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--max-scale", type=float, default=1.0)
    parser.add_argument("--time-budget-s", type=float, default=0.0)
    args = parser.parse_args()
    import time
    job_start, last_clip_s = time.monotonic(), 0.0
    config = load_config(args.config)
    torch.manual_seed(int(config["generator_seed"]))
    torch.cuda.set_device(0)
    started = ensure_single_rank_process_group(torch.device("cuda", 0))
    device = torch.device("cuda", 0)
    try:
        pipeline = dwm.common.create_instance_from_config(
            config["pipeline"], output_path=str(args.clips_root), config=config,
            device=device, resume_from=None)
        pipeline.vae.eval()
        pipeline.decoder.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu"), strict=True)
        pipeline.decoder.eval()
        pipeline.decoder_wrapper.eval()
        pipeline.rasterize_mode = "antialiased"
        pipeline.active_rasterize_eps2d = 0.1
        install_strict_pano_hardseam_stub()
        dataset_config = build_dataset_config(
            config["training_dataset"], pkl_path=args.data_pkl,
            fps_stride_tuples=[[10, 20]])
        dataset_config["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(dataset_config)
        by_key = {}
        for i, (clip, audit) in enumerate(zip(dataset.base_dataset.clips,
                                               dataset.base_dataset.clip_audit)):
            by_key[(audit["scene_token"], int(clip[0]["timestamp"]))] = i
        from dwm.datasets.common import CollateFnIgnoring
        w, h = 1280, 640
        K = torch.tensor(np.stack([pinhole(w, h, 90.0)] * 2), device=device,
                         dtype=pipeline.pixel_directions.dtype)
        for index in range(args.start, args.end):
            clip_dir = args.clips_root / f"clip{index:03d}"
            summary_path = clip_dir / "summary.json"
            if not summary_path.is_file():
                print(f"[gs149] missing video clip {index}", flush=True)
                continue
            if (clip_dir / "gs_novel_views.mp4").is_file():
                print(f"[gs149] skip {index}", flush=True)
                continue
            if (args.time_budget_s and last_clip_s and time.monotonic() - job_start
                    + 1.2 * last_clip_s > args.time_budget_s):
                print(f"[gs149] budget reached before clip {index}", flush=True)
                break
            clip_start = time.monotonic()
            summary = json.loads(summary_path.read_text())
            key = (summary["scene_token"], int(summary["anchor_timestamp_us"]))
            if key not in by_key:
                raise KeyError(f"clip {index}: no GS dataset clip for {key}")
            batch = next(iter(torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset, [by_key[key]]), batch_size=1,
                shuffle=False, num_workers=0, collate_fn=CollateFnIgnoring(keys=[]))))
            pipeline._prepare_nuscenes6_supervision(batch)
            latents = torch.load(clip_dir / "latents.pt", map_location="cpu",
                                 weights_only=True)["latents"]
            if latents.shape[0] != FRAMES:
                raise ValueError(f"clip {index}: {latents.shape[0]} latent frames")
            original, novel = [], []
            for t in range(FRAMES):
                selector = torch.tensor([t], dtype=torch.long)
                lo, po = pipeline._frame_ray_origins(batch, selector)
                g = pipeline.decoder_wrapper(
                    latents[t:t + 1].to(device, dtype=pipeline.latent_directions.dtype),
                    pipeline.latent_directions.unsqueeze(0),
                    pipeline.pixel_directions.unsqueeze(0),
                    None, previous_latents=None, temporal_context_mask=0.0,
                    latent_origins=lo, pixel_origins=po, camera_origin_scale=1.0)[0][0]
                rendered, _, _, _ = pipeline._render_own_supervision(g)
                rendered = torch.cat(list(pipeline.mask_supervision_for_display(rendered)), dim=-1)
                target = pipeline._target_rgb_for_supervision(
                    batch["vae_images"][:, selector].flatten(0, 1).to(device))
                target = torch.cat(list(pipeline.mask_supervision_for_display(target)), dim=-1)
                original.append(torch.cat((target, rendered), dim=-2).clamp(0, 1).cpu())
                compact = g[g[:, 11:14].amax(dim=-1) <= args.max_scale]
                c2w = torch.tensor(cameras(t), device=device, dtype=K.dtype)
                views, _, _ = pipeline.render_targets(
                    compact, target_c2w=c2w, target_intrinsics=K, width=w, height=h)
                novel.append(torch.cat(list(views), dim=-1).clamp(0, 1).cpu())
            fps = float(batch["fps"][0])
            save_tensor_to_video(str(clip_dir / "gs_original_cameras.mp4"),
                                 "libx264", fps, original)
            save_tensor_to_video(str(clip_dir / "gs_novel_views.mp4"),
                                 "libx264", fps, novel)
            last_clip_s = time.monotonic() - clip_start
            print(f"[gs149] clip {index} done ({last_clip_s:.0f}s)", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
