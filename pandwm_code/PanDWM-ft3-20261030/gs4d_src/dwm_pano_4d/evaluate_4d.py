"""Held-out-frame novel-view evaluation for 3D vs 4D Gaussian decoders.

For each validation clip, frames (a, t, b) = (0, gap, 2*gap) are decoded from
GT latents.  Frame t is never shown to the scene representation: the static
Gaussians of a and b are moved into t's ego frame and rendered at t's cameras.
Scores are computed on static (no dynamic box), covered (alpha > 0.5), valid
pixels only:

  psnr_pooled   static(a) + static(b) -> t        (what 4D training targets)
  psnr_single   static(a) -> t                    (one-source cross render)
  coverage_*    fraction of valid pixels the representation covers
  psnr_own      frame t decoded and rendered at its own cameras (round trip)

Run the same command with the G2 checkpoint and with a 4D checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import dwm.common
from dwm_pano.evaluate_gs import (
    build_dataset_config,
    ensure_single_rank_process_group,
    install_strict_pano_hardseam_stub,
    load_config,
)
from dwm_pano_4d.geometry import points_in_boxes, relative_ego_transform, transform_gaussians


def masked_psnr(rendered, target, mask):
    mask = mask.to(rendered.dtype)
    while mask.ndim < rendered.ndim:
        mask = mask.unsqueeze(1)
    mask = torch.broadcast_to(mask, rendered.shape)
    count = mask.sum()
    if count < 1:
        return float("nan")
    mse = ((rendered - target) ** 2 * mask).sum() / count
    return float(10.0 * torch.log10(1.0 / mse.clamp_min(1e-10)))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-pkl", required=True, type=Path)
    parser.add_argument("--clips", type=int, default=50)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--box-margin", type=float, default=0.3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    torch.manual_seed(int(config["generator_seed"]))
    torch.cuda.set_device(0)
    started = ensure_single_rank_process_group(torch.device("cuda", 0))
    device = torch.device("cuda", 0)
    try:
        pipeline = dwm.common.create_instance_from_config(
            config["pipeline"], output_path=str(args.output.parent), config=config,
            device=device, resume_from=None)
        pipeline.vae.eval()
        pipeline.decoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"),
                                         strict=True)
        pipeline.decoder.eval()
        pipeline.decoder_wrapper.eval()
        install_strict_pano_hardseam_stub()
        dataset_config = build_dataset_config(config["validation_dataset"], pkl_path=args.data_pkl,
                                              fps_stride_tuples=[[10, 20]])
        dataset_config["base_dataset"]["sequence_length"] = 2 * args.gap + 1
        dataset = dwm.common.create_instance_from_config(dataset_config)
        from dwm.datasets.common import CollateFnIgnoring
        collate = CollateFnIgnoring(keys=[])
        step = max(1, len(dataset) // args.clips)
        rows = []
        content = pipeline.pixel_content_ray_valid.reshape(-1)
        for index in range(0, len(dataset), step)[:args.clips]:
            batch = collate([dataset[index]])
            pipeline._prepare_nuscenes6_supervision(batch)
            frames = torch.tensor([0, args.gap, 2 * args.gap])
            halves = batch["vae_images"][:, frames].flatten(0, 1).to(device)
            lo, po = pipeline._frame_ray_origins(batch, frames)
            e2g = batch["ego_transforms"][0, frames, 0].to(device, dtype=torch.float32)
            decoded, static = [], []
            for k in range(3):
                g = pipeline.decoder_wrapper(
                    pipeline.encode_half_latents(halves[k:k + 1]),
                    pipeline.latent_directions.unsqueeze(0),
                    pipeline.pixel_directions.unsqueeze(0), None,
                    previous_latents=None, temporal_context_mask=0.0,
                    latent_origins=None if lo is None else lo[k:k + 1],
                    pixel_origins=None if po is None else po[k:k + 1],
                    camera_origin_scale=1.0)[0][0]
                frame = int(frames[k])
                owner = points_in_boxes(
                    g[:, 0:3], batch["box_centers"][0, frame].to(device),
                    batch["box_rotations"][0, frame].to(device),
                    batch["box_half_sizes"][0, frame].to(device),
                    valid=batch["box_valid"][0, frame], margin=args.box_margin)
                decoded.append((g, owner))
                static.append(g[(owner < 0) & content])
            target = pipeline._target_rgb_for_supervision(halves[1:2])
            g_t, owner_t = decoded[1]
            own_rgb, _, _, own_valid = pipeline._render_own_supervision(g_t)
            dynamic_t = g_t[(owner_t >= 0) & content]
            if dynamic_t.shape[0]:
                _, _, dyn_alpha, _ = pipeline._render_own_supervision(dynamic_t)
                static_pixels = dyn_alpha <= 0.3
            else:
                static_pixels = torch.ones_like(own_valid, dtype=torch.bool)
            to_t = [relative_ego_transform(e2g[s], e2g[1]) for s in (0, 2)]
            pooled = torch.cat([transform_gaussians(static[0], to_t[0]),
                                transform_gaussians(static[2], to_t[1])])
            single = transform_gaussians(static[0], to_t[0])
            row = {"clip_index": index, "scene_token": batch["scene_token"][0],
                   "psnr_own": masked_psnr(own_rgb, target, own_valid)}
            for name, scene in (("pooled", pooled), ("single", single)):
                rgb, _, alpha, valid = pipeline._render_own_supervision(scene)
                support = valid & static_pixels
                covered = support & (alpha > 0.5)
                row[f"psnr_{name}"] = masked_psnr(rgb, target, covered)
                row[f"coverage_{name}"] = float(covered.float().sum()
                                                / support.float().sum().clamp_min(1.0))
            rows.append(row)
            print("[eval4d]", json.dumps(row), flush=True)
        keys = [k for k in rows[0] if k.startswith(("psnr_", "coverage_"))]
        mean = {k: sum(r[k] for r in rows if not math.isnan(r[k]))
                / max(1, sum(1 for r in rows if not math.isnan(r[k]))) for k in keys}
        report = {"checkpoint": str(args.checkpoint), "clips": len(rows),
                  "gap_frames": args.gap, "mean": mean, "rows": rows,
                  "note": "held-out frame t never enters the pooled/single scenes"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"mean": mean, "clips": len(rows)}, indent=2), flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
