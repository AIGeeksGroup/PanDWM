"""4D Gaussian demo (frame t + static fill from nearby frames) for the 149 eval clips.

Same rendering as ``export_gs_4d_clean.py`` (long demos; that file is untouched), with the clip
loading of ``export_gs_clean_views.py --clips-root``: each clip is a 35-frame window whose latents
live in ``<clips-root>/clipNNN/latents.pt``.  Writes, per clip, into a new directory:

  <out-root>/clipNNN/4d_clean_compare.mp4   camera 3 m behind: per-frame 3D | 4D filled
  <out-root>/clipNNN/4d_clean_views.mp4     4D filled, near-rig views (chase, sweep, pan)
  <out-root>/clipNNN/clean4d_metadata.json

Finished clips are skipped, so a job can be resubmitted; ``--time-budget-s`` stops before a clip
that would not fit.
"""

from __future__ import annotations

import argparse
import json
import time
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
from dwm_pano.export_gs_4d_clean import COMPARE_C2W, intrinsics
from dwm_pano.export_gs_4d_demo import boxes_in_ego, ego_to_world, inside_boxes, transform_gaussians
from dwm_pano.export_gs_batch149 import FRAMES
from dwm_pano.export_gs_clean_views import VIEWS, fill_cracks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--clips-root", required=True, type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=149)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--time-budget-s", type=float, default=0.0)
    p.add_argument("--window", default="-20,-15,-10,-5,5,10")
    p.add_argument("--static-range", type=float, default=40.0)
    p.add_argument("--static-min-opacity", type=float, default=0.3)
    p.add_argument("--max-scale", type=float, default=1.0)
    p.add_argument("--box-margin", type=float, default=0.3)
    p.add_argument("--views", default="chase,sweep,pan")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--hfov-deg", type=float, default=70.0)
    p.add_argument("--fill-width", type=int, default=61)
    p.add_argument("--max-frames", type=int, default=0, help="test: render only N frames")
    return p.parse_args()


@torch.no_grad()
def render_clip(pipe, batch, infos, latents, args, device, out_dir):
    offsets = sorted(int(x) for x in args.window.split(",") if x.strip())
    views = [v for v in args.views.split(",") if v]
    n_frames = latents.shape[0]
    frames = min(n_frames, args.max_frames) if args.max_frames else n_frames
    pipe._prepare_nuscenes6_supervision(batch)
    world = ego_to_world(infos)
    content = getattr(pipe, "pixel_content_ray_valid", None)

    def decode(t):
        selector = torch.tensor([t], dtype=torch.long)
        lo, po = pipe._frame_ray_origins(batch, selector)
        g = pipe.decoder_wrapper(
            latents[t:t + 1].to(device, dtype=pipe.latent_directions.dtype),
            pipe.latent_directions.unsqueeze(0), pipe.pixel_directions.unsqueeze(0),
            None, previous_latents=None, temporal_context_mask=0.0,
            latent_origins=lo, pixel_origins=po, camera_origin_scale=1.0)[0][0]
        if content is not None and content.numel() == g.shape[0]:
            g = g[content.reshape(-1).to(g.device)]
        return g[g[:, 11:14].amax(dim=-1) <= args.max_scale]

    static_cache = {}

    def static_world(s):
        if s not in static_cache:
            g = decode(s)
            c, r, h = boxes_in_ego(infos[s])
            dynamic = inside_boxes(g[:, 0:3], c, r, h, args.box_margin)
            keep = (~dynamic & (torch.linalg.norm(g[:, 0:2], dim=-1) <= args.static_range)
                    & (g[:, 6] >= args.static_min_opacity))
            static_cache[s] = transform_gaussians(g[keep], world[s])
        return static_cache[s]

    K_cmp, h_cmp = intrinsics(args.width, args.hfov_deg, 14.0, 16.0)
    K_view, h_view = intrinsics(args.width, args.hfov_deg, 14.0, 7.0)
    Kc = torch.tensor(K_cmp[None], device=device, dtype=torch.float32)
    Kv = torch.tensor(np.stack([K_view] * len(views)), device=device, dtype=torch.float32)

    def render(g, c2w_list, K, height):
        c2w = torch.tensor(np.stack(c2w_list), device=device, dtype=torch.float32)
        rgb, _, alpha = pipe.render_targets(g.float(), target_c2w=c2w, target_intrinsics=K,
                                            width=args.width, height=height)
        return rgb.clamp(0, 1), alpha

    def finish(rgb, alpha):
        out = rgb.cpu()
        if args.fill_width:
            out = torch.stack([fill_cracks(x, a, args.fill_width) for x, a in zip(out, alpha.cpu())])
        return out

    compare_rows, view_rows = [], []
    to_world = lambda t: torch.as_tensor(world[t], device=device, dtype=torch.float32)
    for t in range(frames):
        for s in list(static_cache):
            if s < t + offsets[0]:
                del static_cache[s]
        current = decode(t).float()
        neighbours = [static_world(t + o) for o in offsets if 0 <= t + o < n_frames]
        fill = transform_gaussians(torch.cat(neighbours).float(), torch.linalg.inv(to_world(t))) \
            if neighbours else current[:0]

        def composite(c2w_list, K, height):
            rgb_c, a_c = render(current, c2w_list, K, height)
            rgb_f, a_f = render(fill, c2w_list, K, height) if fill.shape[0] else \
                (torch.zeros_like(rgb_c), torch.zeros_like(a_c))
            return rgb_c, a_c, rgb_c + (1.0 - a_c[:, None]) * rgb_f, a_c + (1.0 - a_c) * a_f

        rgb_c, a_c, rgb4, a4 = composite([COMPARE_C2W], Kc, h_cmp)
        compare_rows.append(torch.cat([finish(rgb_c, a_c)[0], finish(rgb4, a4)[0]], dim=-1))
        _, _, rgbv, av = composite([VIEWS[v][1](t) for v in views], Kv, h_view)
        view_rows.append(torch.cat(list(finish(rgbv, av)), dim=-1))

    fps = float(batch["fps"][0])
    for name, rows in (("4d_clean_compare.mp4", compare_rows), ("4d_clean_views.mp4", view_rows)):
        tmp = out_dir / (name[:-4] + ".tmp.mp4")
        save_tensor_to_video(str(tmp), "libx264", fps, rows)
        tmp.replace(out_dir / name)
    return frames, offsets, views


def main():
    args = parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    started = ensure_single_rank_process_group(device)
    try:
        config = load_config(args.config)
        torch.manual_seed(int(config["generator_seed"]))
        args.out_root.mkdir(parents=True, exist_ok=True)
        pipe = dwm.common.create_instance_from_config(
            config["pipeline"], output_path=str(args.out_root), config=config,
            device=device, resume_from=None)
        pipe.vae.eval()
        pipe.decoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
        pipe.decoder.eval()
        pipe.decoder_wrapper.eval()
        pipe.rasterize_mode = "antialiased"
        pipe.active_rasterize_eps2d = 0.1
        install_strict_pano_hardseam_stub()

        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl,
                                   fps_stride_tuples=[[10, 20]])
        cfg["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        by_key = {(a["scene_token"], int(c[0]["timestamp"])): i
                  for i, (c, a) in enumerate(zip(base.clips, base.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring

        job_start, last = time.monotonic(), 0.0
        for index in range(args.start, args.end):
            src = args.clips_root / f"clip{index:03d}"
            dst = args.out_root / f"clip{index:03d}"
            if (dst / "clean4d_metadata.json").is_file():
                continue
            if not (src / "latents.pt").is_file():
                print(f"[gs-4d-clips] missing clip {index}", flush=True)
                continue
            if args.time_budget_s and last and \
                    time.monotonic() - job_start + 1.2 * last > args.time_budget_s:
                print(f"[gs-4d-clips] budget reached before clip {index}", flush=True)
                break
            t0 = time.monotonic()
            summary = json.loads((src / "summary.json").read_text())
            ds_index = by_key[(summary["scene_token"], int(summary["anchor_timestamp_us"]))]
            batch = next(iter(torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset, [ds_index]), batch_size=1, shuffle=False,
                num_workers=0, collate_fn=CollateFnIgnoring(keys=[]))))
            latents = torch.load(src / "latents.pt", map_location="cpu", weights_only=True)["latents"]
            dst.mkdir(parents=True, exist_ok=True)
            frames, offsets, views = render_clip(pipe, batch, base.clips[ds_index], latents,
                                                 args, device, dst)
            (dst / "clean4d_metadata.json").write_text(json.dumps({
                "purpose": "training-free 4D demo v2 on an eval clip: frame t + static Gaussians "
                           "of nearby frames filling only where frame t has no coverage",
                "clip_index": index, "scene_token": summary["scene_token"],
                "anchor_timestamp_us": summary["anchor_timestamp_us"],
                "latents": str(src / "latents.pt"), "window_frame_offsets": offsets,
                "static_range_m": args.static_range, "static_min_opacity": args.static_min_opacity,
                "max_scale_m": args.max_scale,
                "compare_columns": ["per-frame 3D", "4D: per-frame 3D + nearby-frame static fill"],
                "views": [VIEWS[v][0] for v in views], "crack_fill_px": args.fill_width,
                "frames": frames, "checkpoint": str(args.checkpoint),
            }, indent=2) + "\n")
            last = time.monotonic() - t0
            print(f"[gs-4d-clips] clip {index} done ({last:.0f}s)", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
