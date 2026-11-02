"""Training-free 4D Gaussian demo, version 2: hole filling from nearby frames.

The first 4D demo (``export_gs_4d_demo.py``) pooled static Gaussians from the
whole 19.5 s trajectory into one map.  Decoder depth is off by about 5 m, so
the frames disagree, and the multi-frame filter that removes the
disagreement leaves a sparse speckle.  This version keeps frame t as the
primary layer and uses other frames only where frame t has no coverage:

  image = render(frame t) + (1 - alpha(frame t)) * render(static of nearby frames)

"Nearby" is ``--window`` frames (default -20 .. +10, i.e. 2 s back and 1 s
ahead).  Each nearby frame is decoded in its own ego frame, its static part
(outside the padded 3D boxes, within ``--static-range`` m) is moved into the
world frame with the logged ego poses, and then into frame t's view.  The
ground behind and under the car, never seen by frame t, was seen by earlier
frames, so that is what the 4D layer fills.

Outputs, in a new directory (earlier demos are not touched):
  4d_clean_compare.mp4  camera 3 m behind, 1.8 m up: per-frame 3D | 4D filled
  4d_clean_views.mp4    4D filled from the near-rig views of export_gs_clean_views
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
from dwm_pano.export_gs_4d_demo import (
    boxes_in_ego,
    ego_to_world,
    inside_boxes,
    transform_gaussians,
)
from dwm_pano.export_gs_clean_views import RIG_HEIGHT, VIEWS, fill_cracks
from dwm_pano.export_gs_long_demo import CONTEXT, FRAMES, load_rollout_latents, look_at_c2w


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--scene-token", required=True)
    p.add_argument("--latent-source", choices=("rollout", "gt"), default="rollout")
    p.add_argument("--rollout-dir", type=Path)
    p.add_argument("--window", default="-20,-15,-10,-5,5,10",
                   help="frame offsets whose static Gaussians fill frame t's holes")
    p.add_argument("--static-range", type=float, default=40.0)
    p.add_argument("--static-min-opacity", type=float, default=0.3)
    p.add_argument("--max-scale", type=float, default=1.0)
    p.add_argument("--box-margin", type=float, default=0.3)
    p.add_argument("--views", default="chase,sweep,pan")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--hfov-deg", type=float, default=70.0)
    p.add_argument("--fill-width", type=int, default=61)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.latent_source == "rollout" and args.rollout_dir is None:
        p.error("--latent-source rollout needs --rollout-dir")
    return args


def intrinsics(width, hfov_deg, up_deg, down_deg):
    focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    up = focal * math.tan(math.radians(up_deg))
    height = int(round(up + focal * math.tan(math.radians(down_deg))))
    height += height % 2
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, up], [0.0, 0.0, 1.0]]), height


# Higher, farther camera for the 3D-vs-4D comparison: here per-frame 3D shows
# the unobserved ground behind the car, which earlier frames did observe.
COMPARE_C2W = look_at_c2w([-3.0, 0.0, RIG_HEIGHT + 0.2], [15.0, 0.0, RIG_HEIGHT - 1.4])


@torch.no_grad()
def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    offsets = sorted(int(x) for x in args.window.split(",") if x.strip())
    views = [v for v in args.views.split(",") if v]
    config = load_config(args.config)
    torch.manual_seed(int(config["generator_seed"]))
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    started = ensure_single_rank_process_group(device)
    try:
        pipe = dwm.common.create_instance_from_config(
            config["pipeline"], output_path=str(args.out), config=config,
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
        matches = [i for i, a in enumerate(base.clip_audit) if a["scene_token"] == args.scene_token]
        if len(matches) != 1:
            raise ValueError(f"scene token has {len(matches)} {FRAMES}-frame clips")
        infos = base.clips[matches[0]]
        from dwm.datasets.common import CollateFnIgnoring
        batch = next(iter(torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, matches), batch_size=1, shuffle=False,
            num_workers=0, collate_fn=CollateFnIgnoring(keys=[]))))
        pipe._prepare_nuscenes6_supervision(batch)
        images = batch["vae_images"]
        encode = lambda starts: torch.cat([pipe.encode_half_latents(
            images[0, s:s + 2].to(device)) for s in starts], dim=0)
        if args.latent_source == "gt":
            latents = encode(range(0, FRAMES, 2))[:FRAMES].cpu()
        else:
            latents = load_rollout_latents(args.rollout_dir, encode(range(0, CONTEXT, 2))[:CONTEXT])

        world = ego_to_world(infos)
        content = getattr(pipe, "pixel_content_ray_valid", None)
        frames = min(FRAMES, args.max_frames) if args.max_frames else FRAMES

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
            """Static, in-range Gaussians of frame s in the world frame (cached)."""
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

        def render(g, c2w_ego_list, K, height, t):
            """Render frame-t-ego cameras; g is in frame t's ego frame."""
            c2w = torch.tensor(np.stack(c2w_ego_list), device=device, dtype=torch.float32)
            rgb, _, alpha = pipe.render_targets(g.float(), target_c2w=c2w, target_intrinsics=K,
                                                width=args.width, height=height)
            return rgb.clamp(0, 1), alpha

        compare_rows, view_rows = [], []
        to_world = lambda t: torch.as_tensor(world[t], device=device, dtype=torch.float32)
        for t in range(frames):
            for s in list(static_cache):
                if s < t + offsets[0]:
                    del static_cache[s]
            current = decode(t).float()
            neighbours = [static_world(t + o) for o in offsets if 0 <= t + o < FRAMES]
            world_to_t = torch.linalg.inv(to_world(t))
            fill = transform_gaussians(torch.cat(neighbours).float(), world_to_t) \
                if neighbours else current[:0]

            def composite(c2w_list, K, height):
                rgb_c, a_c = render(current, c2w_list, K, height, t)
                rgb_f, a_f = render(fill, c2w_list, K, height, t) if fill.shape[0] else \
                    (torch.zeros_like(rgb_c), torch.zeros_like(a_c))
                rgb = rgb_c + (1.0 - a_c[:, None]) * rgb_f
                alpha = a_c + (1.0 - a_c) * a_f
                return rgb_c, a_c, rgb, alpha

            def finish(rgb, alpha):
                out = rgb.cpu()
                if args.fill_width:
                    out = torch.stack([fill_cracks(x, a, args.fill_width)
                                       for x, a in zip(out, alpha.cpu())])
                return out

            rgb_c, a_c, rgb4, a4 = composite([COMPARE_C2W], Kc, h_cmp)
            compare_rows.append(torch.cat([finish(rgb_c, a_c)[0], finish(rgb4, a4)[0]], dim=-1))
            _, _, rgbv, av = composite([VIEWS[v][1](t) for v in views], Kv, h_view)
            view_rows.append(torch.cat(list(finish(rgbv, av)), dim=-1))
            if t % 25 == 0:
                print(f"[gs-4d-clean] frame {t}/{frames}", flush=True)

        fps = float(batch["fps"][0])
        save_tensor_to_video(str(args.out / "4d_clean_compare.mp4"), "libx264", fps, compare_rows)
        save_tensor_to_video(str(args.out / "4d_clean_views.mp4"), "libx264", fps, view_rows)
        (args.out / "clean4d_metadata.json").write_text(json.dumps({
            "purpose": "training-free 4D demo v2: frame t + static Gaussians of nearby "
                       "frames filling only where frame t has no coverage",
            "scene_token": args.scene_token, "latent_source": args.latent_source,
            "rollout_dir": str(args.rollout_dir) if args.rollout_dir else None,
            "window_frame_offsets": offsets, "static_range_m": args.static_range,
            "static_min_opacity": args.static_min_opacity, "max_scale_m": args.max_scale,
            "compare_columns": ["per-frame 3D", "4D: per-frame 3D + nearby-frame static fill"],
            "compare_camera": "3 m behind, 0.2 m above the rig",
            "views": [VIEWS[v][0] for v in views], "crack_fill_px": args.fill_width,
            "frames": frames, "checkpoint": str(args.checkpoint),
        }, indent=2) + "\n")
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
