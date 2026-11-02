"""v3 review renders of the 149 eval clips (2026-10-19, report section 40).

The four videos of export_gs_v2_clips.py with the same names, views and finishing (depth-edge filter, max-scale
cap, nearby-frame static fill, --bg-fill sky / ground colours, push-pull hole fill), but the scene comes from
gs_scene_v3.SceneBuilder, so the v3 switches apply: --ownership (B1), --da3-relabel (C1), --sky-layer (B2).
With the switches off the scene is v2's.

--sky-layer also gives the pinhole review views a sky behind the Gaussians: along each pixel's ray direction, the
frame's own decoded pano where DA3 calls the pixel sky (inside the +/-16.7 deg band), and above the band a
continuation of the band's top rows (horizontally blurred) fading to the frame's mean sky colour 10 deg higher.

  <out-root>/clipNNN/4d_clean_compare.mp4   camera 3 m behind: per-frame 3D | 4D filled
  <out-root>/clipNNN/4d_clean_views.mp4     4D filled, near-rig views (chase, sweep, pan)
  <out-root>/clipNNN/3d_views.mp4           frame t only, near-rig views
  <out-root>/clipNNN/3d_far_views.mp4       frame t only, far chase 4 m behind / 2.5 m up | +/-2 m sweep
  <out-root>/clipNNN/clean4d_metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dwm.common
from dwm.utils.preview import save_tensor_to_video
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.export_gs_4d_clean import COMPARE_C2W, intrinsics
from dwm_pano.export_gs_batch149 import FRAMES
from dwm_pano.export_gs_batch149 import cameras as far_cameras
from dwm_pano.export_gs_clean_views import VIEWS
from dwm_pano.export_gs_long_demo import pinhole
from dwm_pano.export_gs_v2_clips import background, pushpull_fill
from dwm_pano.gs_geometry import PANO_LAT_HALF_RAD
from dwm_pano.gs_scene_v3 import SceneBuilder, SceneOptions, background_row_fill, composite, composite_depth


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
    p.add_argument("--max-frames", type=int, default=0, help="test: render only N frames")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--hfov-deg", type=float, default=70.0)
    p.add_argument("--window", default="-20,-15,-10,-5,5,10")
    p.add_argument("--edge-rel-h", type=float, default=0.1)
    p.add_argument("--edge-rel-v", type=float, default=0.3)
    p.add_argument("--ownership", action="store_true")
    p.add_argument("--da3-relabel", action="store_true")
    p.add_argument("--sky-layer", action="store_true")
    p.add_argument("--da3-mode", choices=["ray", "scale"], default="ray", help="scale = C1b (section 40.3)")
    p.add_argument("--fill-mode", choices=["pushpull", "bgrow"], default="pushpull",
                   help="bgrow = background row fill first, push-pull for the rest (B3, section 40.4)")
    p.add_argument("--drop-bottom-rows", type=int, default=0,
                   help="drop the Gaussians of the lowest N decoder rows (section 41.6, C2's band-edge line)")
    p.add_argument("--da3-src", default="")
    p.add_argument("--da3-anyview", default="")
    p.add_argument("--da3-metric", default="")
    return p.parse_args()


def ray_directions(c2w, K, width, height):
    """(V, 4, 4) OpenCV c2w and (V, 3, 3) K -> (V, H, W, 3) unit ray directions in the ego frame."""
    u = torch.arange(width, device=c2w.device, dtype=torch.float32) + 0.5
    v = torch.arange(height, device=c2w.device, dtype=torch.float32) + 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    pix = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)                         # (H, W, 3)
    cam = torch.einsum("vij,hwj->vhwi", torch.linalg.inv(K), pix)
    world = torch.einsum("vij,vhwj->vhwi", c2w[:, :3, :3], cam)
    return F.normalize(world, dim=-1)


class SkyEnv:
    """Sky colour along ray directions from one frame's decoded pano canvas and its DA3 sky mask."""

    def __init__(self, canvas, sky, fade_deg=10.0, blur_px=81):
        self.canvas, self.sky = canvas.float(), sky.float()                          # (3, H, Wc), (H, Wc)
        height, width = sky.shape
        self.height, self.width = height, width
        top = canvas[:, :6].mean(dim=1, keepdim=True)                                # (3, 1, Wc)
        pad = blur_px // 2
        kernel = torch.ones(3, 1, 1, blur_px, device=canvas.device) / blur_px
        self.top = F.conv2d(F.pad(top[None], (pad, pad, 0, 0), mode="circular"), kernel, groups=3)[0]
        sky_pixels = self.sky >= 0.5
        self.mean_sky = canvas[:, sky_pixels].mean(dim=1) if bool(sky_pixels.any()) else top.mean(dim=(1, 2))
        self.fade = math.radians(fade_deg)

    def __call__(self, dirs):
        """(V, H, W, 3) directions -> rgb (V, 3, H, W), mask (V, H, W)."""
        px, py, pz = -dirs[..., 1], dirs[..., 0], dirs[..., 2]
        azimuth = torch.atan2(px, py)
        elevation = torch.asin(pz.clamp(-1.0, 1.0))
        column = (azimuth + math.pi) / (2.0 * math.pi) * self.width
        row = (PANO_LAT_HALF_RAD - elevation) / (2.0 * PANO_LAT_HALF_RAD) * self.height
        gx = column / self.width * 2.0 - 1.0
        gy = (row / self.height * 2.0 - 1.0).clamp(-1.0, 1.0)
        grid = torch.stack([gx, gy], dim=-1)
        views = dirs.shape[0]
        canvas = self.canvas[None].expand(views, -1, -1, -1)
        rgb_band = F.grid_sample(canvas, grid, mode="bilinear", padding_mode="border", align_corners=False)
        sky_band = F.grid_sample(self.sky[None, None].expand(views, -1, -1, -1), grid, mode="bilinear",
                                 padding_mode="border", align_corners=False)[:, 0]
        top = F.grid_sample(self.top[None].expand(views, -1, -1, -1),
                            torch.stack([gx, torch.zeros_like(gx)], dim=-1), mode="bilinear",
                            padding_mode="border", align_corners=False)
        above = elevation > PANO_LAT_HALF_RAD
        w = ((elevation - PANO_LAT_HALF_RAD) / self.fade).clamp(0.0, 1.0)[:, None]
        rgb_above = (1.0 - w) * top + w * self.mean_sky.view(1, 3, 1, 1)
        in_band_sky = (elevation.abs() <= PANO_LAT_HALF_RAD) & (sky_band >= 0.5)
        mask = (above | in_band_sky).float()
        rgb = torch.where(above[:, None], rgb_above, rgb_band)
        return rgb, mask


@torch.no_grad()
def render_clip(pipe, batch, infos, latents, args, opts, device, out_dir, da3):
    scene = SceneBuilder(pipe, batch, infos, latents, opts, device, da3=da3)
    views = ("chase", "sweep", "pan")
    frames = min(FRAMES, args.max_frames) if args.max_frames else FRAMES
    K_cmp, h_cmp = intrinsics(args.width, args.hfov_deg, 14.0, 16.0)
    K_view, h_view = intrinsics(args.width, args.hfov_deg, 14.0, 7.0)
    Kc = torch.tensor(K_cmp[None], device=device, dtype=torch.float32)
    Kv = torch.tensor(np.stack([K_view] * len(views)), device=device, dtype=torch.float32)
    Kf = torch.tensor(np.stack([pinhole(1280, 640, 90.0)] * 2), device=device, dtype=torch.float32)

    def render(g, c2w, K, width, height):
        rgb, depth, alpha = pipe.render_targets(g.float(), target_c2w=c2w, target_intrinsics=K, width=width,
                                                height=height)
        return rgb.clamp(0, 1), alpha, depth

    def finish(rgb, alpha, t, K, c2w, width, height, env, depth=None):
        if env is not None:                                                       # B2: sky behind the Gaussians
            sky_rgb, sky_mask = env(ray_directions(c2w, K, width, height))
            rgb, alpha = composite(rgb, alpha, sky_rgb * sky_mask[:, None], sky_mask)
        sky, ground = scene.colours[t]
        rgb = torch.stack([x + (1.0 - a.clamp(0, 1))[None] * background(x.shape[1], x.shape[2], float(k[1, 2]),
                                                                       sky, ground, x.device)
                           for x, a, k in zip(rgb, alpha, K)])
        if args.fill_mode == "bgrow" and depth is not None:                       # B3
            pairs = [background_row_fill(x, a, d) for x, a, d in zip(rgb, alpha, depth)]
            rgb, alpha = torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])
        return torch.stack([pushpull_fill(x, a) for x, a in zip(rgb.cpu(), alpha.cpu())]).clamp(0, 1)

    rows = {"4d_clean_compare.mp4": [], "4d_clean_views.mp4": [], "3d_views.mp4": [], "3d_far_views.mp4": []}
    for t in range(frames):
        current = scene.decode(t).float()
        fill = scene.fill(t)
        env = None
        if opts.sky_layer:
            sky = (scene.da3_maps(t)["sky"].float() >= 0.3).float()
            env = SkyEnv(scene.canvas(t), pipe._unpadded_stitch_canvas(sky[None, :, None])[0, 0])
        for name, c2w_list, K, width, height in (
                ("compare", [COMPARE_C2W], Kc, args.width, h_cmp),
                ("views", [VIEWS[v][1](t) for v in views], Kv, args.width, h_view),
                ("far", list(far_cameras(t)), Kf, 1280, 640)):
            c2w = torch.tensor(np.stack(c2w_list), device=device, dtype=torch.float32)
            rgb_c, a_c, d_c = render(current, c2w, K, width, height)
            if name == "far":
                rows["3d_far_views.mp4"].append(torch.cat(list(finish(rgb_c, a_c, t, K, c2w, width, height, env,
                                                                      d_c)), dim=-1))
                continue
            if fill.shape[0]:
                rgb_f, a_f, d_f = render(fill, c2w, K, width, height)
                rgb4, a4 = composite(rgb_c, a_c, rgb_f, a_f)
                d4 = composite_depth(d_c, a_c, d_f, a_f)
            else:
                rgb4, a4, d4 = rgb_c, a_c, d_c
            if name == "compare":
                rows["4d_clean_compare.mp4"].append(torch.cat(
                    [finish(rgb_c, a_c, t, K, c2w, width, height, env, d_c)[0],
                     finish(rgb4, a4, t, K, c2w, width, height, env, d4)[0]], dim=-1))
            else:
                rows["4d_clean_views.mp4"].append(torch.cat(list(finish(rgb4, a4, t, K, c2w, width, height, env,
                                                                        d4)), dim=-1))
                rows["3d_views.mp4"].append(torch.cat(list(finish(rgb_c, a_c, t, K, c2w, width, height, env,
                                                                  d_c)), dim=-1))
    fps = float(batch["fps"][0])
    for name, frames_list in rows.items():
        tmp = out_dir / (name[:-4] + ".tmp.mp4")
        save_tensor_to_video(str(tmp), "libx264", fps, frames_list)
        tmp.replace(out_dir / name)
    return frames, scene.summary()


def main():
    args = parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    opts = SceneOptions(edge_rel_h=args.edge_rel_h, edge_rel_v=args.edge_rel_v,
                        window=tuple(int(x) for x in args.window.split(",") if x.strip()),
                        ownership=args.ownership, da3_relabel=args.da3_relabel, sky_layer=args.sky_layer,
                        da3_mode=args.da3_mode, drop_bottom_rows=args.drop_bottom_rows)
    da3 = None
    if args.da3_relabel or args.sky_layer:
        from dwm_pano_4d.da3_depth import Da3Ranger
        da3 = Da3Ranger(args.da3_src, args.da3_anyview, args.da3_metric, device)
    started = ensure_single_rank_process_group(device)
    try:
        config = load_config(args.config)
        torch.manual_seed(int(config["generator_seed"]))
        args.out_root.mkdir(parents=True, exist_ok=True)
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(args.out_root),
                                                      config=config, device=device, resume_from=None)
        pipe.vae.eval()
        pipe.decoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
        pipe.decoder.eval()
        pipe.decoder_wrapper.eval()
        pipe.rasterize_mode = "antialiased"
        pipe.active_rasterize_eps2d = 0.1
        install_strict_pano_hardseam_stub()
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl, fps_stride_tuples=[[10, 20]])
        cfg["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        by_key = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in
                  enumerate(zip(base.clips, base.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring
        job_start, last = time.monotonic(), 0.0
        for index in range(args.start, args.end):
            src = args.clips_root / f"clip{index:03d}"
            dst = args.out_root / f"clip{index:03d}"
            if (dst / "clean4d_metadata.json").is_file():
                continue
            if args.time_budget_s and last and time.monotonic() - job_start + 1.2 * last > args.time_budget_s:
                print(f"[gs-v3] budget reached before clip {index}", flush=True)
                break
            t0 = time.monotonic()
            summary = json.loads((src / "summary.json").read_text())
            ds_index = by_key[(summary["scene_token"], int(summary["anchor_timestamp_us"]))]
            batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [ds_index]),
                                                          batch_size=1, shuffle=False, num_workers=0,
                                                          collate_fn=CollateFnIgnoring(keys=[]))))
            latents = torch.load(src / "latents.pt", map_location="cpu", weights_only=True)["latents"]
            dst.mkdir(parents=True, exist_ok=True)
            frames, stats = render_clip(pipe, batch, base.clips[ds_index], latents, args, opts, device, dst, da3)
            (dst / "clean4d_metadata.json").write_text(json.dumps({
                "purpose": "v3 review renders (report section 40): v2 scene + v3 switches",
                "clip_index": index, "scene_token": summary["scene_token"],
                "anchor_timestamp_us": summary["anchor_timestamp_us"], "latents": str(src / "latents.pt"),
                "frames": frames, "checkpoint": str(args.checkpoint),
                "scene": {k: getattr(opts, k) for k in ("edge_rel_h", "edge_rel_v", "window", "ownership",
                                                        "da3_relabel", "da3_mode", "sky_layer", "static_range",
                                                        "static_min_opacity", "max_scale", "drop_bottom_rows")},
                "scene_stats": stats, "fill_mode": args.fill_mode,
            }, indent=2) + "\n")
            last = time.monotonic() - t0
            print(f"[gs-v3] clip {index} done ({last:.0f}s) {json.dumps(stats)}", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
