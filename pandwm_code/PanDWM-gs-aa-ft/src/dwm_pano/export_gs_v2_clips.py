"""v2 of export_gs_4d_clips.py (2026-10-17, report section 34): the same 4D renders for the 149 eval clips plus two
render-side fixes and a 3D-only near-rig view.  With the new options off (default) the 4D videos are those of
export_gs_4d_clips.py.
  * --edge-rel-h / --edge-rel-v > 0: drop the Gaussians of pixels at depth discontinuities of the decoder's range image
    (relative jump to a horizontal / vertical neighbour above the threshold), dilated by --edge-dilate pixels.  These
    pixels sit between foreground and background and show up as curtains / streaks away from the rig.  Applied to
    frame t and to the nearby-frame static fill.  The vertical threshold is separate because the far road changes
    range fast from row to row near the horizon.
  * --bg-fill: pixels still uncovered after the fill get a background: above the horizon row the mean colour of the
    frame's top panorama rows (sky), below it the mean of its bottom rows (ground), blended over 8 rows.  Cosmetic.
  * --views-3d: also write 3d_views.mp4 = frame t's own Gaussians only (no fill), same near-rig views.
  * --smooth-k KH,KW (v2b): component-wise KH x KW median, over the decoder's pixel grid, of every Gaussian's offset
    from its ray origin; scales follow the change of distance and the edge filter then reads the smoothed distances.
    Removes the row-to-row depth jitter that makes trunks, poles and fences wavy away from the rig.
  * --inpaint-all (v2b): inpaint every uncovered pixel (alpha < 0.5, cv2 Telea), not only the thin cracks of
    --fill-width.  Cosmetic, like --bg-fill.
  * --fill-mode pushpull (v2c): with --inpaint-all, fill the holes with a multiscale average of the covered pixels
    (push-pull, bilinear between levels) instead of cv2 Telea, which drags bright lamps and lane marks into long
    streaks across large holes (night clips).
  * --dac-weights (v3, 2026-10-18, report section 37): place every frame's Gaussians at the range DAC estimates on
    the VAE-decoded frame (dwm_pano_4d/dac_depth.py; needs --dac-root / --dac-config and dwm_pano_4d on the path):
    means move along the decoder's exact rays, xyz offset dropped, scales follow the range ratio, pixels without a
    DAC range below --dac-far-cut keep the decoder geometry.  The edge filter then reads the DAC range.
  * --vae-roundtrip (v3): decode each frame's latents to RGB with the VAE and encode them again (posterior mean)
    before the Gaussian decoder, i.e. give the decoder latents from the distribution it was trained on (VAE
    encodings of real images) instead of the video model's raw latents.
  * metadata v3 also records the decoder range image's roughness: median |log r(y+1) - log r(y)| (vertical) and
    |log r(x+1) - log r(x)| (horizontal) over all pixels, mean over frames; --dump-range writes frame 0's range.
  * --views-far (v2b): also write 3d_far_views.mp4 = frame t's own Gaussians from the two far cameras of
    export_gs_batch149.py (gs_novel_views.mp4: chase 4 m behind / 2.5 m up | +/-2 m sweep, 1280x640, 90 deg), with
    the same fixes.
Original header follows.

4D Gaussian demo (frame t + static fill from nearby frames) for the 149 eval clips.

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
from dwm_pano.export_gs_batch149 import cameras as far_cameras
from dwm_pano.export_gs_long_demo import pinhole
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
    p.add_argument("--edge-rel-h", type=float, default=0.0)
    p.add_argument("--edge-rel-v", type=float, default=0.0)
    p.add_argument("--edge-dilate", type=int, default=1)
    p.add_argument("--bg-fill", action="store_true")
    p.add_argument("--views-3d", action="store_true")
    p.add_argument("--smooth-k", default="", help="KH,KW median window of the Gaussian offsets; empty = off")
    p.add_argument("--inpaint-all", action="store_true")
    p.add_argument("--views-far", action="store_true")
    p.add_argument("--fill-mode", choices=["telea", "pushpull"], default="telea")
    p.add_argument("--dac-root", default="")
    p.add_argument("--dac-config", default="")
    p.add_argument("--dac-weights", default="")
    p.add_argument("--dac-far-cut", type=float, default=99.0)
    p.add_argument("--vae-roundtrip", action="store_true")
    p.add_argument("--dump-range", action="store_true")
    return p.parse_args()


def median2d(x, kh, kw):
    """(N, H, W) -> (N, H, W) median over a kh x kw window, replicate padding."""
    y = torch.nn.functional.pad(x.unsqueeze(1), (kw // 2, kw // 2, kh // 2, kh // 2), mode="replicate")
    y = y.unfold(2, kh, 1).unfold(3, kw, 1)
    return y.reshape(*y.shape[:4], -1).median(dim=-1).values.squeeze(1)


def smooth_means(g, origins, shape, kh, kw):
    """Median-smooth each Gaussian's offset from its ray origin over the (V, H, W) pixel grid (decoder order v h w).
    Returns the new Gaussians and the smoothed distance image (V, H, W)."""
    v, h, w = shape
    d = (g[:, 0:3].float() - origins).reshape(v, h, w, 3).permute(0, 3, 1, 2).reshape(v * 3, h, w)
    ds = median2d(d, kh, kw).reshape(v, 3, h, w).permute(0, 2, 3, 1).reshape(-1, 3)
    d = d.reshape(v, 3, h, w).permute(0, 2, 3, 1).reshape(-1, 3)
    dist = ds.norm(dim=-1)
    ratio = (dist / d.norm(dim=-1).clamp_min(1e-3)).clamp(0.5, 2.0)
    out = g.clone()
    out[:, 0:3] = (origins + ds).to(g.dtype)
    out[:, 11:14] = g[:, 11:14] * ratio[:, None].to(g.dtype)
    return out, dist.reshape(v, h, w)


def pushpull_fill(rgb, alpha, thr=0.5):
    """Fill uncovered pixels (alpha < thr) of a (3, H, W) image with a smooth multiscale average of the covered ones."""
    w = (alpha.float() >= thr).float()[None, None]
    c = rgb.float()[None] * w
    levels = [(c, w)]
    while max(c.shape[-2:]) > 1:                                      # down to the global mean
        c = torch.nn.functional.avg_pool2d(c, 2, ceil_mode=True)
        w = torch.nn.functional.avg_pool2d(w, 2, ceil_mode=True)
        levels.append((c, w))
    fill = None
    for c, w in reversed(levels):
        col = c / w.clamp_min(1e-6)
        if fill is not None:
            up = torch.nn.functional.interpolate(fill, size=c.shape[-2:], mode="bilinear", align_corners=False)
            col = w * col + (1.0 - w) * up
        fill = col
    return torch.where(alpha.float()[None] >= thr, rgb, fill[0].to(rgb.dtype))


@torch.no_grad()
def vae_roundtrip(pipe, latents):
    """(1, 2, C, h, w) latents -> VAE decode -> VAE encode (posterior mean), same scaling as encode_half_latents."""
    images = pipe.decode_half_latents(latents)                          # (1, 2, 3, H, W) in [0, 1]
    flat = images.flatten(0, 1) * 2.0 - 1.0
    posterior = pipe.vae.encode(flat.to(pipe.vae.dtype)).latent_dist
    out = (posterior.mean - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return out.unflatten(0, (1, 2)).to(latents.dtype)


def range_roughness(rng):
    """median |dlog r| between vertical and horizontal neighbours of a (V, H, W) range image."""
    lr = rng.float().clamp_min(1e-3).log()
    return (float((lr[:, 1:] - lr[:, :-1]).abs().median()), float((lr[:, :, 1:] - lr[:, :, :-1]).abs().median()))


def inpaint_holes(rgb, alpha, radius=5):
    """Inpaint every uncovered pixel (alpha < 0.5) of a (3, H, W) image in [0, 1]."""
    import cv2
    hole = (alpha.float().numpy() < 0.5).astype(np.uint8)
    if not hole.any():
        return rgb
    image = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    filled = cv2.inpaint(np.ascontiguousarray(image), hole, radius, cv2.INPAINT_TELEA)
    return torch.from_numpy(filled).permute(2, 0, 1).float() / 255.0


def edge_mask(rng, rel_h, rel_v, dilate):
    """(V, H, W) range image -> flat bool mask (V*H*W, decoder order v h w) of depth-discontinuity pixels."""
    r = rng.float().clamp_min(1e-3)
    m = torch.zeros_like(r, dtype=torch.bool)
    if rel_h > 0:
        e = (r[:, :, 1:] - r[:, :, :-1]).abs() / torch.minimum(r[:, :, 1:], r[:, :, :-1]) > rel_h
        m[:, :, 1:] |= e
        m[:, :, :-1] |= e
    if rel_v > 0:
        e = (r[:, 1:, :] - r[:, :-1, :]).abs() / torch.minimum(r[:, 1:, :], r[:, :-1, :]) > rel_v
        m[:, 1:, :] |= e
        m[:, :-1, :] |= e
    if dilate > 0 and bool(m.any()):
        k = 2 * dilate + 1
        m = torch.nn.functional.max_pool2d(m.float().unsqueeze(1), k, stride=1, padding=dilate).squeeze(1) > 0
    return m.reshape(-1)


def background(height, width, horizon, sky, ground, device):
    """(3, H, W) background: sky above the horizon row, ground below, linear blend over 8 rows."""
    rows = torch.arange(height, device=device, dtype=torch.float32)
    w = ((rows - (horizon - 4.0)) / 8.0).clamp(0.0, 1.0).view(1, height, 1)
    return (1.0 - w) * sky.view(3, 1, 1) + w * ground.view(3, 1, 1)


@torch.no_grad()
def render_clip(pipe, batch, infos, latents, args, device, out_dir):
    offsets = sorted(int(x) for x in args.window.split(",") if x.strip())
    views = [v for v in args.views.split(",") if v]
    n_frames = latents.shape[0]
    frames = min(n_frames, args.max_frames) if args.max_frames else n_frames
    pipe._prepare_nuscenes6_supervision(batch)
    world = ego_to_world(infos)
    content = getattr(pipe, "pixel_content_ray_valid", None)

    colours = {}
    smooth = tuple(int(k) for k in args.smooth_k.split(",")) if args.smooth_k else None
    dac_cache = {}                                                      # v3: frame -> DAC range on the grid
    rt_cache = {}                                                       # v3: frame -> round-tripped latents
    roughness = {}                                                      # v3: frame -> (vertical, horizontal)

    def decode(t):
        selector = torch.tensor([t], dtype=torch.long)
        lo, po = pipe._frame_ray_origins(batch, selector)
        lat_t = latents[t:t + 1].to(device, dtype=pipe.latent_directions.dtype)
        if args.vae_roundtrip:                                          # v3
            if t not in rt_cache:
                rt_cache[t] = vae_roundtrip(pipe, lat_t)
            lat_t = rt_cache[t]
        out = pipe.decoder_wrapper(
            lat_t,
            pipe.latent_directions.unsqueeze(0), pipe.pixel_directions.unsqueeze(0),
            None, previous_latents=None, temporal_context_mask=0.0,
            latent_origins=lo, pixel_origins=po, camera_origin_scale=1.0)
        g, rng = out[0][0], out[1][0]                                   # v2: keep the range image
        if t not in roughness:                                          # v3: on the decoder's own range
            roughness[t] = range_roughness(rng)
            if args.dump_range and t == 0:
                import cv2
                lr = rng.float().clamp(1.0, 100.0).log().cpu().numpy() / np.log(100.0)
                img = cv2.applyColorMap((255 * (1 - np.concatenate(list(lr), axis=0))).astype(np.uint8),
                                        cv2.COLORMAP_TURBO)
                cv2.imwrite(str(out_dir / "range_t0.png"), img)
        if args.dac is not None:                                        # v3: DAC range on the decoded frame
            from dwm_pano_4d.dac_depth import pano_from_halves, range_to_decoder_grid, relabel_gaussians
            if t not in dac_cache:
                rgb_t = pipe.decode_half_latents(latents[t:t + 1].to(device, dtype=pipe.latent_directions.dtype))[0]
                dac_cache[t] = range_to_decoder_grid(args.dac.pano_range(pano_from_halves(rgb_t)),
                                                     rng.shape[1], rng.shape[2], device)
            r_dac = dac_cache[t]
            grid_content = content.view(rng.shape).to(device) if content is not None and content.numel() == \
                rng.numel() else torch.ones_like(r_dac, dtype=torch.bool)
            valid = grid_content & (r_dac > 0.5) & (r_dac < args.dac_far_cut)
            dirs = pipe.pixel_directions.reshape(-1, 3).float()
            origins = po.reshape(-1, 3).to(device, torch.float32) if po is not None else torch.zeros_like(dirs)
            g, rng = relabel_gaussians(g.float(), rng.float(), r_dac, valid, origins, dirs)
        if args.bg_fill and t not in colours:                           # v2: sky / ground colours
            rgb = g[:, 3:6].reshape(*rng.shape, 3)
            colours[t] = (rgb[:, :8].reshape(-1, 3).mean(0), rgb[:, -8:].reshape(-1, 3).mean(0))
        if smooth:                                                      # v2b: median-smoothed geometry
            origins = po.reshape(-1, 3).to(g.device, torch.float32) if po is not None else \
                torch.zeros(g.shape[0], 3, device=g.device)
            g, rng = smooth_means(g, origins, tuple(rng.shape), *smooth)
        keep = torch.ones(g.shape[0], dtype=torch.bool, device=g.device)
        if content is not None and content.numel() == g.shape[0]:
            keep &= content.reshape(-1).to(g.device)
        if args.edge_rel_h > 0 or args.edge_rel_v > 0:                  # v2: depth-edge filter
            keep &= ~edge_mask(rng, args.edge_rel_h, args.edge_rel_v, args.edge_dilate).to(g.device)
        g = g[keep]
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

    def finish(rgb, alpha, t=None, K=None):
        if args.bg_fill and t in colours and K is not None:             # v2: background under the holes
            sky, ground = colours[t]
            rgb = torch.stack([x + (1.0 - a.clamp(0, 1))[None] * background(
                x.shape[1], x.shape[2], float(k[1, 2]), sky, ground, x.device)
                for x, a, k in zip(rgb, alpha, K)])
        out = rgb.cpu()
        if args.inpaint_all:                                            # v2b: every hole, not only cracks
            fill = pushpull_fill if args.fill_mode == "pushpull" else inpaint_holes
            out = torch.stack([fill(x, a) for x, a in zip(out, alpha.cpu())])
        elif args.fill_width:
            out = torch.stack([fill_cracks(x, a, args.fill_width) for x, a in zip(out, alpha.cpu())])
        return out

    compare_rows, view_rows, view3d_rows, far_rows = [], [], [], []
    K_far = torch.tensor(np.stack([pinhole(1280, 640, 90.0)] * 2), device=device, dtype=torch.float32)
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
        compare_rows.append(torch.cat([finish(rgb_c, a_c, t, Kc)[0], finish(rgb4, a4, t, Kc)[0]], dim=-1))
        rgb3, a3, rgbv, av = composite([VIEWS[v][1](t) for v in views], Kv, h_view)
        view_rows.append(torch.cat(list(finish(rgbv, av, t, Kv)), dim=-1))
        if args.views_3d:                                               # v2: frame t only, no fill
            view3d_rows.append(torch.cat(list(finish(rgb3, a3, t, Kv)), dim=-1))
        if args.views_far:                                              # v2b: batch149's far cameras, frame t
            c2w = torch.tensor(far_cameras(t), device=device, dtype=torch.float32)
            rgbf, _, af = pipe.render_targets(current.float(), target_c2w=c2w, target_intrinsics=K_far,
                                              width=1280, height=640)
            far_rows.append(torch.cat(list(finish(rgbf.clamp(0, 1), af, t, K_far)), dim=-1))
        colours.pop(t - 1, None)

    fps = float(batch["fps"][0])
    outputs = [("4d_clean_compare.mp4", compare_rows), ("4d_clean_views.mp4", view_rows)]
    if args.views_3d:
        outputs.append(("3d_views.mp4", view3d_rows))
    if args.views_far:
        outputs.append(("3d_far_views.mp4", far_rows))
    for name, rows in outputs:
        tmp = out_dir / (name[:-4] + ".tmp.mp4")
        save_tensor_to_video(str(tmp), "libx264", fps, rows)
        tmp.replace(out_dir / name)
    render_clip.last_roughness = [sum(v[i] for v in roughness.values()) / max(1, len(roughness)) for i in (0, 1)]
    render_clip.frame_roughness = {int(k): [round(v[0], 5), round(v[1], 5)] for k, v in sorted(roughness.items())}
    return frames, offsets, views


def main():
    args = parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    args.dac = None
    if args.dac_weights:                                                # v3
        from dwm_pano_4d.dac_depth import DacRanger
        args.dac = DacRanger(args.dac_root, args.dac_config, args.dac_weights, device)
        print(f"[gs-4d-clips] DAC loaded ({args.dac.config['model_name']})", flush=True)
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
                "v2": {"edge_rel_h": args.edge_rel_h, "edge_rel_v": args.edge_rel_v,
                       "edge_dilate": args.edge_dilate, "bg_fill": args.bg_fill, "views_3d": args.views_3d,
                       "smooth_k": args.smooth_k, "inpaint_all": args.inpaint_all, "views_far": args.views_far,
                       "fill_mode": args.fill_mode,
                       "dac_weights": args.dac_weights, "dac_far_cut": args.dac_far_cut,
                       "vae_roundtrip": args.vae_roundtrip,
                       "range_roughness_vh": getattr(render_clip, "last_roughness", None),
                       "range_roughness_by_frame": getattr(render_clip, "frame_roughness", None)},
            }, indent=2) + "\n")
            last = time.monotonic() - t0
            print(f"[gs-4d-clips] clip {index} done ({last:.0f}s)", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
