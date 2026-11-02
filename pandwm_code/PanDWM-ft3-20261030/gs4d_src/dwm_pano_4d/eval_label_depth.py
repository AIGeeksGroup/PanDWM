"""Can better depth fix the off-rig GS renders?  (2026-10-17, report section 36; the user chose option A.)

Feasibility test on labelled TRAINING frames (the G2 depth subset p2).  The same decoded Gaussians are placed either
at the decoder's own predicted range ("pred") or at the DAC depth label ("label": means moved along each pixel's ray
to the label range with the xyz offset dropped, scales multiplied by label / pred; pixels without a label below
--far-cut keep the decoder geometry).  Both are scored like evaluate_4d.py: frames (a, t, b) = (0, gap, 2 gap),
frame t rendered at its own six cameras from the static Gaussians of a ("single") or a + b ("pooled"), static
covered pixels only, plus the round trip "own".  The "*_edge" variants also drop the depth-edge pixels exactly like
export_gs_v2_clips.py (0.1 / 0.3, dilate 1, on that variant's range image).

For the first --video-clips clips it also renders the review views of frame t's own Gaussians, top = pred,
bottom = label: views_*.mp4 = near-rig chase | sweep | pan (960 x 256 each), far_*.mp4 = export_gs_batch149's far
chase | sweep (1280 x 640 each); *_raw = no fixes (holes black), *_v2 = edge filter + push-pull hole fill.

The labels come with the pano data (HF dataset zeyuren2002/nuscene-360_train, source-camera z); the decoder trained
on these scenes, so "pred" is if anything better here than on generated frames.

v2 (2026-10-18, report section 37): with --dac-weights, adds "dac" / "dac_edge": the range estimated by DAC
(dac_depth.py) on the VAE round trip of the frame (encode + decode, what a generated frame looks like to DAC), used
as the range along the exact rays; DAC on the original 600 x 9600 jpg is scored against the label too (geometry
stats only).  Videos then have three rows: pred / label / dac.

v3 (2026-10-19, report section 40): with --da3-anyview, adds "da3" / "da3_edge": Depth Anything 3 (da3_depth.py)
on the VAE round trip, the six cameras of frames (0, gap, 2 gap) reconstructed together with their known
intrinsics, rig extrinsics and ego poses; the ground-plane scale is the primary range (section 40.1; the pose and
metric-model scales and DA3 on the original, not round-tripped, frame are scored against the label too).  Pixels DA3 calls sky keep the decoder
geometry, like pixels without a DA3 range below --far-cut.  Video frames use the window (t - gap, t, t + gap).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the label files' compression filter)
import numpy as np
import torch
import torch.nn.functional as F

import dwm.common
from dwm.utils.preview import save_tensor_to_video
from dwm_pano.datasets.pano_projection import PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE
from dwm_pano.evaluate_gs import (
    build_dataset_config,
    ensure_single_rank_process_group,
    install_strict_pano_hardseam_stub,
    load_config,
)
from dwm_pano.export_gs_4d_clean import intrinsics
from dwm_pano.export_gs_batch149 import cameras as far_cameras
from dwm_pano.export_gs_clean_views import VIEWS
from dwm_pano.export_gs_long_demo import pinhole
from dwm_pano.export_gs_v2_clips import edge_mask, pushpull_fill
from dwm_pano_4d.evaluate_4d import masked_psnr
from dwm_pano_4d.geometry import points_in_boxes, relative_ego_transform, transform_gaussians

VARIANTS = ("pred", "pred_edge", "label", "label_edge")
DAC_VARIANTS = ("dac", "dac_edge")
DA3_VARIANTS = ("da3", "da3_edge")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--clips", type=int, default=25)
    p.add_argument("--gap", type=int, default=4)
    p.add_argument("--box-margin", type=float, default=0.3)
    p.add_argument("--far-cut", type=float, default=99.0)
    p.add_argument("--resize", choices=["nearest", "bilinear"], default="nearest",
                   help="label 600x9600 -> decoder grid; bilinear (the training loader) blends depths across edges")
    p.add_argument("--video-clips", type=int, default=3)
    p.add_argument("--video-frames", type=int, default=20)
    p.add_argument("--max-scale", type=float, default=1.0)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--video-dir", required=True, type=Path)
    p.add_argument("--dac-root", default="")
    p.add_argument("--dac-config", default="")
    p.add_argument("--dac-weights", default="")
    p.add_argument("--da3-src", default="", help="Depth-Anything-3 repo src/ (v3)")
    p.add_argument("--da3-anyview", default="", help="DA3-LARGE-1.1 folder (config.json + model.safetensors)")
    p.add_argument("--da3-metric", default="", help="DA3METRIC-LARGE folder")
    p.add_argument("--da3-scale", choices=["ground", "pose", "metric"], default="ground",
                   help="DA3 scale used to place the Gaussians (section 40.1: ground)")
    p.add_argument("--da3-relabel-mode", choices=["ray", "scale"], default="ray",
                   help="ray: move along the pixel ray, offset dropped (sections 36-40.2); scale: about the ray "
                        "origin, footprint kept (C1b, section 40.3)")
    return p.parse_args()


def band_bias(a, b, mask, bands=4):
    """median log(a / b) per horizontal band of the (V, H, W) decoder grid (top to bottom)."""
    out = {}
    height = a.shape[1]
    for i in range(bands):
        rows = slice(i * height // bands, (i + 1) * height // bands)
        d = (a[:, rows].clamp_min(1e-3).log() - b[:, rows].clamp_min(1e-3).log())[mask[:, rows]]
        out[f"band{i}"] = float(d.median()) if d.numel() else float("nan")
    return out


def log_stats(prefix, a, b, mask):
    """mean / median |log a - log b| and median log(a / b) over mask."""
    d = (a.clamp_min(1e-3).log() - b.clamp_min(1e-3).log())[mask]
    if d.numel() == 0:
        return {f"{prefix}_log_l1": float("nan"), f"{prefix}_log_median": float("nan"),
                f"{prefix}_log_bias": float("nan")}
    return {f"{prefix}_log_l1": float(d.abs().mean()), f"{prefix}_log_median": float(d.abs().median()),
            f"{prefix}_log_bias": float(d.median())}


def load_label(pipe, batch, frame, mode, device):
    """DAC label of one frame as Euclidean range on the decoder grid (2, H, W) + validity, like _load_frame_ranges."""
    cfg = pipe.training_config
    ts = int(batch["frame_timestamps_us"][0, frame])
    with h5py.File(os.path.join(cfg["range_dir"], f"{ts}.h5"), "r") as handle:
        z = np.asarray(handle["depth"], dtype=np.float32)
    valid_full = torch.from_numpy(z > 0.0).float()
    if pipe.half_range_correction is None:
        full = torch.from_numpy((z * pipe.range_correction).astype(np.float32))
    else:
        full = torch.from_numpy(z)
    size = (cfg["half_height"], cfg["half_width"])
    ranges, valids = [], []
    for half_index, (c0, c1) in enumerate((PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE)):
        src, sv = full[None, None, :, c0:c1], valid_full[None, None, :, c0:c1]
        if mode == "nearest":
            half = F.interpolate(src, size=size, mode="nearest-exact")
            hv = F.interpolate(sv, size=size, mode="nearest-exact") > 0.5
        else:
            weighted = F.interpolate(src * sv, size=size, mode="bilinear", antialias=True)
            weight = F.interpolate(sv, size=size, mode="bilinear", antialias=True)
            hv = weight >= cfg.get("range_min_valid_fraction", 0.5)
            half = weighted / weight.clamp_min(1e-6)
        if pipe.half_range_correction is not None:
            half = half * pipe.half_range_correction_cpu[half_index][None, None]
        ranges.append(half[0, 0])
        valids.append(hv[0, 0])
    return torch.stack(ranges).to(device), torch.stack(valids).to(device)


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config)
    torch.manual_seed(int(config["generator_seed"]))
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    started = ensure_single_rank_process_group(device)
    try:
        pipe = dwm.common.create_instance_from_config(
            config["pipeline"], output_path=str(args.output.parent), config=config, device=device,
            resume_from=None)
        pipe.vae.eval()
        pipe.decoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
        pipe.decoder.eval()
        pipe.decoder_wrapper.eval()
        install_strict_pano_hardseam_stub()
        dcfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl,
                                    fps_stride_tuples=[[10, 20]])
        dcfg["base_dataset"]["sequence_length"] = max(2 * args.gap + 1, args.video_frames)
        dataset = dwm.common.create_instance_from_config(dcfg)
        from dwm.datasets.common import CollateFnIgnoring
        collate = CollateFnIgnoring(keys=[])
        step = max(1, len(dataset) // args.clips)
        indices = list(range(0, len(dataset), step))[:args.clips]
        print(f"[labeldepth] dataset {len(dataset)} clips, using {len(indices)}", flush=True)

        K_view, h_view = intrinsics(960, 70.0, 14.0, 7.0)
        views = ("chase", "sweep", "pan")
        Kv = torch.tensor(np.stack([K_view] * len(views)), device=device, dtype=torch.float32)
        Kf = torch.tensor(np.stack([pinhole(1280, 640, 90.0)] * 2), device=device, dtype=torch.float32)

        dac = None
        if args.dac_weights:
            from dwm_pano_4d.dac_depth import DacRanger, pano_from_halves, range_to_decoder_grid, relabel_gaussians
            dac = DacRanger(args.dac_root, args.dac_config, args.dac_weights, device)
            print(f"[labeldepth] DAC loaded ({dac.config['model_name']})", flush=True)
        da3 = None
        if args.da3_anyview:                                               # v3
            from dwm_pano_4d.da3_depth import Da3Ranger
            from dwm_pano_4d.dac_depth import relabel_gaussians
            da3 = Da3Ranger(args.da3_src, args.da3_anyview, args.da3_metric, device)
            print(f"[labeldepth] DA3 loaded ({da3.anyview_name} + {da3.metric_name})", flush=True)
        variants_all = VARIANTS + (DAC_VARIANTS if dac is not None else ()) + (DA3_VARIANTS if da3 is not None else ())
        rgb_dir = Path(dcfg["base_dataset"]["rgb_dir"])

        rows = []
        for n, index in enumerate(indices):
            batch = collate([dataset[index]])
            pipe._prepare_nuscenes6_supervision(batch)
            dirs = pipe.pixel_directions.reshape(-1, 3).float()
            content = pipe.pixel_content_ray_valid.reshape(-1)
            e2g = batch["ego_transforms"][0, :, 0].to(device, dtype=torch.float32)
            da3_cache = {}                                                 # v3: frame -> DA3 maps
            clip_frames = batch["vae_images"].shape[1]

            def da3_window(frame_list, store, round_trip=True):
                """DA3 on the given frames together; caches the maps of the frames in `store`."""
                halves = batch["vae_images"][0, frame_list].to(device)
                if round_trip:
                    halves = torch.stack([pipe.decode_half_latents(pipe.encode_half_latents(h[None]))[0]
                                          for h in halves])
                canvases = pipe._unpadded_stitch_canvas(halves.float())
                res = da3.frame_ranges(pipe, canvases, batch["ego_transforms"][0, frame_list, 0])
                out = {}
                for i, f in enumerate(frame_list):
                    if f in store:
                        out[f] = {name: res[name][i] for name in ("range_pose", "range_metric", "range_ground",
                                                                     "sky", "conf")}
                        out[f].update({name: res[name] for name in ("scale_pose", "scale_metric", "scale_ground",
                                                                       "pose_over_metric", "ground_over_metric")})
                return out

            def da3_maps(t):
                if t not in da3_cache:
                    window = sorted({max(0, t - args.gap), t, min(clip_frames - 1, t + args.gap)})
                    da3_cache.update(da3_window(window, {t}))
                return da3_cache[t]

            if da3 is not None:                                            # the scored frames, together
                scored = [0, args.gap, 2 * args.gap]
                da3_cache.update(da3_window(scored, set(scored)))

            def decode(t):
                """variant -> (Gaussians, pixel keep mask over the decoder grid); plus geometry stats."""
                sel = torch.tensor([t])
                lo, po = pipe._frame_ray_origins(batch, sel)
                half = batch["vae_images"][:, sel].flatten(0, 1).to(device)
                enc = pipe.encode_half_latents(half)
                out = pipe.decoder_wrapper(
                    enc, pipe.latent_directions.unsqueeze(0),
                    pipe.pixel_directions.unsqueeze(0), None, previous_latents=None,
                    temporal_context_mask=0.0, latent_origins=lo, pixel_origins=po, camera_origin_scale=1.0)
                g, rng = out[0][0].float(), out[1][0].float()
                latents = enc
                origins = po.reshape(-1, 3).to(device, torch.float32) if po is not None else torch.zeros_like(dirs)
                label, valid = load_label(pipe, batch, t, args.resize, device)
                use = (valid & (label > 0.5) & (label < args.far_cut)).reshape(-1) & content
                r_pred, r_lab = rng.reshape(-1), label.reshape(-1)
                g_lab = g.clone()
                g_lab[use, 0:3] = origins[use] + dirs[use] * r_lab[use, None]
                ratio = (r_lab / r_pred.clamp_min(1e-3)).clamp(0.25, 4.0)
                g_lab[use, 11:14] = g[use, 11:14] * ratio[use, None]
                rng_lab = torch.where(use.view_as(rng), label, rng)
                keep_pred = ~edge_mask(rng, 0.1, 0.3, 1).to(device)
                keep_lab = ~edge_mask(rng_lab, 0.1, 0.3, 1).to(device)
                every = torch.ones_like(keep_pred)
                offset = (g[:, 0:3] - origins - dirs * r_pred[:, None]).norm(dim=-1)[content]
                rel_offset = offset / r_pred[content].clamp_min(1e-3)
                log_err = (r_pred[use].clamp_min(1e-3).log() - r_lab[use].clamp_min(1e-3).log()).abs()
                stats = {"label_use_fraction": float(use.float().sum() / content.float().sum().clamp_min(1)),
                         "pred_vs_label_log_l1": float(log_err.mean()) if log_err.numel() else float("nan"),
                         "pred_vs_label_log_median": float(log_err.median()) if log_err.numel() else float("nan"),
                         "xyz_offset_median_m": float(offset.median()),
                         "xyz_offset_p95_m": float(offset.quantile(0.95)) if offset.numel() < 2 ** 24 else
                         float(offset[torch.randperm(offset.numel(), device=device)[:2 ** 20]].quantile(0.95)),
                         "xyz_offset_rel_median": float(rel_offset.median()),
                         "edge_fraction_pred": float(1 - keep_pred.float().mean()),
                         "edge_fraction_label": float(1 - keep_lab.float().mean())}
                result = {"pred": (g, every), "pred_edge": (g, keep_pred),
                          "label": (g_lab, every), "label_edge": (g_lab, keep_lab)}
                if dac is not None:
                    shape = tuple(rng.shape)
                    rt = pipe.decode_half_latents(latents)[0]                  # (2, 3, H, W) in [0, 1]
                    r_dac = range_to_decoder_grid(dac.pano_range(pano_from_halves(rt)), shape[1], shape[2], device)
                    dac_valid = content.view(shape) & (r_dac > 0.5) & (r_dac < args.far_cut)
                    g_dac, rng_dac = relabel_gaussians(g, rng, r_dac, dac_valid, origins, dirs)
                    keep_dac = ~edge_mask(rng_dac, 0.1, 0.3, 1).to(device)
                    result["dac"], result["dac_edge"] = (g_dac, every), (g_dac, keep_dac)
                    lab_mask = use.view(shape)
                    stats.update(log_stats("dac_vae_vs_label", r_dac, label, lab_mask))
                    stats.update(log_stats("pred_vs_label", rng, label, lab_mask))
                    if t == args.gap:                                          # the scored frame: DAC on the jpg too
                        import cv2
                        ts = int(batch["frame_timestamps_us"][0, t])
                        jpg = cv2.cvtColor(cv2.imread(str(rgb_dir / f"{ts}.jpg")), cv2.COLOR_BGR2RGB)
                        r_jpg = range_to_decoder_grid(dac.pano_range(jpg.astype(np.float32) / 255.0),
                                                      shape[1], shape[2], device)
                        stats.update(log_stats("dac_jpg_vs_label", r_jpg, label, lab_mask))
                    stats["edge_fraction_dac"] = float(1 - keep_dac.float().mean())
                if da3 is not None:                                        # v3
                    shape = tuple(rng.shape)
                    maps = da3_maps(t)
                    r_da3, sky = maps[f"range_{args.da3_scale}"].float(), maps["sky"].float()
                    da3_valid = content.view(shape) & (r_da3 > 0.5) & (r_da3 < args.far_cut) & (sky < 0.3)
                    if args.da3_relabel_mode == "scale":
                        from dwm_pano_4d.da3_depth import rescale_gaussians
                        g_da3, rng_da3 = rescale_gaussians(g, rng, r_da3, da3_valid, origins)
                    else:
                        g_da3, rng_da3 = relabel_gaussians(g, rng, r_da3, da3_valid, origins, dirs)
                    keep_da3 = ~edge_mask(rng_da3, 0.1, 0.3, 1).to(device)
                    result["da3"], result["da3_edge"] = (g_da3, every), (g_da3, keep_da3)
                    lab_mask = use.view(shape)
                    stats.update(log_stats("da3_vae_vs_label", r_da3, label, lab_mask))
                    for other in ("pose", "metric", "ground"):
                        stats.update(log_stats(f"da3{other[0]}_vae_vs_label", maps[f"range_{other}"].float(), label,
                                               lab_mask))
                    stats.update(log_stats("pred_vs_label", rng, label, lab_mask))
                    stats.update({f"da3_bias_{k}": v for k, v in band_bias(r_da3, label, lab_mask).items()})
                    stats.update({f"pred_bias_{k}": v for k, v in band_bias(rng, label, lab_mask).items()})
                    far_label = (valid & (label >= args.far_cut)).view(shape) & content.view(shape)
                    stats["da3_sky_on_label_far"] = float((sky[far_label] >= 0.3).float().mean()) \
                        if far_label.any() else float("nan")
                    stats["da3_sky_on_label_near"] = float((sky[lab_mask] >= 0.3).float().mean())
                    stats["da3_use_fraction"] = float(da3_valid.float().sum() / content.float().sum().clamp_min(1))
                    for name in ("scale_pose", "scale_metric", "scale_ground", "pose_over_metric",
                                 "ground_over_metric"):
                        stats[f"da3_{name}"] = maps[name]
                    stats["edge_fraction_da3"] = float(1 - keep_da3.float().mean())
                    if t == args.gap:                                      # the scored frame without round trip
                        orig = da3_window([0, args.gap, 2 * args.gap], {args.gap}, round_trip=False)[args.gap]
                        stats.update(log_stats("da3_orig_vs_label", orig[f"range_{args.da3_scale}"].float(), label,
                                               lab_mask))
                return result, stats

            frames = [0, args.gap, 2 * args.gap]
            decoded = [decode(t) for t in frames]
            target = pipe._target_rgb_for_supervision(
                batch["vae_images"][:, [frames[1]]].flatten(0, 1).to(device))
            to_t = [relative_ego_transform(e2g[frames[s]], e2g[frames[1]]) for s in (0, 2)]
            row = {"clip_index": index, "scene_token": batch["scene_token"][0],
                   "geometry_t": decoded[1][1]}
            for name in variants_all:
                static, own = [], None
                for k in range(3):
                    g, keep = decoded[k][0][name]
                    frame = frames[k]
                    owner = points_in_boxes(
                        g[:, 0:3], batch["box_centers"][0, frame].to(device),
                        batch["box_rotations"][0, frame].to(device),
                        batch["box_half_sizes"][0, frame].to(device),
                        valid=batch["box_valid"][0, frame], margin=args.box_margin)
                    static.append(g[keep & (owner < 0) & content])
                    if k == 1:
                        own = (g[keep], g[keep & (owner >= 0) & content])
                g_t, dynamic_t = own
                own_rgb, _, _, own_valid = pipe._render_own_supervision(g_t)
                if dynamic_t.shape[0]:
                    _, _, dyn_alpha, _ = pipe._render_own_supervision(dynamic_t)
                    static_pixels = dyn_alpha <= 0.3
                else:
                    static_pixels = torch.ones_like(own_valid, dtype=torch.bool)
                pooled = torch.cat([transform_gaussians(static[0], to_t[0]), transform_gaussians(static[2], to_t[1])])
                single = transform_gaussians(static[0], to_t[0])
                row[f"{name}/psnr_own"] = masked_psnr(own_rgb, target, own_valid)
                for tag, scene in (("pooled", pooled), ("single", single)):
                    rgb, _, alpha, valid = pipe._render_own_supervision(scene)
                    support = valid & static_pixels
                    covered = support & (alpha > 0.5)
                    row[f"{name}/psnr_{tag}"] = masked_psnr(rgb, target, covered)
                    row[f"{name}/coverage_{tag}"] = float(covered.float().sum() / support.float().sum().clamp_min(1.0))
            rows.append(row)
            print("[labeldepth]", json.dumps(row), flush=True)

            if n < args.video_clips:
                out_dir = args.video_dir / f"clip{n:02d}_{batch['scene_token'][0][:8]}"
                out_dir.mkdir(parents=True, exist_ok=True)
                videos = {"views_raw": [], "views_v2": [], "far_raw": [], "far_v2": []}
                T = min(batch["vae_images"].shape[1], args.video_frames)
                for t in range(T):
                    variants, _ = decode(t)
                    rows_raw = ("pred", "label") + (("dac",) if dac is not None else ()) + \
                        (("da3",) if da3 is not None else ())
                    for mode, pairs in (("raw", rows_raw), ("v2", tuple(f"{r}_edge" for r in rows_raw))):
                        for kind, c2w_list, K, w, h in (
                                ("views", [VIEWS[v][1](t) for v in views], Kv, 960, h_view),
                                ("far", list(far_cameras(t)), Kf, 1280, 640)):
                            c2w = torch.tensor(np.stack(c2w_list), device=device, dtype=torch.float32)
                            stack = []
                            for name in pairs:
                                g, keep = variants[name]
                                g = g[keep & content]
                                g = g[g[:, 11:14].amax(dim=-1) <= args.max_scale]
                                rgb, _, alpha = pipe.render_targets(g, target_c2w=c2w, target_intrinsics=K,
                                                                    width=w, height=h)
                                rgb, alpha = rgb.clamp(0, 1).cpu(), alpha.cpu()
                                if mode == "v2":
                                    rgb = torch.stack([pushpull_fill(x, a) for x, a in zip(rgb, alpha)])
                                stack.append(torch.cat(list(rgb), dim=-1))
                            videos[f"{kind}_{mode}"].append(torch.cat(stack, dim=-2))
                fps = float(batch["fps"][0])
                for name, frames_list in videos.items():
                    tmp = out_dir / f"{name}.tmp.mp4"
                    save_tensor_to_video(str(tmp), "libx264", fps, frames_list)
                    tmp.replace(out_dir / f"{name}.mp4")
                print(f"[labeldepth] videos -> {out_dir}", flush=True)

        keys = [k for k in rows[0] if "/" in k]
        mean = {k: sum(r[k] for r in rows if not math.isnan(r[k])) /
                max(1, sum(1 for r in rows if not math.isnan(r[k]))) for k in keys}
        gkeys = list(rows[0]["geometry_t"])
        geometry = {k: float(np.nanmean([r["geometry_t"][k] for r in rows])) for k in gkeys}
        report = {"checkpoint": str(args.checkpoint), "data_pkl": str(args.data_pkl), "clips": len(rows),
                  "gap_frames": args.gap, "resize": args.resize, "far_cut_m": args.far_cut,
                  "mean": mean, "geometry_mean": geometry, "rows": rows,
                  "note": "training scenes with DAC labels; held-out frame t never enters pooled / single"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"mean": mean, "geometry_mean": geometry}, indent=2), flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
