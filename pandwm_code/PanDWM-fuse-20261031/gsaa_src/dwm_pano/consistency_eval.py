"""Multi-view consistency of the lateral-shift views, without ground truth (2026-11-01,
PanDWM-d2-refiner-20261021/CONSIST_PLAN_20261101.md).

FID / FVD of the shifted views are unpaired: they reward realistic frames but would not notice two views of the same
scene that disagree.  The G3b Gaussians give each shifted view a geometry, so the image of view s2 can be warped into
view s1: every canvas pixel of s1 with Gaussian coverage (alpha > 0.95, 0.5 m < range < 80 m) becomes the 3D point
origin + direction * range (the exact ray the renderer sampled, calibrated_stitch.sampling_grid_to_ego_rays), is
projected into the s2 canvas (azimuth / elevation of its direction from each camera centre; a first guess from 1D
per-column / per-row tables refined by Gauss-Newton on the canvas's own azimuth / elevation maps, because the
OneBEV canvas is not separable: rows drift by up to ~25 px from a single elevation table, run 786947; accepted only
in a column owned by that camera, with a residual below 0.5 px) and kept if s2 sees it (alpha > 0.95, range within 5 % + 0.2 m).  The same mask is used for every image kind, so the
kinds differ only in their pixels:

    raw       the G3b render at s1 vs the warped G3b render at s2 (consistent by construction: the ceiling)
    d2f       the D2-refined view at s1 vs the warped refined view at s2
    vid_raw   the generated video (0 m only) vs the warped G3b render at s2     (pairs with s1 = 0)
    vid_d2f   the generated video vs the warped refined view at s2              (does the refined view show the
                                                                                 generated scene?)
    d2m       d2v2_eval's mask-only composite (refined inside the dilated / feathered fixmask, render elsewhere)
    vid_d2m   the generated video vs the warped d2m view
    d2f_x     (with --refined2, a second refinement with another seed) the refined view at s1 vs the warped SECOND
              refinement at s2; at s1 = s2 the floor two samples of the same view reach
(2026-11-01 run 2 added d2m / vid_d2m / d2f_x, CONSIST_PLAN_20261101.md section 4.)

PSNR on the masked pixels at full resolution, after 4 x 4 average pooling (less sensitive to sharpness differences:
a sharper image pays more for the same sub-pixel misregistration; also 8 x 8), and on the near pixels (range
< 20 m).  Pair "0:0" of raw checks the warp itself (should be near-lossless; smoke 786964: 64 dB); "0:0" of vid_raw /
vid_d2f is the same-view difference between the generated video and the G3b render / the refined 0 m view, the floor
the shifted pairs of that kind start from.  Shards write jsonl rows; --merge averages them.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

KINDS = ("raw", "d2f", "d2m", "d2f_x", "vid_raw", "vid_d2f", "vid_d2m")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", type=Path)
    p.add_argument("--data-pkl", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--clips-root", type=Path, help="generated clips (latents.pt, summary.json)")
    p.add_argument("--inputs", type=Path, help="d2v2_export.py out-root (raw renders)")
    p.add_argument("--refined", type=Path, help="d2_refine.py out-root")
    p.add_argument("--refined2", type=Path, default=None, help="a second d2_refine.py out-root (other seed)")
    p.add_argument("--dilate", type=int, default=6, help="d2m mask, as d2v2_eval")
    p.add_argument("--feather", type=int, default=6)
    p.add_argument("--clips", default="")
    p.add_argument("--frames", default="8,18,28")
    p.add_argument("--pairs", default="0:0,0:1,0:2,0:4,1:2,2:4,0:-1,0:-2,0:-4,-1:-2,-2:-4",
                   help="target:source; the source image is warped into the target view")
    p.add_argument("--fps-stride", default="10,20")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--da3-src")
    p.add_argument("--da3-anyview")
    p.add_argument("--da3-metric")
    return p.parse_args()


# ---------------------------------------------------------------- canvas geometry (CPU-testable)
def _unwrap(a):
    d = torch.remainder(a[1:] - a[:-1] + math.pi, 2 * math.pi) - math.pi
    return torch.cat([a[:1], a[:1] + torch.cumsum(d, 0)])


def axis_tables(dirs):
    """(H, W, 3) unit canvas ray directions -> dict with increasing azimuth (per column) and elevation (per row)
    tables and their signs: column c has azimuth az_sign * az[c], row r elevation el_sign * el[r]."""
    az = torch.atan2(dirs[..., 1], dirs[..., 0]).median(dim=0).values
    el = torch.atan2(dirs[..., 2], torch.linalg.norm(dirs[..., :2], dim=-1)).median(dim=1).values
    az = _unwrap(az)
    az_sign = 1.0 if float(az[-1] - az[0]) > 0 else -1.0
    el_sign = 1.0 if float(el[-1] - el[0]) > 0 else -1.0
    return {"az": az * az_sign, "el": el * el_sign, "az_sign": az_sign, "el_sign": el_sign}


def _frac_index(xq, xt):
    """Fractional index of xq in the increasing table xt, clamped to [0, n - 1]."""
    n = xt.shape[0]
    i = torch.searchsorted(xt, xq.contiguous()).clamp(1, n - 1)
    x0, x1 = xt[i - 1], xt[i]
    idx = (i - 1).to(xq.dtype) + (xq - x0) / (x1 - x0).clamp_min(1e-12)
    return idx.clamp(0, n - 1)


def _wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def canvas_maps(dirs, tables):
    """(6, H, W): signed azimuth (unwrapped to its column's table value), signed elevation and their derivatives
    d/dcol, d/drow of every canvas ray, for the Gauss-Newton refinement in project()."""
    az = tables["az_sign"] * torch.atan2(dirs[..., 1], dirs[..., 0])
    az = tables["az"][None, :] + _wrap(az - tables["az"][None, :])
    el = tables["el_sign"] * torch.atan2(dirs[..., 2], torch.linalg.norm(dirs[..., :2], dim=-1))
    daz_r, daz_c = torch.gradient(az)
    del_r, del_c = torch.gradient(el)
    return torch.stack([az, el, daz_c, daz_r, del_c, del_r])


def project(points, centres, owner, tables, maps, iters=5, tol_px=0.5):
    """(N, 3) points -> (row, col) fractional canvas coordinates, range from the owning camera centre and a found
    flag, for the rig whose six camera centres are `centres` (6, 3).  owner (W,) is the camera of each column; a
    point belongs to camera k only if its direction from k lands, within tol_px after `iters` Gauss-Newton steps on
    `maps` (canvas_maps), inside the band in a column owned by k."""
    n = points.shape[0]
    row = torch.full((n,), float("nan"), device=points.device)
    col, rng = row.clone(), row.clone()
    found = torch.zeros(n, dtype=torch.bool, device=points.device)
    az_t, el_t = tables["az"], tables["el"]
    h, w = maps.shape[-2:]
    for k in range(centres.shape[0]):
        q = points - centres[k]
        az = tables["az_sign"] * torch.atan2(q[:, 1], q[:, 0])
        az = az_t[0] + torch.remainder(az - az_t[0], 2 * math.pi)
        el = tables["el_sign"] * torch.atan2(q[:, 2], torch.linalg.norm(q[:, :2], dim=-1))
        c, r = _frac_index(az, az_t), _frac_index(el, el_t)
        for _ in range(iters + 1):
            m = sample(maps, r, c)
            e_az, e_el = _wrap(az - m[0]), el - m[1]
            a, b, cc, d = m[2], m[3], m[4], m[5]
            det = a * d - b * cc
            det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
            dc, dr = (d * e_az - b * e_el) / det, (-cc * e_az + a * e_el) / det
            step = torch.sqrt(dc * dc + dr * dr)
            if _ == iters:
                break
            c, r = (c + dc.clamp(-50, 50)).clamp(0, w - 1), (r + dr.clamp(-50, 50)).clamp(0, h - 1)
        ok = (step < tol_px) & ~found & (r > 0) & (r < h - 1)
        ok &= owner[c.round().long().clamp(0, w - 1)] == k
        row, col = torch.where(ok, r, row), torch.where(ok, c, col)
        rng = torch.where(ok, torch.linalg.norm(q, dim=-1), rng)
        found |= ok
    return row, col, rng, found


def sample(img, row, col):
    """Bilinear samples of (C, H, W) at fractional (row, col) (N,) -> (C, N)."""
    h, w = img.shape[-2:]
    grid = torch.stack([(col + 0.5) / w * 2 - 1, (row + 0.5) / h * 2 - 1], dim=-1).view(1, 1, -1, 2)
    return F.grid_sample(img[None].float(), grid.to(img.device).float(), mode="bilinear", padding_mode="border",
                         align_corners=False)[0, :, 0]


def psnr_masked(a, b, mask, pool=1):
    """PSNR of (3, H, W) images in [0, 1] over mask (H, W); with pool > 1 after pool x pool averaging, keeping only
    fully masked cells.  NaN for an empty mask."""
    m = mask.float()[None]
    if pool > 1:
        a, b = F.avg_pool2d(a[None], pool)[0], F.avg_pool2d(b[None], pool)[0]
        m = (F.avg_pool2d(m[None], pool)[0] > 0.999).float()
    count = float(m.sum())
    if count < 1:
        return float("nan")
    mse = float((((a - b) ** 2).mean(0, keepdim=True) * m).sum()) / count
    return 10 * math.log10(1.0 / max(mse, 1e-10))


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def run(args):
    import dwm.common
    from dwm_pano.calibrated_stitch import sampling_grid_to_ego_rays
    from dwm_pano.d2v2_eval import decode_latents, soft_mask
    from dwm_pano.d2v2_export import parse_clips
    from dwm_pano.d2v2_fuse_export import stitched_range
    from dwm_pano.d2v2_make_pairs import moved_cameras, scene_options
    from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
        install_strict_pano_hardseam_stub, load_config
    from dwm_pano.export_gs_batch149 import FRAMES
    from dwm_pano.gs_scene_v3 import SceneBuilder
    from dwm_pano_4d.da3_depth import Da3Ranger
    from dwm_pano.recon.render_ours import camera_time_infos

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    frames = [int(x) for x in args.frames.split(",")]
    pairs = [tuple(float(v) for v in x.split(":")) for x in args.pairs.split(",") if x.strip()]
    shifts = sorted({s for p in pairs for s in p})
    fps, stride = (float(x) for x in args.fps_stride.split(","))
    da3 = Da3Ranger(args.da3_src, args.da3_anyview, args.da3_metric, device)
    started = ensure_single_rank_process_group(device)
    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / f"rows_{args.task}.jsonl"
    done = set()
    if rows_path.is_file():
        done = {json.loads(line)["clip"] for line in rows_path.read_text().splitlines() if line.strip()}
    try:
        config = load_config(args.config)
        torch.manual_seed(int(config["generator_seed"]))
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(args.out), config=config,
                                                      device=device, resume_from=None)
        pipe.vae.eval()
        pipe.decoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
        pipe.decoder.eval()
        pipe.decoder_wrapper.eval()
        pipe.rasterize_mode = "antialiased"
        install_strict_pano_hardseam_stub()
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl,
                                   fps_stride_tuples=[[int(fps), stride]])
        cfg["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        by_key = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in
                  enumerate(zip(base.clips, base.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring
        opts = scene_options()
        for index in parse_clips(args.clips):
            if index in done:
                continue
            t0 = time.monotonic()
            src = args.clips_root / f"clip{index:03d}"
            summary = json.loads((src / "summary.json").read_text())
            ds_index = by_key[(summary["scene_token"], int(summary["anchor_timestamp_us"]))]
            batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [ds_index]), batch_size=1,
                                                          shuffle=False, num_workers=0,
                                                          collate_fn=CollateFnIgnoring(keys=[]))))
            frame_ts = [int(x) for x in batch["frame_timestamps_us"][0].tolist()]
            latents = torch.load(src / "latents.pt", map_location="cpu", weights_only=True)["latents"]
            scene = SceneBuilder(pipe, batch, camera_time_infos(base.clips[ds_index]), latents, opts, device, da3=da3)
            hw = (pipe.nuscenes_render_height, pipe.nuscenes_render_width)
            owner = pipe.stitch_owner[0].to(device)
            valid = pipe.stitch_valid.to(device)
            rays = {}
            for s in shifts:
                o, d, _ = sampling_grid_to_ego_rays(pipe.stitch_sampling_grid, pipe.stitch_owner,
                                                    moved_cameras(pipe, s).to(device),
                                                    pipe.nuscenes_render_intrinsics, source_size=hw)
                rays[s] = (o.reshape(*valid.shape, 3), d.reshape(*valid.shape, 3))
            tables = axis_tables(rays[0.0][1])
            maps = canvas_maps(rays[0.0][1], tables)
            centres = {s: moved_cameras(pipe, s)[:, :3, 3].to(device) for s in shifts}
            if index == parse_clips(args.clips)[0]:                  # how well the tables invert the canvas itself
                pts = rays[0.0][0][valid] + rays[0.0][1][valid] * 10.0
                r, c, _, f = project(pts, centres[0.0], owner, tables, maps)
                rr, cc = torch.nonzero(valid, as_tuple=True)
                err = torch.stack([r - rr, c - cc], -1)[f].abs()
                print(f"[consist] inverse check: found {float(f.float().mean()):.4f}, |drow| mean "
                      f"{float(err[:, 0].mean()):.3f} max {float(err[:, 0].max()):.3f}, |dcol| mean "
                      f"{float(err[:, 1].mean()):.3f} max {float(err[:, 1].max()):.3f}", flush=True)
            def refined(root, s):
                ref = torch.load(root / f"clip{index:03d}" / f"d2_dy{s:g}.pt", map_location="cpu", weights_only=True)
                if ref["meta"]["frame_timestamps_us"] != frame_ts:
                    raise RuntimeError(f"clip {index} dy {s:g}: refined frames differ ({root})")
                img = decode_latents(pipe, ref["latents"][frames].float())
                if img.shape[-2:] != valid.shape:
                    img = F.interpolate(img, size=valid.shape, mode="bilinear", align_corners=False)
                return img

            images = {"raw": {}, "d2f": {}, "d2m": {}, "d2f_x": {}}
            for s in shifts:
                pack = torch.load(args.inputs / f"clip{index:03d}" / f"render_dy{s:g}.pt", map_location="cpu",
                                  weights_only=True)
                if pack["meta"]["frame_timestamps_us"] != frame_ts:
                    raise RuntimeError(f"clip {index} dy {s:g}: render frames differ")
                images["raw"][s] = pack["pixels"][frames].to(device).float() / 255.0
                images["d2f"][s] = refined(args.refined, s)
                w = soft_mask(pack["fixmask"][frames].to(device), args.dilate, args.feather)[:, None]
                images["d2m"][s] = w * images["d2f"][s] + (1.0 - w) * images["raw"][s]
                if args.refined2 is not None:
                    images["d2f_x"][s] = refined(args.refined2, s)
            video = decode_latents(pipe, latents[frames].float())
            if video.shape[-2:] != valid.shape:
                video = F.interpolate(video, size=valid.shape, mode="bilinear", align_corners=False)
            out_rows = []
            for i, t in enumerate(frames):
                cur, fill = scene.decode(t), scene.fill(t)
                geo = {s: stitched_range(pipe, cur, fill, s) for s in shifts}
                for s1, s2 in pairs:
                    rng1, alpha1 = geo[s1]
                    m1 = valid & (alpha1 > 0.95) & (rng1 > 0.5) & (rng1 < 80.0)
                    o1, d1 = rays[s1]
                    pts = o1[m1] + d1[m1] * rng1[m1][:, None]
                    r, c, rho, f = project(pts, centres[s2], owner, tables, maps)
                    rng2, alpha2 = geo[s2]
                    rc = torch.stack([r.nan_to_num(0), c.nan_to_num(0)])
                    g2 = sample(torch.stack([rng2, alpha2]), rc[0], rc[1])
                    vis = f & (g2[1] > 0.95) & ((rho - g2[0]).abs() < 0.05 * rho + 0.2)
                    mask = torch.zeros_like(m1)
                    mask[m1] = vis
                    near = mask.clone()
                    near[m1] = vis & (rng1[m1] < 20.0)
                    kinds = [("raw", images["raw"][s1][i], images["raw"][s2][i]),
                             ("d2f", images["d2f"][s1][i], images["d2f"][s2][i]),
                             ("d2m", images["d2m"][s1][i], images["d2m"][s2][i])]
                    if args.refined2 is not None:
                        kinds.append(("d2f_x", images["d2f"][s1][i], images["d2f_x"][s2][i]))
                    if s1 == 0.0:
                        kinds += [("vid_raw", video[i], images["raw"][s2][i]),
                                  ("vid_d2f", video[i], images["d2f"][s2][i]),
                                  ("vid_d2m", video[i], images["d2m"][s2][i])]
                    for kind, target, source in kinds:
                        if s1 == s2 and kind in ("d2f", "d2m"):             # an image against itself
                            continue
                        warped = torch.zeros_like(target)
                        warped[:, m1] = sample(source, rc[0], rc[1])
                        out_rows.append({"clip": index, "frame": t, "pair": f"{s1:g}:{s2:g}", "kind": kind,
                                         "mask_frac": float(mask[valid].float().mean()),
                                         "near_frac": float(near[valid].float().mean()),
                                         "psnr": psnr_masked(target, warped, mask),
                                         "psnr_pool4": psnr_masked(target, warped, mask, pool=4),
                                         "psnr_pool8": psnr_masked(target, warped, mask, pool=8),
                                         "psnr_near": psnr_masked(target, warped, near)})
            with rows_path.open("a") as fh:
                for row in out_rows:
                    fh.write(json.dumps(row) + "\n")
            del scene
            print(f"[consist] clip {index} done ({time.monotonic() - t0:.0f}s)", flush=True)
        print("CONSIST_TASK_DONE", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


def merge(args):
    rows = []
    for path in sorted(args.out.glob("rows_*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    clips = sorted({r["clip"] for r in rows})
    summary = {"clips": len(clips), "rows": len(rows), "pairs": {}}
    for pair in sorted({r["pair"] for r in rows}, key=lambda p: (abs(float(p.split(":")[1]) - float(p.split(":")[0])),
                                                                p)):
        summary["pairs"][pair] = {}
        for kind in KINDS:
            sel = [r for r in rows if r["pair"] == pair and r["kind"] == kind]
            if not sel:
                continue
            entry = {"n": len(sel), "mask_frac": sum(r["mask_frac"] for r in sel) / len(sel)}
            for key in ("psnr", "psnr_pool4", "psnr_pool8", "psnr_near"):
                vals = [r[key] for r in sel if r[key] == r[key]]
                entry[key] = round(sum(vals) / len(vals), 2) if vals else None
            summary["pairs"][pair][kind] = entry
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    a = parse_args()
    merge(a) if a.merge else run(a)
