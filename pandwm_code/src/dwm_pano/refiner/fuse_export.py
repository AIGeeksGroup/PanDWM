"""One Gaussian scene per clip, completed by the render-conditioned video model.

Instead of refining rendered images with a second model after the Gaussians (the d2f arm of refiner/eval.py), the
refiner's outputs are folded back into the scene, and the scored views are plain renders of that one Gaussian set:

  1. base scene A: the scene of the generated clip ("a"), or of the refiner's 0 m output ("r0");
  2. for every completion shift c (default +-2, +-4 m): scene B_c = the decoder on the refiner's output at c
     (refine.py's <refined>/clipNNN/d2_dy<c>.pt), with the clip's infos moved by c (d2v2_shift_infos);
  3. B_c keeps only the Gaussians of its pixels where A, rendered at c, is a hole (alpha < 0.5; select "hole") or a
     hole or magnified more than mag_thr times against its source view (select "fix"), dilated by --mask-dilate
     canvas pixels;
  4. the union (A plus every kept B_c moved into A's ego frame; current over 4D fill as in render_frame) is rendered
     at the evaluation shifts and written in refiner.export's pack format, so refiner/eval.py (raw arm) scores it unchanged.

Variants are named "<base>_<select>" with base in {a, r0} and select in {none, hole, fix}, plus optional "+flags"
(see FLAGS). Variants of a clip share the scenes:
    <out-root>/<variant>/clipNNN/render_dy<d>.pt   {"pixels": (35, 3, 256, 4114) uint8, "fixmask": (35, 256, 4114)
                                                     uint8 (holes of the fused render), "meta"}
--vae-colour colours every Gaussian with its VAE-decoded pixel (SceneOptions.vae_colour).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import dwm.common
from dwm_pano.geometry.calibrated_stitch import stitch_camera_views_from_owned_grid
from dwm_pano.gs.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.gs.render import FRAMES, pushpull_fill
from dwm_pano.gs.scene import SceneBuilder, composite, composite_depth
from dwm_pano.refiner.export import parse_clips
from dwm_pano.refiner.make_pairs import d2v2_shift_infos, moved_cameras, render_frame, scene_options
from dwm_pano.refiner.render import canvas_halves, render_stitched

BASES = ("a", "r0")
SELECTS = ("none", "hole", "fix")
# Variant flags: "+fill": B_c also adds its 4D fill; "+carve": drop completion Gaussians in another view's free space
# (free_space_keep); "+under": render A (current over fill) over the completion layers instead of depth-sorting them
# together; "+align": scale each completion layer's ranges to A's rendered ranges first (range_alignment).
# "<base>_holefill" / "_fixfill" are accepted for "<base>_hole+fill" / "_fix+fill".
FLAGS = ("fill", "carve", "under", "align")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--clips-root", required=True, type=Path)
    p.add_argument("--refined", required=True, type=Path, help="refine.py out-root")
    p.add_argument("--clips", required=True, help="comma list or start:end")
    p.add_argument("--variants", default="a_none,a_hole,a_fix,r0_none,r0_fix")
    p.add_argument("--complete-shifts", default="2,-2,4,-4")
    p.add_argument("--shifts", default="0,1,-1,2,-2,4,-4", help="evaluation shifts")
    p.add_argument("--fps-stride", default="10,20")
    p.add_argument("--mag-thr", type=float, default=2.0)
    p.add_argument("--mask-dilate", type=int, default=4)
    p.add_argument("--vae-colour", action="store_true")
    p.add_argument("--carve-tol-rel", type=float, default=0.08)
    p.add_argument("--carve-tol-abs", type=float, default=0.25)
    p.add_argument("--align-clamp", type=float, default=0.7, help="max |log range factor| of +align")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--time-budget-s", type=float, default=0.0)
    p.add_argument("--da3-src", required=True)
    p.add_argument("--da3-anyview", required=True)
    p.add_argument("--da3-metric", required=True)
    return p.parse_args()


def parse_variants(text):
    """'a_hole+carve, r0_fix' -> [(name, base, select, frozenset(flags)), ...]."""
    out = []
    for v in (x.strip() for x in text.split(",") if x.strip()):
        head, *flags = v.split("+")
        base, _, select = head.partition("_")
        if select in ("holefill", "fixfill"):
            select, flags = select.removesuffix("fill"), flags + ["fill"]
        if base not in BASES or select not in SELECTS or any(f not in FLAGS for f in flags) \
                or len(set(flags)) != len(flags) or (flags and select == "none"):
            raise ValueError(f"variant {v!r}: want <{'|'.join(BASES)}>_<{'|'.join(SELECTS)}>[+{'|+'.join(FLAGS)}]")
        out.append((v, base, select, frozenset(flags)))
    if len({o[0] for o in out}) != len(out):
        raise ValueError("duplicate variants")
    return out


def selection_halves(mask_canvas, dilate):
    """(H, 4114) bool canvas -> (2, H, 2304) bool over the decoder grid, dilated by `dilate` canvas pixels."""
    m = mask_canvas.float()[None, None]
    if dilate > 0:
        m = F.max_pool2d(m, 2 * dilate + 1, stride=1, padding=dilate)
    return canvas_halves(m[0, 0]) > 0.5


def shifted(g, c):
    """Gaussians of a scene whose ego frame is the base ego moved by c along y, in the base ego frame."""
    out = g.clone()
    out[:, 1] += float(c)
    return out


@torch.no_grad()
def render_layers(pipe, cur, fill, layers, layer_fill, dy, under=False):
    """render_frame for base Gaussians (cur over fill) plus completion layers (lists of (N, 14) in the base ego):
    depth-sorted together with the base (default), or composited under the whole base (under).  rgb (3, H, W),
    alpha (H, W)."""
    c2w = moved_cameras(pipe, dy)
    extra = torch.cat(layers) if layers else cur[:0]
    efill = torch.cat(layer_fill) if layer_fill else cur[:0]

    def over(front, back):
        rgb, alpha = render_stitched(pipe, front, c2w)
        if back.shape[0]:
            rgb_b, alpha_b = render_stitched(pipe, back, c2w)
            rgb, alpha = composite(rgb, alpha, rgb_b, alpha_b)
        return rgb, alpha

    if under:
        rgb, alpha = over(cur, fill)
        if extra.shape[0] or efill.shape[0]:
            rgb_b, alpha_b = over(extra, efill)
            rgb, alpha = composite(rgb, alpha, rgb_b, alpha_b)
    else:
        rgb, alpha = over(torch.cat([cur, extra]), torch.cat([fill, efill]))
    alpha = torch.where(pipe.stitch_valid[None], alpha, torch.zeros_like(alpha))
    return rgb[0].clamp(0, 1), alpha[0]


@torch.no_grad()
def reference_view(pipe, cur, fill, dy):
    """(c2w (6, 4, 4), depth (6, h, w), alpha (6, h, w)) of cur over fill at the six nuScenes cameras moved by dy
    (gsplat RGB+ED: camera-z depth normalised by alpha)."""
    c2w = moved_cameras(pipe, dy)
    kw = dict(target_c2w=c2w, target_intrinsics=pipe.nuscenes_render_intrinsics,
              width=pipe.nuscenes_render_width, height=pipe.nuscenes_render_height)
    _, depth, alpha = pipe.render_targets(cur, **kw)
    if fill.shape[0]:
        _, depth_f, alpha_f = pipe.render_targets(fill, **kw)
        depth, alpha = composite_depth(depth, alpha, depth_f, alpha_f), alpha + (1.0 - alpha) * alpha_f
    return c2w, depth, alpha


@torch.no_grad()
def stitched_range(pipe, cur, fill, dy):
    """(H, 4114) range from each owning camera centre along its pixel ray, and alpha, of cur over fill at the six
    nuScenes cameras moved by dy (per-camera camera-z times |K^-1 (u, v, 1)|, stitched like the renders)."""
    c2w = moved_cameras(pipe, dy)
    K = pipe.nuscenes_render_intrinsics
    h, w = pipe.nuscenes_render_height, pipe.nuscenes_render_width
    kw = dict(target_c2w=c2w, target_intrinsics=K, width=w, height=h)
    _, depth, alpha = pipe.render_targets(cur, **kw)
    if fill.shape[0]:
        _, depth_f, alpha_f = pipe.render_targets(fill, **kw)
        depth, alpha = composite_depth(depth, alpha, depth_f, alpha_f), alpha + (1.0 - alpha) * alpha_f
    ys = torch.arange(h, device=depth.device, dtype=torch.float32) + 0.5
    xs = torch.arange(w, device=depth.device, dtype=torch.float32) + 0.5
    x = (xs[None, None, :] - K[:, 0, 2, None, None]) / K[:, 0, 0, None, None]
    y = (ys[None, :, None] - K[:, 1, 2, None, None]) / K[:, 1, 1, None, None]
    rng = depth * torch.sqrt(x * x + y * y + 1.0)
    stitched, _ = stitch_camera_views_from_owned_grid(torch.stack([rng, alpha], dim=1), pipe.stitch_sampling_grid,
                                                      pipe.stitch_owner, pipe.stitch_valid)
    return stitched[0], torch.where(pipe.stitch_valid, stitched[1], torch.zeros_like(stitched[1]))


def range_alignment(ranges_b, range_a, alpha_a, clamp=0.7, erode=2, far=80.0):
    """(2, H, 2304) factors exp(l) for a completion layer's ranges: l = log(A's rendered range / the layer's range) on the
    pixels both see (A alpha > 0.95, eroded by `erode`, 0.5 < ranges < far), push-pull filled into the rest of each half
    (the holes take the correction of their surroundings), clamped to +-clamp. Returns (factors, fraction of pixels
    used).
    """
    rho_a, a = canvas_halves(range_a), canvas_halves(alpha_a)
    rho_b = ranges_b.to(rho_a)
    ok = (a > 0.95) & (rho_a > 0.5) & (rho_a < far) & (rho_b > 0.5) & (rho_b < far)
    if erode > 0:
        ok = -F.max_pool2d(-ok.float()[:, None], 2 * erode + 1, stride=1, padding=erode)[:, 0] > 0.5
    ell = torch.where(ok, torch.log(rho_a.clamp_min(1e-3) / rho_b.clamp_min(1e-3)), torch.zeros_like(rho_a))
    if not bool(ok.any()):
        return torch.ones_like(rho_b), 0.0
    filled = torch.stack([pushpull_fill(ell[v][None].expand(3, -1, -1), ok[v].float())[0] for v in range(2)])
    return filled.clamp(-clamp, clamp).exp(), float(ok.float().mean())


def free_space_keep(pipe, g, refs, tol_rel, tol_abs, pool=3):
    """(N,) bool: False for Gaussians whose centre lies in front of the surface another view sees, i.e. in space that view
    observed as empty: inside a camera, reference alpha > 0.5 and camera z < depth * (1 - tol_rel) - tol_abs. Depth and
    alpha are min-pooled over pool x pool pixels, so only clear violations away from edges and holes count (completion
    layers decoded at other shifts otherwise float in front of A and of the view's own layer).
    """
    p = g[:, 0:3]
    keep = torch.ones(p.shape[0], dtype=torch.bool, device=p.device)
    if not p.shape[0]:
        return keep
    K = pipe.nuscenes_render_intrinsics.to(p)
    for c2w, depth, alpha in refs:
        cams, h, w = depth.shape
        d_min = -F.max_pool2d(-depth[:, None].float(), pool, stride=1, padding=pool // 2)[:, 0]
        a_min = -F.max_pool2d(-alpha[:, None].float(), pool, stride=1, padding=pool // 2)[:, 0]
        view = torch.linalg.inv(c2w.to(p))
        pc = torch.einsum("kij,nj->kni", view[:, :3, :3], p) + view[:, None, :3, 3]
        z = pc[..., 2]
        zc = z.clamp_min(1e-3)
        u = K[:, None, 0, 0] * pc[..., 0] / zc + K[:, None, 0, 2]
        v = K[:, None, 1, 1] * pc[..., 1] / zc + K[:, None, 1, 2]
        inside = (z > 0.3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ui, vi = u.clamp(0, w - 1).long(), v.clamp(0, h - 1).long()
        cam = torch.arange(cams, device=p.device)[:, None].expand_as(ui)
        d_ref, a_ref = d_min[cam, vi, ui], a_min[cam, vi, ui]
        free = inside & (a_ref > 0.5) & (z < d_ref * (1.0 - tol_rel) - tol_abs)
        keep &= ~free.any(0)
    return keep


def reset(scene, keep, align=False):
    scene.pixel_keep = keep
    scene.range_scale = {} if align else None
    scene._decoded, scene._static, scene._ranges = {}, {}, {}


@torch.no_grad()
def completion_masks(pipe, base, shifts, thr, dilate):
    """{c: {"hole": {t: (2, H, 2304)}, "fix": {...}}} from the base scene rendered at each completion shift."""
    valid = pipe.stitch_valid
    out = {c: {"hole": {}, "fix": {}} for c in shifts}
    for t in range(FRAMES):
        for c in shifts:
            _, alpha, mag_map = render_frame(pipe, base, t, c, with_mag=True)
            hole = (alpha < 0.5) & valid
            magnified = (mag_map > thr) & ~hole & valid
            out[c]["hole"][t] = selection_halves(hole, dilate).cpu()
            out[c]["fix"][t] = selection_halves(hole | magnified, dilate).cpu()
    return out


@torch.no_grad()
def main():
    args = parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    variants = parse_variants(args.variants)
    opts = scene_options()
    opts.vae_colour = bool(args.vae_colour)
    shifts = [float(s) for s in args.shifts.split(",") if s.strip()]
    cshifts = [float(s) for s in args.complete_shifts.split(",") if s.strip()]
    for group in (shifts, cshifts):
        if any(float(f"{d:g}") != d for d in group) or len({f"{d:g}" for d in group}) != len(group):
            raise ValueError("shifts must have distinct, exactly representable %g names")
    if any(c == 0 for c in cshifts):
        raise ValueError("completion shifts must be non-zero")
    fps, stride = (float(x) for x in args.fps_stride.split(","))
    from dwm_pano.recon.render_ours import camera_time_infos
    from dwm_pano_4d.da3_depth import Da3Ranger
    da3 = Da3Ranger(args.da3_src, args.da3_anyview, args.da3_metric, device)
    started = ensure_single_rank_process_group(device)
    try:
        config = load_config(args.config)
        torch.manual_seed(int(config["generator_seed"]))
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(args.out_root),
                                                      config=config, device=device, resume_from=None)
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
        base_ds = dataset.base_dataset
        by_key = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in
                  enumerate(zip(base_ds.clips, base_ds.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring
        job_start, last = time.monotonic(), 0.0
        for index in parse_clips(args.clips):
            todo = [(v, b, s, f) for v, b, s, f in variants
                    if not all((args.out_root / v / f"clip{index:03d}" / f"render_dy{d:g}.pt").is_file()
                               for d in shifts)]
            if not todo:
                continue
            if args.time_budget_s and last and time.monotonic() - job_start + 1.3 * last > args.time_budget_s:
                print(f"[fuse-export] budget reached before clip {index}", flush=True)
                break
            t0 = time.monotonic()
            src = args.clips_root / f"clip{index:03d}"
            ref_dir = args.refined / f"clip{index:03d}"
            summary = json.loads((src / "summary.json").read_text())
            ds_index = by_key[(summary["scene_token"], int(summary["anchor_timestamp_us"]))]
            batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [ds_index]),
                                                          batch_size=1, shuffle=False, num_workers=0,
                                                          collate_fn=CollateFnIgnoring(keys=[]))))
            frame_ts = [int(x) for x in batch["frame_timestamps_us"][0].tolist()]
            infos = camera_time_infos(base_ds.clips[ds_index])

            def refined(c):
                pack = torch.load(ref_dir / f"d2_dy{c:g}.pt", map_location="cpu", weights_only=True)
                if pack["meta"]["frame_timestamps_us"] != frame_ts:
                    raise RuntimeError(f"clip {index} dy {c:g}: refined frames differ from the dataset clip")
                lat = pack["latents"].float()
                if lat.shape[0] != FRAMES:
                    raise ValueError(f"clip {index} dy {c:g}: {lat.shape[0]} refined frames")
                return lat

            bases = {}
            for b in sorted({o[1] for o in todo}):
                if b == "a":
                    lat = torch.load(src / "latents.pt", map_location="cpu", weights_only=True)["latents"]
                    if lat.shape[0] != FRAMES:
                        raise ValueError(f"clip {index}: {lat.shape[0]} latent frames")
                else:
                    lat = refined(0.0)
                bases[b] = SceneBuilder(pipe, batch, infos, lat, opts, device, da3=da3)
            needs = any(o[2] != "none" for o in todo)
            comps = {c: SceneBuilder(pipe, batch, d2v2_shift_infos(infos, c), refined(c), opts, device, da3=da3)
                     for c in cshifts} if needs else {}
            masks = {b: completion_masks(pipe, bases[b], cshifts, args.mag_thr, args.mask_dilate)
                     for b in bases if any(o[1] == b and o[2] != "none" for o in todo)}
            for v, b, s, flags in todo:
                base = bases[b]
                used = {} if s == "none" else comps
                for c, scene in used.items():
                    reset(scene, masks[b][c][s], align="align" in flags)
                out = {d: {"pixels": [], "fixmask": [], "coverage": []} for d in shifts}
                kept = {f"{c:g}": [] for c in used}
                carved = {f"{c:g}": [] for c in used}
                aligned = {f"{c:g}": [] for c in used}
                for t in range(FRAMES):
                    cur, fill = base.decode(t), base.fill(t)
                    if "align" in flags:
                        for c, scene in used.items():
                            scene.decode(t)                                  # stores the unscaled ranges in scene._ranges[t]
                            range_a, alpha_a = stitched_range(pipe, cur, fill, c)
                            factor, used_frac = range_alignment(scene._ranges[t], range_a, alpha_a, args.align_clamp)
                            scene.range_scale[t] = factor
                            scene._decoded.pop(t, None)
                            aligned[f"{c:g}"].append(float(factor.log().abs().mean()))
                    layers = {c: shifted(scene.decode(t), c) for c, scene in used.items()}
                    layer_fill = [shifted(scene.fill(t), c) for c, scene in used.items()] if "fill" in flags else []
                    if "carve" in flags and layers:
                        refs = {0.0: reference_view(pipe, cur, fill, 0.0)}
                        refs.update({c: reference_view(pipe, torch.cat([cur, g]), fill, c) for c, g in layers.items()})
                        for c in list(layers):
                            keep = free_space_keep(pipe, layers[c], [r for vv, r in refs.items() if vv != c],
                                                   args.carve_tol_rel, args.carve_tol_abs)
                            carved[f"{c:g}"].append(float((~keep).float().mean()) if keep.numel() else 0.0)
                            layers[c] = layers[c][keep]
                        del refs
                    for c, g in layers.items():
                        kept[f"{c:g}"].append(int(g.shape[0]))
                    for d in shifts:
                        rgb, alpha = render_layers(pipe, cur, fill, list(layers.values()), layer_fill, d,
                                                   under="under" in flags)
                        valid = pipe.stitch_valid
                        hole = (alpha < 0.5) & valid
                        canvas = pushpull_fill(rgb, alpha).clamp(0, 1)
                        o = out[d]
                        o["pixels"].append((canvas * 255.0).round().clamp(0, 255).to(torch.uint8).cpu())
                        o["fixmask"].append(hole.to(torch.uint8).cpu())
                        o["coverage"].append(float((alpha[valid] >= 0.5).float().mean()))
                dst = args.out_root / v / f"clip{index:03d}"
                dst.mkdir(parents=True, exist_ok=True)
                for d in shifts:
                    o = out[d]
                    meta = {"clip_index": index, "dy": d, "variant": v, "base": b, "select": s,
                            "flags": sorted(flags), "carve_tol": [args.carve_tol_rel, args.carve_tol_abs],
                            "carved_fraction_mean": {k: sum(x) / len(x) for k, x in carved.items() if x},
                            "align_mean_abs_log": {k: sum(x) / len(x) for k, x in aligned.items() if x},
                            "scene_token": summary["scene_token"],
                            "anchor_timestamp_us": int(summary["anchor_timestamp_us"]),
                            "frame_timestamps_us": frame_ts, "mode": "fuse_v3", "coverage": o["coverage"],
                            "coverage_mean": sum(o["coverage"]) / FRAMES, "complete_shifts": list(used),
                            "completion_gaussians_mean": {k: sum(x) / len(x) for k, x in kept.items()},
                            "mag_thr": args.mag_thr, "mask_dilate": args.mask_dilate,
                            "vae_colour": bool(args.vae_colour), "checkpoint": str(args.checkpoint),
                            "refined": str(args.refined), "scene": {k: getattr(opts, k) for k in (
                                "da3_window", "da3_fuse", "plane_offset_px", "box_reading", "drop_bottom_rows",
                                "vae_colour")},
                            "scene_stats": base.summary()}
                    path = dst / f"render_dy{d:g}.pt"
                    tmp = path.with_suffix(".tmp")
                    torch.save({"pixels": torch.stack(o["pixels"]), "fixmask": torch.stack(o["fixmask"]),
                                "meta": meta}, tmp)
                    tmp.replace(path)
                print(f"[fuse-export] clip {index} {v} cov "
                      f"{[round(sum(out[d]['coverage']) / FRAMES, 3) for d in shifts]} kept "
                      f"{ {k: round(sum(x) / max(1, len(x))) for k, x in kept.items()} } carved "
                      f"{ {k: round(sum(x) / len(x), 3) for k, x in carved.items() if x} } align "
                      f"{ {k: round(sum(x) / len(x), 3) for k, x in aligned.items() if x} }", flush=True)
            last = time.monotonic() - t0
            print(f"[fuse-export] clip {index} done ({last:.0f}s)", flush=True)
        print("FUSE_EXPORT_TASK_DONE", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
