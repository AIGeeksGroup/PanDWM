"""Leave-one-out error decomposition (2026-10-30, OVERFIT_DIAG_20261029.md follow-up).  Inference only.

For deterministic samples of the training dataset (clip k, frames 0 / 4 / 8 of its 19-frame window, target = frame 4),
the leave-one-out render of pipeline.py's train_step is rebuilt exactly: static Gaussians of frames 0 and 8 moved into
frame 4's ego plus frame 4's own dynamic Gaussians, rendered and stitched like the supervision; mask = stitch-valid,
alpha > loo_min_alpha, not a dynamic pixel.  The own render (frame 4's Gaussians only) is scored on the same mask.

Per-pixel error e = mean over RGB of |render - target|, decomposed into
  raw          as the training loss;
  affine       after one least-squares 3x4 colour affine (render -> target) fitted on the mask (global colour /
               exposure differences);
  shift        affine-corrected, then the minimum over integer shifts within +-2 px (sub-few-pixel misregistration);
and split by pixel class (on frame 4's own render depth and the target image): depth edge (relative jump > 10 % to an
8-neighbour, dilated 2 px), non-edge near (< 10 m) / mid (10-30 m) / far (> 30 m), non-edge high texture (top 20 % of the
target gradient magnitude) / flat (bottom 50 %).  Also the gradient-energy ratio render / target (blur).

2026-10-30 per-camera variant ("camera", always computed next to the frame-level "frame" one): every source Gaussian is
moved with the ego pose of ITS camera's exposure (info cams[c] ego2global) instead of the frame's mean camera-time pose,
X_t = inv(E_mean,t) E_c,s X, and every target camera c' is rendered at inv(E_mean,t) E_c',t cam2ego_c' (its own exposure
pose); the owner camera of a Gaussian is the pipeline's pixel_camera_indices.  Same samples, same scoring.

2026-10-30 geometry-source variants (--variants decoder,da3frame,da3clip): the decoder's Gaussians are slid along their
own rays to another range exactly like inference (dwm_pano_4d.da3_depth.rescale_gaussians on stitch-content pixels with
0.5 < range < 99 m and DA3 sky < 0.3; others keep the decoder range), then the frame-level leave-one-out is scored:
  decoder   no change;
  da3frame  per-frame DA3 (gs_scene_v3 da3_window "frame", windows t-4 / t / t+4, ground scale; the v4 inference depth);
  da3clip   one clip-level DA3 pass over the 19 frames + static depth fusion (GS_CLIPDEPTH_PLAN V2, the temporally
            consistent depth).
DA3 runs on the recorded panoramas of the clip (SceneBuilder canvases set from vae_images), poses = camera-time infos.

    python -m dwm_pano_4d.loo_decompose -c <config> [--checkpoint <decoder.pth>] --samples 40 --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dwm.common
from dwm_pano_4d.geometry import points_in_boxes, relative_ego_transform, transform_gaussians

FRAMES = (0, 4, 8)
TARGET = 1


def grad_mag(img):
    g = img.mean(0)
    dx = (g[:, 1:] - g[:, :-1]).abs()
    dy = (g[1:, :] - g[:-1, :]).abs()
    out = torch.zeros_like(g)
    out[:, :-1] += dx
    out[:-1, :] += dy
    return out


def depth_edges(depth, rel=0.10, dilate=2):
    d = depth.clamp_min(1e-3)
    pad = F.pad(d[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    h, w = d.shape
    edge = torch.zeros_like(d, dtype=torch.bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            edge |= (nb - d).abs() / torch.minimum(nb, d) > rel
    k = 2 * dilate + 1
    return F.max_pool2d(edge.float()[None, None], k, stride=1, padding=dilate)[0, 0] > 0


def affine_fit(render, target, mask):
    x = render.permute(1, 2, 0)[mask].double()                       # (N, 3)
    y = target.permute(1, 2, 0)[mask].double()
    X = torch.cat([x, torch.ones_like(x[:, :1])], 1)
    M = torch.linalg.lstsq(X, y).solution                             # (4, 3)
    full = torch.cat([render.permute(1, 2, 0).double(), torch.ones_like(render[:1].permute(1, 2, 0)).double()], -1)
    return (full @ M).float().permute(2, 0, 1).clamp(0, 1)


def shift_min(render, target, radius=2):
    h, w = target.shape[-2:]
    pad = F.pad(render[None], (radius,) * 4, mode="replicate")[0]
    best = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = pad[:, radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            e = (shifted - target).abs().mean(0)
            best = e if best is None else torch.minimum(best, e)
    return best


def pose_matrix(t, q, device):
    from pyquaternion import Quaternion
    m = np.eye(4)
    m[:3, :3] = Quaternion(q).rotation_matrix
    m[:3, 3] = np.asarray(t, dtype=np.float64)
    return torch.as_tensor(m, dtype=torch.float32, device=device)


def per_camera(g, cam, transforms):
    """Move each camera's Gaussians with its own 4x4 transform; g (N, 14), cam (N,) in 0..5."""
    parts = [transform_gaussians(g[cam == c], transforms[c]) for c in range(len(transforms)) if bool((cam == c).any())]
    return torch.cat(parts) if parts else g[:0]


def score(pipe, scene, own_scene, dynamic, target, target_c2w, minalpha, acc, sharp):
    """Render, mask and accumulate one variant; returns the per-sample (loo, own) means or None."""
    rgb, _, alpha, valid = pipe._render_nuscenes6_stitched(scene, target_c2w=target_c2w)
    own_rgb, own_depth, _, _ = pipe._render_nuscenes6_stitched(own_scene, target_c2w=target_c2w)
    rgb, alpha, own_rgb, own_depth = rgb[0].float(), alpha[0].float(), own_rgb[0].float(), own_depth[0].float()
    valid = valid.reshape(alpha.shape).bool()
    mask = valid & (alpha > minalpha)
    if dynamic.shape[0]:
        _, _, dyn_alpha, _ = pipe._render_nuscenes6_stitched(dynamic, target_c2w=target_c2w)
        mask &= ~(dyn_alpha[0] > pipe.gs4d["dynamic_pixel_alpha"])
    if int(mask.sum()) < 1000:
        return None
    raw = (rgb.clamp(0, 1) - target).abs().mean(0)
    own = (own_rgb.clamp(0, 1) - target).abs().mean(0)
    aff = affine_fit(rgb.clamp(0, 1), target, mask)
    e_aff = (aff - target).abs().mean(0)
    e_shift = shift_min(aff, target)
    own_shift = shift_min(affine_fit(own_rgb.clamp(0, 1), target, mask), target)
    edge = depth_edges(own_depth) & mask
    ne = mask & ~edge
    gm = grad_mag(target)
    q80, q50 = torch.quantile(gm[ne][:200000], 0.8), torch.quantile(gm[ne][:200000], 0.5)
    cls = {"all": mask, "edge": edge, "near": ne & (own_depth < 10), "mid": ne & (own_depth >= 10) & (own_depth < 30),
           "far": ne & (own_depth >= 30), "texture": ne & (gm >= q80), "flat": ne & (gm <= q50)}
    for c, m in cls.items():
        acc[c]["pixels"] += float(m.sum())
        for k, e in (("raw", raw), ("own", own), ("affine", e_aff), ("shift", e_shift), ("own_shift", own_shift)):
            acc[c][k] += float(e[m].sum())
    sharp.append({"loo": float(grad_mag(rgb.clamp(0, 1))[mask].mean() / gm[mask].mean()),
                  "own": float(grad_mag(own_rgb.clamp(0, 1))[mask].mean() / gm[mask].mean())})
    return {"loo_raw": float(raw[mask].mean()), "own": float(own[mask].mean()),
            "coverage": float(mask.float().mean() / valid.float().mean()), "_maps": (rgb, raw, mask, edge)}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--png", type=int, default=3)
    p.add_argument("--variants", default="frame,camera", help="frame,camera | decoder,da3frame,da3clip")
    args = p.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    from dwm_pano.evaluate_gs import ensure_single_rank_process_group
    started = ensure_single_rank_process_group(device)
    config = json.loads(args.config.read_text())
    pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(args.out.parent), config=config,
                                                  device=device, resume_from=None)
    if args.checkpoint:
        pipe.decoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
    pipe.decoder.eval()
    pipe.decoder_wrapper.eval()
    dataset = dwm.common.create_instance_from_config(config["training_dataset"])
    from dwm.datasets.common import CollateFnIgnoring
    n = len(dataset)
    picks = np.linspace(0, n - 1, args.samples).round().astype(int).tolist()
    classes = ("all", "edge", "near", "mid", "far", "texture", "flat")
    acc = {c: {k: 0.0 for k in ("pixels", "raw", "own", "affine", "shift", "own_shift")} for c in classes}
    acc_cam = {c: {k: 0.0 for k in ("pixels", "raw", "own", "affine", "shift", "own_shift")} for c in classes}
    sharp, sharp_cam, rows = [], [], []
    base = dataset.base_dataset
    minalpha = pipe.gs4d["loo_min_alpha"]
    variants = [v for v in args.variants.split(",") if v]
    geometry_mode = any(v in ("decoder", "da3frame", "da3clip") for v in variants)
    accs = {v: {c: {k: 0.0 for k in ("pixels", "raw", "own", "affine", "shift", "own_shift")} for c in classes}
            for v in variants}
    sharps = {v: [] for v in variants}
    da3 = None
    if geometry_mode and any(v.startswith("da3") for v in variants):
        from dwm_pano_4d.da3_depth import Da3Ranger
        cfg_da3 = config["pipeline"]["gs4d_da3"]
        da3 = Da3Ranger(cfg_da3["repo_src"], cfg_da3["anyview_dir"], cfg_da3["metric_dir"], device)
    for si, idx in enumerate(picks):
        batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [idx]), batch_size=1,
                                                      collate_fn=CollateFnIgnoring(keys=[]))))
        pipe._prepare_nuscenes6_supervision(batch)
        frames = torch.tensor(FRAMES, dtype=torch.long)                 # _frame_ray_origins wants a tensor
        half = batch["vae_images"][:, frames].flatten(0, 1).to(device)
        lo, po = pipe._frame_ray_origins(batch, frames)
        e2g = batch["ego_transforms"][0, frames, 0].to(device, dtype=torch.float32)
        gs, owners, preds = [], [], []
        for k, frame in enumerate(frames.tolist()):
            lat = pipe.encode_half_latents(half[k:k + 1])
            g, pred_range, _ = pipe.decoder(lat, pipe.latent_directions.unsqueeze(0), pipe.pixel_directions.unsqueeze(0), None,
                                   previous_latents=None, temporal_context_mask=0.0,
                                   latent_origins=None if lo is None else lo[k:k + 1],
                                   pixel_origins=None if po is None else po[k:k + 1], camera_origin_scale=1.0)
            g = g[0]
            preds.append(pred_range[0].float())
            owner = points_in_boxes(g[:, 0:3], batch["box_centers"][0, frame].to(device),
                                    batch["box_rotations"][0, frame].to(device),
                                    batch["box_half_sizes"][0, frame].to(device),
                                    valid=batch["box_valid"][0, frame], margin=pipe.gs4d["box_margin_m"])
            owner[~pipe.pixel_content_ray_valid.reshape(-1).to(owner.device)] = -2
            gs.append(g)
            owners.append(owner)
        t = TARGET
        target = pipe._target_rgb_for_supervision(half[t:t + 1])[0].float()            # (3, H, W)
        if geometry_mode:
            rig = pipe.nuscenes_camera_to_ego.float().to(device)
            from dwm_pano.gs_scene_v3 import SceneBuilder, SceneOptions
            from dwm_pano.recon.targets import camera_time_infos
            from dwm_pano_4d.da3_depth import rescale_gaussians
            clip_infos = camera_time_infos(base.clips[idx])
            all_halves = batch["vae_images"][0].to(device).float()
            dummy = torch.zeros(all_halves.shape[0], 1)
            res_row = {}
            for name in variants:
                if name == "decoder":
                    g_src = gs
                else:
                    opts = SceneOptions(da3_relabel=True, da3_mode="scale", box_reading="mmdet3d_v0x",
                                        da3_window="frame" if name == "da3frame" else "clip",
                                        da3_fuse=(name == "da3clip"))
                    scene = SceneBuilder(pipe, batch, clip_infos, dummy, opts, device, da3=da3)
                    for f in range(all_halves.shape[0]):
                        scene._canvas[f] = pipe._unpadded_stitch_canvas(all_halves[f:f + 1])[0].clamp(0, 1)
                    g_src = []
                    for k, frame in enumerate(frames.tolist()):
                        maps = scene.da3_maps(frame)
                        r, sky = maps["range"].float(), maps["sky"].float() >= 0.3
                        content = pipe.pixel_content_ray_valid.view(r.shape).to(device)
                        valid = content & (r > 0.5) & (r < 99.0) & ~sky
                        origins = po[k].reshape(-1, 3).to(device, torch.float32)
                        g2, _ = rescale_gaussians(gs[k], preds[k], r, valid, origins)
                        g_src.append(g2)
                own_list = []
                for k, frame in enumerate(frames.tolist()):
                    o = points_in_boxes(g_src[k][:, 0:3], batch["box_centers"][0, frame].to(device),
                                        batch["box_rotations"][0, frame].to(device),
                                        batch["box_half_sizes"][0, frame].to(device),
                                        valid=batch["box_valid"][0, frame], margin=pipe.gs4d["box_margin_m"])
                    o[~pipe.pixel_content_ray_valid.reshape(-1).to(o.device)] = -2
                    own_list.append(o)
                moved_v = [transform_gaussians(g_src[s][own_list[s] == -1], relative_ego_transform(e2g[s], e2g[t]))
                           for s in range(len(frames))]
                dyn_v = g_src[t][own_list[t] >= 0]
                res = score(pipe, torch.cat([moved_v[s] for s in range(len(frames)) if s != t] + [dyn_v]), g_src[t],
                            dyn_v, target, rig, minalpha, accs[name], sharps[name])
                if res is None:
                    res_row = None
                    break
                res.pop("_maps")
                res_row[name] = res
            if res_row:
                rows.append({"index": int(idx), **res_row})
                print(f"[loo-decomp] {si} clip {idx}: " + " ".join(f"{k} {v['loo_raw']:.4f}/{v['own']:.4f}"
                                                                     for k, v in res_row.items()), flush=True)
            continue
        # variant "frame": the training loss's frame-level (mean camera-time) poses
        moved = [transform_gaussians(gs[s][owners[s] == -1], relative_ego_transform(e2g[s], e2g[t]))
                 for s in range(len(frames))]
        dynamic_t = gs[t][owners[t] >= 0]
        loo_scene = torch.cat([moved[s] for s in range(len(frames)) if s != t] + [dynamic_t])
        rig = pipe.nuscenes_camera_to_ego.float().to(device)
        res_f = score(pipe, loo_scene, gs[t], dynamic_t, target, rig, minalpha, acc, sharp)
        # variant "camera": each camera at its own exposure pose
        from dwm_pano.recon.targets import CAM_ORDER
        infos = [base.clips[idx][int(f)] for f in frames.tolist()]
        e_cam = [[pose_matrix(i["cams"][n]["ego2global_translation"], i["cams"][n]["ego2global_rotation"], device)
                  for n in CAM_ORDER] for i in infos]
        inv_t = torch.linalg.inv(e2g[t])
        cam = pipe.pixel_camera_indices.to(device)[:, None, :].expand(-1, pipe.pixel_directions.shape[1], -1).reshape(-1)
        moved_c = [per_camera(gs[s][owners[s] == -1], cam[owners[s] == -1], [inv_t @ e_cam[s][c] for c in range(6)])
                   for s in range(len(frames))]
        dyn_c = per_camera(dynamic_t, cam[owners[t] >= 0], [inv_t @ e_cam[t][c] for c in range(6)])
        own_c = per_camera(gs[t], cam, [inv_t @ e_cam[t][c] for c in range(6)])
        c2w_c = torch.stack([inv_t @ e_cam[t][c] @ rig[c] for c in range(6)])
        res_c = score(pipe, torch.cat([moved_c[s] for s in range(len(frames)) if s != t] + [dyn_c]), own_c, dyn_c,
                      target, c2w_c, minalpha, acc_cam, sharp_cam)
        offs = [float((inv_t @ e_cam[t][c])[:3, 3].norm()) for c in range(6)]
        mean_dev = float((e2g[t] - pose_matrix(infos[t]["ego2global_translation"], infos[t]["ego2global_rotation"],
                                               device))[:3, 3].norm())
        if res_f is None or res_c is None:
            continue
        rgb, raw, mask, edge = res_f.pop("_maps")
        rgb_c, raw_c, mask_c, _ = res_c.pop("_maps")
        rows.append({"index": int(idx), "frame": res_f, "camera": res_c, "cam_offsets_m": offs,
                     "batch_vs_info_mean_pose_m": mean_dev})
        if si < args.png:
            from PIL import Image
            em, emc = (raw / 0.15).clamp(0, 1), (raw_c / 0.15).clamp(0, 1)
            panel = torch.cat([target, rgb.clamp(0, 1), torch.stack([em, em * 0.3, 1 - em]) * mask[None],
                               rgb_c.clamp(0, 1), torch.stack([emc, emc * 0.3, 1 - emc]) * mask_c[None]], 1)
            img = (F.interpolate(panel[None], scale_factor=0.5, mode="area")[0].permute(1, 2, 0).cpu().numpy() * 255)
            Image.fromarray(img.astype(np.uint8)).save(args.out.with_name(f"{args.out.stem}_sample{si}.png"))
        print(f"[loo-decomp] {si} clip {idx}: frame loo {res_f['loo_raw']:.4f} camera loo {res_c['loo_raw']:.4f} "
              f"own {res_f['own']:.4f} / {res_c['own']:.4f} cam offsets max {max(offs):.3f} m", flush=True)
    def classes_of(a):
        return {c: {"pixel_share": v["pixels"] / max(a["all"]["pixels"], 1),
                    "error_share": v["raw"] / max(a["all"]["raw"], 1e-9),
                    **{k: v[k] / max(v["pixels"], 1) for k in ("raw", "own", "affine", "shift", "own_shift")}}
                for c, v in a.items()}
    if geometry_mode:
        summary = {"samples": len(rows), "checkpoint": args.checkpoint or "config initial", "frames": FRAMES,
                   "variants": {v: {"classes": classes_of(accs[v]),
                                    "sharpness_ratio": {k: float(np.mean([s[k] for s in sharps[v]])) if sharps[v]
                                                        else None for k in ("loo", "own")}} for v in variants},
                   "rows": rows}
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
        print("LOO_DECOMP_GEOMETRY " + json.dumps({v: {k: round(x, 4) for k, x in summary["variants"][v]["classes"]["all"].items()}
                                                   for v in variants}), flush=True)
        if started:
            torch.distributed.destroy_process_group()
        return
    offs = np.array([r["cam_offsets_m"] for r in rows]) if rows else np.zeros((0, 6))
    summary = {"samples": len(rows), "checkpoint": args.checkpoint or "config initial", "frames": FRAMES,
               "classes": classes_of(acc), "classes_camera": classes_of(acc_cam),
               "sharpness_ratio": {k: float(np.mean([s[k] for s in sharp])) for k in ("loo", "own")},
               "sharpness_ratio_camera": {k: float(np.mean([s[k] for s in sharp_cam])) for k in ("loo", "own")},
               "cam_offset_m": {"median": float(np.median(offs)) if offs.size else None,
                                "p90": float(np.quantile(offs, 0.9)) if offs.size else None,
                                "max": float(offs.max()) if offs.size else None},
               "batch_vs_info_mean_pose_m_max": max((r["batch_vs_info_mean_pose_m"] for r in rows), default=None),
               "rows": rows}
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print("LOO_DECOMP " + json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    if started:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
