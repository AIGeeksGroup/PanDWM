"""Training pairs for the D2 v2 refiner (2026-10-31, PanDWM-d2-refiner-20261021/D2V2_PLAN_20261031.md).

D2 v1 (d2_make_train.py) trained on renders at the ORIGINAL pose with Gaussians culled / taken from neighbours; at test
time the refiner sees renders at a SHIFTED pose, so train and test inputs differed (report section 43; the 10-25
survey, cause 1).  Here the degradation is the round trip of DiST-4D's cycle consistency / Difix3D's cycle
reconstruction, made with the decoder and scene settings the refiner is evaluated with (G3b, GS_FT3_PLAN_20261030.md:
V2 clip DA3 + fusion, in-plane offsets, camera-time poses, v0.x boxes):

  1. the real clip's 35 frames -> G3b scene A (the ground-truth halves are its canvases);
  2. scene A rendered with every camera moved by dy laterally -> push-pull filled panos V (a "video" at the shifted
     trajectory, with the holes, stretching and edge artefacts of a dy-shift);
  3. V -> VAE latents -> G3b scene B, whose infos are the clip's moved by dy (d2v2_shift_infos: ego pose moved, LiDAR
     frame and boxes kept in place);
  4. scene B rendered back at -dy, i.e. at the recorded cameras, where the ground truth exists.

The render condition of step 4 is written in the D2 v1 format (dwm_pano_d2.data.D2RenderDataset reads it unchanged),
plus the per-latent-pixel fraction of render pixels whose content was magnified > 2x (mag_cull.angular_size, scene
B's sources against the recorded rig), for loss weighting / masks:

    <out-root>/clip<NNNNN>.pt    {"render": (35, 2, 16, 32, 288) fp16, "mask": (35, 2, 32, 288) uint8 coverage,
                                  "magmask": (35, 2, 32, 288) uint8, "meta"}
    <out-root>/clip<NNNNN>.json  meta

A fraction plain_prob of the clips (hash of seed, index) skips the round trip and renders scene A at the recorded pose
(mode "plain": teaches the refiner to leave good renders alone).  |dy| is drawn from DY_CHOICES with a random sign.
Existing clips are skipped, so tasks can be resubmitted; dev / heldout scenes are refused.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dwm.common
from dwm_pano.d2_render import canvas_halves, encode_condition, encode_halves_mean, render_stitched, unit_hash
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.export_gs_batch149 import FRAMES
from dwm_pano.export_gs_v2_clips import pushpull_fill
from dwm_pano.gs_scene_v3 import RIG_HEIGHT_M, SceneBuilder, SceneOptions, composite

VERSION = "d2v2_pairs_v1"
DY_CHOICES = (1.0, 2.0, 2.0, 3.0, 4.0, 4.0)            # |dy| in metres; the evaluated 2 and 4 m weighted double


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path, help="GS config whose training_dataset reads train")
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path, help="G3b decoder")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path, help="split_195_scene_manifest.json (dev / heldout)")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--tasks", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stride", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--plain-prob", type=float, default=0.15)
    p.add_argument("--time-budget-s", type=float, default=0.0)
    p.add_argument("--preview-dir", type=Path, default=None)
    p.add_argument("--previews", type=int, default=2)
    p.add_argument("--count-only", action="store_true")
    p.add_argument("--da3-src", required=True)
    p.add_argument("--da3-anyview", required=True)
    p.add_argument("--da3-metric", required=True)
    return p.parse_args()


def draw_shift(seed, index, plain_prob):
    """("roundtrip", dy) or ("plain", 0.0), a pure function of (seed, index)."""
    if unit_hash(seed, index, "plain") < plain_prob:
        return "plain", 0.0
    k = int(unit_hash(seed, index, "dy") * len(DY_CHOICES)) % len(DY_CHOICES)
    sign = 1.0 if unit_hash(seed, index, "sign") < 0.5 else -1.0
    return "roundtrip", sign * DY_CHOICES[k]


def _pose(t, q):
    from pyquaternion import Quaternion
    m = np.eye(4)
    m[:3, :3] = Quaternion(q).rotation_matrix
    m[:3, 3] = np.asarray(t, dtype=np.float64)
    return m


def d2v2_shift_infos(infos, dy):
    """Copies of the infos with the ego frame moved by dy along its own y axis (the cameras move with it); the
    lidar2ego translation is moved back so that ego2global @ lidar2ego (LiDAR points, gt_boxes) is unchanged."""
    shift = np.eye(4)
    shift[1, 3] = float(dy)
    out = []
    for info in infos:
        e = _pose(info["ego2global_translation"], info["ego2global_rotation"]) @ shift
        l2e = np.linalg.inv(shift) @ _pose(info["lidar2ego_translation"], info["lidar2ego_rotation"])
        new = dict(info)
        new["ego2global_translation"] = e[:3, 3].tolist()
        new["lidar2ego_translation"] = l2e[:3, 3].tolist()
        out.append(new)
    return out


def scene_options():
    """G3b's evaluation scene (GS_FT3_PLAN_20261030.md section 6: V2 + plane_offset_px 1)."""
    return SceneOptions(da3_relabel=True, da3_mode="scale", drop_bottom_rows=2, box_reading="mmdet3d_v0x",
                        da3_window="clip", da3_fuse=True, plane_offset_px=1.0)


def moved_cameras(pipe, dy):
    c2w = pipe.nuscenes_camera_to_ego.float().clone()
    c2w[:, 1, 3] += float(dy)
    return c2w


@torch.no_grad()
def render_frame(pipe, scene, t, dy, with_mag=False):
    """Scene frame t (current over 4D fill) with all cameras moved by dy: rgb (3, H, W), alpha (H, W) and, with
    with_mag, the alpha-weighted magnification (H, W) of the content against each Gaussian's source rig."""
    current, fill = scene.decode(t), scene.fill(t)
    c2w = moved_cameras(pipe, dy)
    rgb, alpha = render_stitched(pipe, current, c2w)
    if fill.shape[0]:
        rgb_f, alpha_f = render_stitched(pipe, fill, c2w)
        rgb, alpha = composite(rgb, alpha, rgb_f, alpha_f)
    alpha = torch.where(pipe.stitch_valid[None], alpha, torch.zeros_like(alpha))
    mag_map = None
    if with_mag:
        from dwm_pano.mag_cull import angular_size
        rig = current.new_tensor([0.0, 0.0, RIG_HEIGHT_M])
        tgt = rig + rig.new_tensor([0.0, float(dy), 0.0])
        parts = []
        for g, src in ((current, rig), (fill, scene.fill_sources.get(t) if fill.shape[0] else None)):
            if not g.shape[0]:
                continue
            mag = angular_size(g, tgt) / angular_size(g, src).clamp_min(1e-6)
            vis = g.clone()
            vis[:, 3:6] = torch.stack([(mag / 10).clamp(0, 1), torch.zeros_like(mag), torch.zeros_like(mag)], -1)
            parts.append(vis)
        m_rgb, m_alpha = render_stitched(pipe, parts[0], c2w)
        if len(parts) > 1:
            f_rgb, f_alpha = render_stitched(pipe, parts[1], c2w)
            m_rgb, m_alpha = composite(m_rgb, m_alpha, f_rgb, f_alpha)
        mag_map = m_rgb[0, 0] * 10.0 / m_alpha[0].clamp_min(1e-3)
    return rgb[0].clamp(0, 1), alpha[0], mag_map


def latent_fraction(pipe, mask_canvas, latents_hw):
    """(H, 4114) bool canvas -> (2, h, w) uint8 fraction per latent pixel of the two halves."""
    halves = canvas_halves(mask_canvas.float())
    factor = halves.shape[-2] // latents_hw[0]
    frac = F.avg_pool2d(halves[:, None], factor)[:, 0]
    return (frac * 255.0).round().clamp(0, 255).to(torch.uint8)


def save_preview(path, rows):
    from PIL import Image
    image = torch.cat([r.float().clamp(0, 1).cpu() if r.dim() == 3 else r.float().clamp(0, 1).cpu()[None]
                       .expand(3, -1, -1) for r in rows], dim=-2)
    image = F.interpolate(image[None], scale_factor=0.5, mode="area")[0]
    Image.fromarray((image.permute(1, 2, 0) * 255).round().byte().numpy()).save(path)


@torch.no_grad()
def main():
    args = parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    manifest = json.loads(args.manifest.read_text())
    forbidden = {item["scene_token"] for key in ("dev", "heldout") for item in manifest[key]}
    opts = scene_options()
    scene_desc = {k: getattr(opts, k) for k in ("edge_rel_h", "edge_rel_v", "window", "da3_relabel", "da3_mode",
                                                 "drop_bottom_rows", "box_reading", "da3_window", "da3_fuse",
                                                 "plane_offset_px", "static_range", "max_scale")}
    started = ensure_single_rank_process_group(device)
    try:
        config = load_config(args.config)
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl,
                                   fps_stride_tuples=[[10, args.stride]])
        cfg["base_dataset"]["sequence_length"] = FRAMES
        if cfg["base_dataset"].get("split") != "train":
            raise ValueError("the GS config's training_dataset must read the train split")
        install_strict_pano_hardseam_stub()
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        order = sorted(range(len(base)), key=lambda i: unit_hash(args.seed, i, "order"))
        if args.limit:
            order = order[:args.limit]
        mine = order[args.task::args.tasks]
        print(f"[d2v2-pairs] {len(base)} clips, {len(order)} in the order, task {args.task}/{args.tasks}: "
              f"{len(mine)}", flush=True)
        if args.count_only:
            return
        from dwm_pano_4d.da3_depth import Da3Ranger
        from dwm_pano.recon.render_ours import camera_time_infos
        da3 = Da3Ranger(args.da3_src, args.da3_anyview, args.da3_metric, device)
        torch.manual_seed(int(config["generator_seed"]))
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(args.out_root),
                                                      config=config, device=device, resume_from=None)
        pipe.vae.eval()
        pipe.decoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
        pipe.decoder.eval()
        pipe.decoder_wrapper.eval()
        pipe.rasterize_mode = "antialiased"
        from dwm.datasets.common import CollateFnIgnoring
        args.out_root.mkdir(parents=True, exist_ok=True)
        job_start, last, done, previews = time.monotonic(), 0.0, 0, 0
        for index in mine:
            out = args.out_root / f"clip{index:05d}.pt"
            meta_path = out.with_suffix(".json")
            if meta_path.is_file() and out.is_file():
                continue
            if args.time_budget_s and last and time.monotonic() - job_start + 1.3 * last > args.time_budget_s:
                print(f"[d2v2-pairs] budget reached after {done} clips", flush=True)
                break
            audit = base.clip_audit[index]
            if audit["scene_token"] in forbidden:
                print(f"[d2v2-pairs] clip {index} is dev/heldout, skipped", flush=True)
                continue
            t0 = time.monotonic()
            mode, dy = draw_shift(args.seed, index, args.plain_prob)
            batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [index]), batch_size=1,
                                                          shuffle=False, num_workers=0,
                                                          collate_fn=CollateFnIgnoring(keys=[]))))
            if batch["split"][0] != "train":
                raise ValueError(f"clip {index} is not a train clip")
            halves = batch["vae_images"][0].to(device).float()                           # (35, 2, 3, H, W)
            infos = camera_time_infos(base.clips[index])
            scene_a = SceneBuilder(pipe, batch, infos, encode_halves_mean(pipe, halves).cpu(), opts, device, da3=da3)
            for t in range(FRAMES):
                scene_a._canvas[t] = pipe._unpadded_stitch_canvas(halves[t:t + 1])[0].clamp(0, 1)
            if mode == "plain":
                scene_b, back = scene_a, 0.0
            else:
                shifted = []
                for t in range(FRAMES):                                  # step 2: the "video" at the shifted pose
                    rgb, alpha, _ = render_frame(pipe, scene_a, t, dy)
                    shifted.append(pushpull_fill(rgb, alpha).clamp(0, 1))
                del scene_a
                v_halves = torch.stack([canvas_halves(c) for c in shifted])              # (35, 2, 3, H, 2304)
                scene_b = SceneBuilder(pipe, batch, d2v2_shift_infos(infos, dy),
                                       encode_halves_mean(pipe, v_halves).cpu(), opts, device, da3=da3)
                for t in range(FRAMES):
                    scene_b._canvas[t] = shifted[t]
                back = -dy
            renders, masks, magmasks, coverage, magfrac = [], [], [], [], []
            for t in range(FRAMES):                                      # step 4: back at the recorded cameras
                rgb, alpha, mag_map = render_frame(pipe, scene_b, t, back, with_mag=True)
                lat, mask, canvas = encode_condition(pipe, rgb, alpha)
                magnified = (mag_map > 2.0) & (alpha >= 0.5) & pipe.stitch_valid
                renders.append(lat.cpu())
                masks.append(mask.cpu())
                magmasks.append(latent_fraction(pipe, magnified, lat.shape[-2:]).cpu())
                coverage.append(float((alpha[pipe.stitch_valid] >= 0.5).float().mean()))
                magfrac.append(float(magnified[pipe.stitch_valid].float().mean()))
                if t == 10 and args.preview_dir is not None and previews < args.previews:
                    args.preview_dir.mkdir(parents=True, exist_ok=True)
                    gt = pipe._unpadded_stitch_canvas(halves[t:t + 1])[0]
                    save_preview(args.preview_dir / f"clip{index:05d}_{mode}_{dy:+.0f}.png",
                                 [gt, canvas, alpha, (mag_map / 5).clamp(0, 1)])
                    previews += 1
            meta = {"version": VERSION, "gs_index": int(index), "scene_token": audit["scene_token"],
                    "anchor_timestamp_us": int(base.clips[index][0]["timestamp"]),
                    "frame_timestamps_us": [int(x) for x in batch["frame_timestamps_us"][0].tolist()],
                    "mode": mode, "dx": 0.0, "dy": dy, "coverage": coverage,
                    "coverage_mean": sum(coverage) / len(coverage), "magnified_mean": sum(magfrac) / len(magfrac),
                    "seed": args.seed, "stride_s": args.stride, "checkpoint": str(args.checkpoint),
                    "scene": scene_desc, "scene_stats": scene_b.summary()}
            if meta["frame_timestamps_us"][0] != meta["anchor_timestamp_us"]:
                raise RuntimeError(f"clip {index}: first frame timestamp is not the anchor")
            tmp = out.with_suffix(".tmp")
            torch.save({"render": torch.stack(renders), "mask": torch.stack(masks), "magmask": torch.stack(magmasks),
                        "meta": meta}, tmp)
            tmp.replace(out)
            meta_path.write_text(json.dumps(meta) + "\n")
            done += 1
            last = time.monotonic() - t0
            print(f"[d2v2-pairs] clip {index} {mode} dy={dy:+.1f} cov={meta['coverage_mean']:.3f} "
                  f"mag={meta['magnified_mean']:.3f} ({last:.0f}s)", flush=True)
        print(f"D2V2_PAIRS_TASK_DONE task={args.task} wrote={done}", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
