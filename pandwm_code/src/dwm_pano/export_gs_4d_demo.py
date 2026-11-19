"""Training-free 4D Gaussian demo from a saved 195-frame closed-loop rollout.

WorldSplat-style composition at inference time, with no new training:

1. every frame is decoded independently by the G2 Gaussian decoder (its own
   ego frame) and moved into one world frame (the first frame's ego frame)
   with the logged ego poses;
2. Gaussians whose centre lies inside a (0.3 m padded) 3D box of that frame
   are *dynamic*; Gaussians farther than ``--near-range`` (sky, far facades)
   are a per-frame *far shell*; everything else is *static*;
3. static Gaussians from every ``--static-stride``-th frame are pooled and
   voxel-deduplicated into one persistent background map;
4. frame t is rendered as  static map  +  frame t's dynamic + far layers.

Outputs: chase-camera video (per-frame 3D | pooled 4D side by side), a fixed
"drone" camera video over the whole trajectory, ``static_world.ply``, per-frame
``dynamic/frameNNN.ply`` and ``trajectory.json`` for the interactive viewer.

Depth comes from the decoder (about 5 m L1 against estimated depth), so the
pooled map shows ghosting where frames disagree.  Qualitative demo only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from pyquaternion import Quaternion

import dwm.common
from dwm.utils.preview import save_tensor_to_video
from dwm_pano.evaluate_gs import (
    build_dataset_config,
    ensure_single_rank_process_group,
    install_strict_pano_hardseam_stub,
    load_config,
)
from dwm_pano.export_gs_long_demo import (
    CONTEXT,
    FRAMES,
    load_rollout_latents,
    look_at_c2w,
    pinhole,
    write_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--data-pkl", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--scene-token", required=True)
    parser.add_argument("--latent-source", choices=("rollout", "gt"),
                        default="rollout")
    parser.add_argument("--rollout-dir", type=Path)
    parser.add_argument("--static-stride", type=int, default=3)
    parser.add_argument("--near-range", type=float, default=60.0)
    # Pooled-map range and multi-frame consistency: a voxel survives only if
    # at least --min-frames different frames put a static Gaussian in it.
    parser.add_argument("--static-range", type=float, default=40.0)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--static-min-opacity", type=float, default=0.3)
    # Gaussians stretched across depth discontinuities (largest axis above
    # this, metres) are dropped from the pooled map; they smear every view.
    parser.add_argument("--static-max-scale", type=float, default=0.5)
    parser.add_argument("--voxel", type=float, default=0.25)
    parser.add_argument("--box-margin", type=float, default=0.3)
    parser.add_argument("--dynamic-export-stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.latent_source == "rollout" and args.rollout_dir is None:
        parser.error("--latent-source rollout needs --rollout-dir")
    return args


def ego_to_world(infos):
    """(T, 4, 4) transforms from each frame's ego frame to frame 0's ego frame."""
    e2g = []
    for info in infos:
        m = np.eye(4)
        m[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        m[:3, 3] = np.asarray(info["ego2global_translation"], dtype=np.float64)
        e2g.append(m)
    g2e0 = np.linalg.inv(e2g[0])
    return np.stack([g2e0 @ m for m in e2g])


def boxes_in_ego(info):
    """Box centres (N,3), rotations (N,3,3) and half sizes (N,3) in ego frame."""
    boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3))
    # (x, y, z, dx, dy, dz, yaw[, extra columns such as velocity]).
    boxes = boxes.reshape(boxes.shape[0], -1)[:, :7]
    if boxes.shape[1] != 7:
        return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3))
    q = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    t = np.asarray(info["lidar2ego_translation"], dtype=np.float64)
    centres = boxes[:, :3] @ q.T + t
    cos, sin = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    rz = np.zeros((boxes.shape[0], 3, 3))
    rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1], rz[:, 2, 2] = cos, -sin, sin, cos, 1
    return centres, q[None] @ rz, boxes[:, 3:6] / 2.0


def inside_boxes(means, centres, rotations, half, margin):
    """Boolean mask of Gaussian centres inside any padded box (torch, on GPU)."""
    inside = torch.zeros(means.shape[0], dtype=torch.bool, device=means.device)
    if centres.shape[0] == 0:
        return inside
    c = torch.as_tensor(centres, device=means.device, dtype=means.dtype)
    r = torch.as_tensor(rotations, device=means.device, dtype=means.dtype)
    h = torch.as_tensor(half, device=means.device, dtype=means.dtype) + margin
    for i in range(c.shape[0]):
        # Cheap sphere prefilter before the oriented-box test.
        radius = torch.linalg.norm(h[i])
        near = torch.linalg.norm(means - c[i], dim=-1) <= radius
        if not bool(near.any()):
            continue
        local = (means[near] - c[i]) @ r[i]
        hit = (local.abs() <= h[i]).all(dim=-1)
        idx = near.nonzero(as_tuple=True)[0]
        inside[idx[hit]] = True
    return inside


def quat_mul(a, b):
    """Hamilton product of (w, x, y, z) quaternions, broadcasting."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def transform_gaussians(g, m):
    """Rigidly move (N,14) Gaussians by a 4x4 transform."""
    m = torch.as_tensor(m, device=g.device, dtype=g.dtype)
    out = g.clone()
    out[:, 0:3] = g[:, 0:3] @ m[:3, :3].T + m[:3, 3]
    q = torch.as_tensor(Quaternion(matrix=m[:3, :3].double().cpu().numpy(),
                                   atol=1e-4).elements,
                        device=g.device, dtype=g.dtype)
    out[:, 7:11] = torch.nn.functional.normalize(quat_mul(q, g[:, 7:11]), dim=-1)
    return out


def voxel_dedup(g, frame_ids, voxel, min_frames):
    """Multi-frame consistent voxels only; keep the most opaque Gaussian each.

    Depth errors put the same surface at different depths in different
    frames, which smears along viewing rays when frames are pooled.  A voxel
    supported by at least ``min_frames`` distinct frames is kept.
    """
    keys = torch.floor(g[:, 0:3] / voxel).to(torch.int64)
    keys -= keys.min(dim=0).values
    span = keys.max(dim=0).values + 1
    flat = ((keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]).cpu().numpy()
    frames = frame_ids.cpu().numpy().astype(np.int64)
    pairs = np.unique(np.stack([flat, frames], axis=1), axis=0)
    voxels, support = np.unique(pairs[:, 0], return_counts=True)
    consistent = voxels[support >= min_frames]
    keep_mask = np.isin(flat, consistent)
    candidates = np.nonzero(keep_mask)[0]
    opacity = g[:, 6].cpu().numpy()[candidates]
    order = candidates[np.argsort(-opacity, kind="stable")]
    _, first = np.unique(flat[order], return_index=True)
    return g[torch.as_tensor(order[first], device=g.device)]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    (args.out / "dynamic").mkdir(parents=True)
    config = load_config(args.config)
    torch.manual_seed(int(config["generator_seed"]))
    torch.cuda.set_device(0)
    started = ensure_single_rank_process_group(torch.device("cuda", 0))
    device = torch.device("cuda", 0)
    try:
        pipeline = dwm.common.create_instance_from_config(
            config["pipeline"], output_path=str(args.out), config=config,
            device=device, resume_from=None)
        pipeline.vae.eval()
        pipeline.decoder.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu"), strict=True)
        pipeline.decoder.eval()
        pipeline.decoder_wrapper.eval()
        pipeline.rasterize_mode = "antialiased"
        pipeline.active_rasterize_eps2d = 0.1

        install_strict_pano_hardseam_stub()
        dataset_config = build_dataset_config(
            config["training_dataset"], pkl_path=args.data_pkl,
            fps_stride_tuples=[[10, 20]])
        dataset_config["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(dataset_config)
        base = dataset.base_dataset
        matches = [i for i, a in enumerate(base.clip_audit)
                   if a["scene_token"] == args.scene_token]
        if len(matches) != 1:
            raise ValueError(f"scene token has {len(matches)} {FRAMES}-frame clips")
        infos = base.clips[matches[0]]
        from dwm.datasets.common import CollateFnIgnoring
        batch = next(iter(torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, matches), batch_size=1,
            shuffle=False, num_workers=0,
            # The plain GS dataset has no clip_text; default collation suffices.
            collate_fn=CollateFnIgnoring(keys=[]))))
        images = batch["vae_images"]

        def encode(starts):
            return torch.cat([pipeline.encode_half_latents(
                images[0, s:s + 2].to(device)) for s in starts], dim=0)

        if args.latent_source == "gt":
            latents = encode(range(0, FRAMES, 2))[:FRAMES].cpu()
        else:
            latents = load_rollout_latents(
                args.rollout_dir, encode(range(0, CONTEXT, 2))[:CONTEXT])

        world = ego_to_world(infos)
        trajectory = world[:, :3, 3]
        content_valid = getattr(pipeline, "pixel_content_ray_valid", None)

        def decode(t):
            selector = torch.tensor([t], dtype=torch.long)
            lo, po = pipeline._frame_ray_origins(batch, selector)
            g = pipeline.decoder_wrapper(
                latents[t:t + 1].to(device, dtype=pipeline.latent_directions.dtype),
                pipeline.latent_directions.unsqueeze(0),
                pipeline.pixel_directions.unsqueeze(0),
                None, previous_latents=None, temporal_context_mask=0.0,
                latent_origins=lo, pixel_origins=po, camera_origin_scale=1.0)[0][0]
            if content_valid is not None and content_valid.numel() == g.shape[0]:
                g = g[content_valid.reshape(-1).to(g.device)]
            centres, rotations, half = boxes_in_ego(infos[t])
            dynamic = inside_boxes(g[:, 0:3], centres, rotations, half, args.box_margin)
            far = torch.linalg.norm(g[:, 0:2], dim=-1) > args.near_range
            return g, dynamic, far

        # Pass 1: pool the static background in the world frame.
        pooled, pooled_frames = [], []
        for t in range(0, FRAMES, args.static_stride):
            g, dynamic, far = decode(t)
            in_range = torch.linalg.norm(g[:, 0:2], dim=-1) <= args.static_range
            keep = (~dynamic & ~far & in_range & (g[:, 6] >= args.static_min_opacity)
                    & (g[:, 11:14].amax(dim=-1) <= args.static_max_scale))
            pooled.append(transform_gaussians(g[keep], world[t]).cpu())
            pooled_frames.append(torch.full((int(keep.sum()),), t, dtype=torch.int32))
            if t % 30 == 0:
                print(f"[gs-4d] pooled frame {t}", flush=True)
        static = voxel_dedup(torch.cat(pooled).to(device), torch.cat(pooled_frames),
                             args.voxel, args.min_frames)
        del pooled, pooled_frames
        static_count = write_ply(args.out / "static_world.ply", static,
                                 torch.ones(static.shape[0], dtype=torch.bool,
                                            device=static.device))
        print(f"[gs-4d] static map: {static_count} Gaussians", flush=True)

        # Pass 2: render per-frame 3D vs pooled 4D, export dynamic layers.
        K1 = pinhole(args.width, args.height, args.fov_deg)
        K = torch.tensor(np.stack([K1]), device=device, dtype=static.dtype)
        chase_ego = look_at_c2w([-4.0, 0.0, 2.5], [10.0, 0.0, 1.0])
        follow_drone_ego = look_at_c2w([-10.0, 0.0, 6.0], [10.0, 0.0, 0.0])
        centre = trajectory.mean(axis=0)
        heading = trajectory[-1] - trajectory[0]
        heading = heading / max(np.linalg.norm(heading[:2]), 1e-6)
        side = np.array([-heading[1], heading[0], 0.0])
        extent = max(np.linalg.norm(trajectory[-1] - trajectory[0]), 20.0)
        drone_c2w = look_at_c2w(
            trajectory[0] - heading * 0.4 * extent + side * 0.3 * extent
            + [0.0, 0.0, 0.35 * extent + 8.0], centre)
        dw, dh = 1280, 720
        Kd = torch.tensor(np.stack([pinhole(dw, dh, 75.0)]), device=device,
                          dtype=static.dtype)

        def render(g, c2w, k, w, h):
            rgb, _, _ = pipeline.render_targets(
                g, target_c2w=torch.as_tensor(c2w[None], device=device, dtype=g.dtype),
                target_intrinsics=k, width=w, height=h)
            return rgb[0].clamp(0.0, 1.0)

        chase_rows, drone_rows, dynamic_records = [], [], []
        for t in range(FRAMES):
            g, dynamic, far = decode(t)
            per_frame_world = transform_gaussians(g[dynamic | far], world[t])
            composed = torch.cat([static, per_frame_world])
            # Elevated views: the far shell (sky, distant facades) would cover
            # the map from above, so show only the pooled map + dynamics.
            dynamic_world = transform_gaussians(g[dynamic], world[t])
            near_scene = torch.cat([static, dynamic_world])
            c2w_world = world[t] @ chase_ego
            single = render(transform_gaussians(g, world[t]), c2w_world, K,
                            args.width, args.height)
            pooled_view = render(composed, c2w_world, K, args.width, args.height)
            chase_rows.append(torch.cat([single, pooled_view], dim=-1).cpu())
            drone_rows.append(torch.cat([
                render(near_scene, world[t] @ follow_drone_ego, Kd, dw, dh),
                render(near_scene, drone_c2w, Kd, dw, dh)], dim=-1).cpu())
            if t % args.dynamic_export_stride == 0:
                n = write_ply(args.out / "dynamic" / f"frame{t:03d}.ply",
                              dynamic_world, dynamic_world[:, 6] >= 0.05)
                dynamic_records.append({"frame": t, "gaussians": n})
            if t % 25 == 0:
                print(f"[gs-4d] rendered frame {t}/{FRAMES}", flush=True)

        fps = float(batch["fps"][0])
        save_tensor_to_video(str(args.out / "4d_chase_3d_vs_4d.mp4"),
                             "libx264", fps, chase_rows)
        save_tensor_to_video(str(args.out / "4d_drone.mp4"),
                             "libx264", fps, drone_rows)
        (args.out / "trajectory.json").write_text(json.dumps({
            "frame_to_world": world.tolist(),
            "fps": fps,
            "world_frame": "ego frame of frame 0: x fwd, y left, z up (m)",
            "drone_c2w_opencv": drone_c2w.tolist(),
        }) + "\n")
        record = {
            "purpose": "training-free 4D Gaussian composition demo (qualitative)",
            "scene_token": args.scene_token,
            "latent_source": args.latent_source,
            "rollout_dir": str(args.rollout_dir) if args.rollout_dir else None,
            "static_gaussians": static_count,
            "static_stride": args.static_stride, "voxel_m": args.voxel,
            "near_range_m": args.near_range, "box_margin_m": args.box_margin,
            "dynamic_frames": dynamic_records,
            "chase_video_columns": ["per-frame 3D (frame t only)",
                                    "4D: pooled static map + frame t dynamic/far"],
            "drone_video_columns": ["follow drone: 10 m behind, 6 m up",
                                    "fixed overview of the whole trajectory"],
            "static_max_scale_m": args.static_max_scale,
            "static_range_m": args.static_range,
            "min_supporting_frames": args.min_frames,
            "caveat": "decoder depth ~5 m L1; pooled map ghosts where frames disagree",
        }
        (args.out / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "dynamic_frames"},
                         indent=2), flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
