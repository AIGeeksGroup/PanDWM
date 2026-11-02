"""GS render conditions for the D2 refiner (2026-10-21, PanDWM-d2-refiner-20261021/D2_PLAN_20261021.md Task 5).

Frame t of a gs_scene_v3 scene (current frame's Gaussians over the static Gaussians of nearby frames) is rendered at
the six nuScenes cameras and stitched with the pipeline's hard seam into the (3, 256, 4114) canvas, as in
eval_gs_shift.py / export_shift_pano_latents.py.  Modes:

* shift      (inference): rig moved by offset = (dx, dy) metres in the ego frame; current frame + fill.
* plain      (training):  original rig; current frame + fill.
* cull       (training):  original rig; the current frame keeps only the Gaussians that the rig moved by offset
                          would capture (inside its +/-16.7 deg band about the rig centre at 1.6 m, and not hidden
                          behind the expected depth of its own render), then + fill.  The holes are those a real
                          shift by -offset would open, seen from the other side.
* neighbors  (training):  original rig; the current frame keeps only its Gaussians inside the padded 3D boxes, so
                          the static scene comes from the nearby frames alone (their depth errors show as
                          misplacement and stretching, as on a real novel view).

encode_condition() fills the holes (alpha < 0.5) with pushpull_fill, cuts the canvas back into the two 2304-wide
halves (left = canvas[..., :2304], right = canvas[..., 1810:]) and VAE-encodes them with the posterior MEAN and
the video model's shift / scaling (the same latent space as the refiner's targets); the coverage mask is the
8 x 8 mean of (alpha >= 0.5) per half at latent resolution, stored as uint8 (x 255).
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

from dwm_pano.export_gs_4d_demo import boxes_in_ego, inside_boxes
from dwm_pano.export_gs_v2_clips import pushpull_fill
from dwm_pano.gs_geometry import PANO_LAT_HALF_RAD
from dwm_pano.gs_scene_v3 import RIG_HEIGHT_M, composite

LEFT_W = 2304
MODES = ("shift", "plain", "cull", "neighbors")
MODE_PROBS = (("cull", 0.70), ("neighbors", 0.15), ("plain", 0.15))
DY_RANGE = (0.5, 4.0)                                   # |dy| = lo + (hi - lo) * u^2: median 1.4 m, 90% 3.3 m
DX_RANGE = (-1.5, 1.5)


def unit_hash(*parts) -> float:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2 ** 64


def draw_degradation(seed, index):
    """(mode, dx, dy) of a training clip, a pure function of (seed, index)."""
    u_mode, u_dy, u_sign, u_dx = (unit_hash(seed, index, k) for k in ("mode", "dy", "sign", "dx"))
    edge, mode = 0.0, MODE_PROBS[-1][0]
    for name, p in MODE_PROBS:
        edge += p
        if u_mode < edge:
            mode = name
            break
    if mode != "cull":
        return mode, 0.0, 0.0
    dy = (DY_RANGE[0] + (DY_RANGE[1] - DY_RANGE[0]) * u_dy ** 2) * (1.0 if u_sign < 0.5 else -1.0)
    dx = DX_RANGE[0] + (DX_RANGE[1] - DX_RANGE[0]) * u_dx
    return mode, float(dx), float(dy)


def moved_rig(pipe, offset):
    c2w = pipe.nuscenes_camera_to_ego.float().clone()
    c2w[:, 0, 3] += float(offset[0])
    c2w[:, 1, 3] += float(offset[1])
    return c2w


def in_band_mask(g, offset):
    """Gaussians inside the +/-16.7 deg band of the rig centre moved by offset (ego frame, height 1.6 m)."""
    centre = g.new_tensor([float(offset[0]), float(offset[1]), RIG_HEIGHT_M])
    rel = g[:, 0:3] - centre
    horizontal = torch.linalg.norm(rel[:, 0:2], dim=-1).clamp_min(1e-3)
    return torch.atan2(rel[:, 2], horizontal).abs() <= PANO_LAT_HALF_RAD


@torch.no_grad()
def visible_mask(pipe, g, c2w, rel=0.03, abs_m=0.3):
    """Gaussians whose centre is not hidden in at least one of the six cameras c2w (expected-depth test)."""
    if g.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=g.device)
    K = pipe.nuscenes_render_intrinsics.float()
    _, depth, alpha = pipe.render_targets(g, target_c2w=c2w, target_intrinsics=K,
                                          width=pipe.nuscenes_render_width, height=pipe.nuscenes_render_height)
    height, width = depth.shape[-2:]
    w2c = torch.linalg.inv(c2w)
    cam = torch.einsum("kij,nj->kni", w2c[:, :3, :3], g[:, 0:3].float()) + w2c[:, None, :3, 3]
    z = cam[..., 2]
    zc = z.clamp_min(1e-6)
    u = K[:, None, 0, 0] * cam[..., 0] / zc + K[:, None, 0, 2]
    v = K[:, None, 1, 1] * cam[..., 1] / zc + K[:, None, 1, 2]
    near = float(pipe.training_config["near_plane"])
    inside = (z > near) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    flat = (v.round().long().clamp(0, height - 1) * width + u.round().long().clamp(0, width - 1))
    d = depth.flatten(1).gather(1, flat)
    a = alpha.flatten(1).gather(1, flat)
    seen = inside & ((a < 0.5) | (z <= d * (1.0 + rel) + abs_m))
    return seen.any(dim=0)


def render_stitched(pipe, g, c2w):
    """(1, 3, H, W) rgb and (1, H, W) alpha of Gaussians g at the six cameras c2w; empty sets render empty."""
    if g.shape[0] == 0:
        h, w = pipe.stitch_valid.shape[-2:]
        return (torch.zeros(1, 3, h, w, device=pipe.stitch_valid.device),
                torch.zeros(1, h, w, device=pipe.stitch_valid.device))
    rgb, _, alpha, _ = pipe._render_nuscenes6_stitched(g, target_c2w=c2w)
    return rgb, alpha


@torch.no_grad()
def degraded_canvas(pipe, scene, t, mode, offset=(0.0, 0.0)):
    """Frame t's render in the given mode: ((3, H, W) rgb in [0, 1] before hole filling, (H, W) alpha)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    current = scene.decode(t)
    if mode == "cull":
        keep = in_band_mask(current, offset)
        keep &= visible_mask(pipe, current, moved_rig(pipe, offset))
        current = current[keep]
    elif mode == "neighbors":
        c, r, h = boxes_in_ego(scene.infos[t])
        current = current[inside_boxes(current[:, 0:3], c, r, h, scene.opts.box_margin)]
    target = moved_rig(pipe, offset) if mode == "shift" else moved_rig(pipe, (0.0, 0.0))
    rgb, alpha = render_stitched(pipe, current, target)
    fill = scene.fill(t)
    if fill.shape[0]:
        rgb_f, alpha_f = render_stitched(pipe, fill, target)
        rgb, alpha = composite(rgb, alpha, rgb_f, alpha_f)
    alpha = torch.where(pipe.stitch_valid[None], alpha, torch.zeros_like(alpha))
    return rgb[0].clamp(0, 1), alpha[0]


def canvas_halves(x):
    """(..., H, 4114) canvas -> (2, ..., H, 2304) halves (left = [:2304], right = [W - 2304:])."""
    width = x.shape[-1]
    return torch.stack([x[..., :LEFT_W], x[..., width - LEFT_W:]])


@torch.no_grad()
def encode_halves_mean(pipe, halves):
    """(N, 2, 3, H, W) images in [0, 1] -> (N, 2, C, h, w) latents (posterior mean, video-model scaling)."""
    vae = pipe.vae
    flat = halves.flatten(0, 1).to(device=pipe.device, dtype=vae.dtype) * 2.0 - 1.0
    shift = vae.config.shift_factor or 0.0
    out = []
    for part in flat.split(4):
        out.append((vae.encode(part).latent_dist.mean - shift) * vae.config.scaling_factor)
    return torch.cat(out).float().unflatten(0, (halves.shape[0], 2))


@torch.no_grad()
def encode_condition(pipe, rgb, alpha):
    """-> (render (2, C, h, w) fp16, mask (2, h, w) uint8, filled canvas (3, H, W))."""
    canvas = pushpull_fill(rgb, alpha).clamp(0, 1)
    latents = encode_halves_mean(pipe, canvas_halves(canvas)[None])[0]
    covered = canvas_halves((alpha >= 0.5).float())                       # (2, H, 2304)
    factor = covered.shape[-2] // latents.shape[-2]
    mask = F.avg_pool2d(covered[:, None], factor)[:, 0]
    if tuple(mask.shape[-2:]) != tuple(latents.shape[-2:]):
        raise RuntimeError(f"mask {tuple(mask.shape)} does not match latents {tuple(latents.shape)}")
    return latents.half(), (mask * 255.0).round().clamp(0, 255).to(torch.uint8), canvas
