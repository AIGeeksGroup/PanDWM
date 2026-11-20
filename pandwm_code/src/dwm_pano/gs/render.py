"""Shared rendering helpers for the Gaussian scene builders and evaluators.

Camera construction (look-at, pinhole and rig-height viewpoints), ego-pose and 3D-box utilities, rigid transforms of
(N, 14) Gaussians, the depth-edge mask and the push-pull fill used when Gaussians are rendered at a novel view.
FRAMES is the 35-frame clip length of the evaluation protocol.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from pyquaternion import Quaternion


def look_at_c2w(position, target):
    """OpenCV camera (x right, y down, z forward) in the ego frame (z up)."""
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, position
    return c2w


def pinhole(width, height, fov_deg):
    focal = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[focal, 0.0, width / 2.0],
                     [0.0, focal, height / 2.0],
                     [0.0, 0.0, 1.0]])


def intrinsics(width, hfov_deg, up_deg, down_deg):
    focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    up = focal * math.tan(math.radians(up_deg))
    height = int(round(up + focal * math.tan(math.radians(down_deg))))
    height += height % 2
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, up], [0.0, 0.0, 1.0]]), height


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


RIG_HEIGHT = 1.6  # nuScenes cameras sit about 1.5-1.7 m above the ground


def yaw_c2w(position, yaw_rad, pitch_rad=0.0):
    direction = [math.cos(yaw_rad) * math.cos(pitch_rad),
                 math.sin(yaw_rad) * math.cos(pitch_rad),
                 math.sin(pitch_rad)]
    return look_at_c2w(position, np.asarray(position) + 10.0 * np.asarray(direction))


VIEWS = {
    # Pure rotation about the rig: no parallax, so no holes or stretching.
    "pan": ("look around: camera stays at the rig, turns +/-40 deg",
            lambda t: yaw_c2w([0.0, 0.0, RIG_HEIGHT],
                              math.radians(40.0) * math.sin(2 * math.pi * t / 60.0))),
    "chase": ("follow: 1.5 m behind, 0.3 m above the rig",
              lambda t: look_at_c2w([-1.5, 0.0, RIG_HEIGHT + 0.3],
                                    [15.0, 0.0, RIG_HEIGHT - 0.4])),
    "sweep": ("sideways: +/-1 m left-right at rig height",
              lambda t: (lambda y: look_at_c2w([0.0, y, RIG_HEIGHT],
                                               [15.0, y, RIG_HEIGHT - 0.2]))(
                  1.0 * math.sin(2 * math.pi * t / 40.0))),
    # Far viewpoint, kept for side-by-side comparison.
    "old_chase": ("old: 4 m behind, 2.5 m up",
                  lambda t: look_at_c2w([-4.0, 0.0, 2.5], [10.0, 0.0, 1.0])),
}


FRAMES = 35


def cameras(t):
    chase = look_at_c2w([-4.0, 0.0, 2.5], [10.0, 0.0, 1.0])
    y = 2.0 * math.sin(2.0 * math.pi * t / 40.0)
    sweep = look_at_c2w([0.0, y, 1.8], [15.0, y, 1.5])
    return np.stack([chase, sweep])
