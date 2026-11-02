"""Rigid-motion helpers for (N, 14) Gaussian tensors.

Layout (gsplat / PanoGaussianDecoder): [means(3), rgb(3), opacity(1),
quaternion wxyz(4), scale(3)].  Ego frames are nuScenes ego coordinates
(x forward, y left, z up, metres).
"""

from __future__ import annotations

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (..., 4) wxyz quaternions (broadcasting)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def rotation_to_quat(rotation: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) proper rotation -> (..., 4) unit wxyz quaternion.

    Shepperd's method, branch-free via the largest diagonal candidate, so it is
    stable for every rotation (including 180-degree turns).
    """
    m = rotation
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    candidates = torch.stack([
        1.0 + trace,
        1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
        1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
        1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
    ], dim=-1)
    choice = candidates.argmax(dim=-1, keepdim=True)
    s = torch.sqrt(candidates.gather(-1, choice).clamp_min(1e-12)) * 2.0
    s = s.squeeze(-1)
    w = torch.stack([
        0.25 * s,
        (m[..., 2, 1] - m[..., 1, 2]) / s,
        (m[..., 0, 2] - m[..., 2, 0]) / s,
        (m[..., 1, 0] - m[..., 0, 1]) / s], dim=-1)
    x = torch.stack([
        (m[..., 2, 1] - m[..., 1, 2]) / s,
        0.25 * s,
        (m[..., 0, 1] + m[..., 1, 0]) / s,
        (m[..., 0, 2] + m[..., 2, 0]) / s], dim=-1)
    y = torch.stack([
        (m[..., 0, 2] - m[..., 2, 0]) / s,
        (m[..., 0, 1] + m[..., 1, 0]) / s,
        0.25 * s,
        (m[..., 1, 2] + m[..., 2, 1]) / s], dim=-1)
    z = torch.stack([
        (m[..., 1, 0] - m[..., 0, 1]) / s,
        (m[..., 0, 2] + m[..., 2, 0]) / s,
        (m[..., 1, 2] + m[..., 2, 1]) / s,
        0.25 * s], dim=-1)
    # options[..., component, case]; pick the case with the largest pivot.
    options = torch.stack([w, x, y, z], dim=-2)
    index = choice.unsqueeze(-2).expand(*choice.shape[:-1], 4, 1)
    quat = options.gather(-1, index).squeeze(-1)
    quat = quat * torch.where(quat[..., :1] < 0, -1.0, 1.0)
    return torch.nn.functional.normalize(quat, dim=-1)


def transform_gaussians(gaussians: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply a 4x4 rigid transform to (N, 14) Gaussians (differentiable)."""
    transform = transform.to(device=gaussians.device, dtype=gaussians.dtype)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    means = gaussians[:, 0:3] @ rotation.T + translation
    quats = torch.nn.functional.normalize(
        quat_mul(rotation_to_quat(rotation), gaussians[:, 7:11]), dim=-1)
    return torch.cat([means, gaussians[:, 3:7], quats, gaussians[:, 11:14]], dim=-1)


def relative_ego_transform(ego_to_global_source, ego_to_global_target):
    """4x4 mapping source-ego coordinates into target-ego coordinates."""
    return torch.linalg.inv(ego_to_global_target) @ ego_to_global_source


def points_in_boxes(points, centers, rotations, half_sizes, valid=None,
                    margin=0.3, chunk=16):
    """Index (N,) of the first padded box containing each point, -1 if none.

    points (N, 3); centers (M, 3); rotations (M, 3, 3) box-to-ego; half_sizes
    (M, 3).  Boxes are tested in chunks with a bounding-sphere prefilter so
    memory stays O(N * chunk).
    """
    owner = torch.full((points.shape[0],), -1, dtype=torch.long,
                       device=points.device)
    if centers.numel() == 0:
        return owner
    centers = centers.to(points)
    rotations = rotations.to(points)
    half = half_sizes.to(points) + margin
    box_ids = torch.arange(centers.shape[0], device=points.device)
    if valid is not None:
        keep = valid.to(device=points.device, dtype=torch.bool)
        centers, rotations, half, box_ids = \
            centers[keep], rotations[keep], half[keep], box_ids[keep]
    for start in range(0, centers.shape[0], chunk):
        c = centers[start:start + chunk]
        r = rotations[start:start + chunk]
        h = half[start:start + chunk]
        radius = torch.linalg.norm(h, dim=-1)
        near = (torch.cdist(points, c) <= radius).any(dim=-1) & (owner < 0)
        if not bool(near.any()):
            continue
        idx = near.nonzero(as_tuple=True)[0]
        local = torch.einsum("nmj,mjk->nmk", points[idx, None, :] - c[None], r)
        hit = (local.abs() <= h[None]).all(dim=-1)
        any_hit = hit.any(dim=-1)
        first = hit.float().argmax(dim=-1)
        owner[idx[any_hit]] = box_ids[start + first[any_hit]]
    return owner


def box_pose(center, rotation):
    """4x4 box-to-ego pose from a centre (3,) and rotation (3, 3)."""
    pose = torch.eye(4, dtype=center.dtype, device=center.device)
    pose[:3, :3] = rotation
    pose[:3, 3] = center
    return pose
