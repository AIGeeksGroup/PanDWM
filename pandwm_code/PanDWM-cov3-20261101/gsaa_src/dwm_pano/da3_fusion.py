"""Static depth fusion across a clip for the DA3 crops (2026-10-26, GS_CLIPDEPTH_PLAN_20261026.md, arm V2).

Works in Depth Anything 3's pinhole crop space (dwm_pano_4d/da3_depth.Da3Ranger): F frames x 6 crops of metric
camera z, crop intrinsics K (pixel centres at i + 0.5), the rig's cam2ego and every frame's ego pose (to frame 0's
ego).  A crop pixel is static if it lies inside its camera's owned patch, DA3 does not call it sky (< 0.3), its range
is in (0.5, far) m and its 3D point lies outside every padded 3D box of its own frame.  For frame t, camera k, every
static pixel's own depth z0 is fused with the static points of the frames t + o (o in offsets, inside the clip),
each moved with the known ego poses into camera k of frame t and z-buffered per pixel (the nearest point wins); a
candidate counts only if |z - z0| <= tol_rel * z0 (occlusion check), and the fused depth is the median of z0 and the
accepted candidates.  Non-static pixels keep z0.

temporal_consistency() is the diagnostic of the plan: over consecutive frames, the median |z_{t-1 -> t} - z_t| / z_t
of the static pixels of frame t that receive a static point of frame t - 1.
"""

from __future__ import annotations

import torch

from dwm_pano.export_gs_4d_demo import inside_boxes

SKY_THRESHOLD = 0.3


def crop_rays(K, h, w):
    """(6, 3, 3) crop intrinsics -> (6, h, w, 3) camera rays with z = 1 through the pixel centres."""
    v, u = torch.meshgrid(torch.arange(h, device=K.device, dtype=K.dtype) + 0.5,
                          torch.arange(w, device=K.device, dtype=K.dtype) + 0.5, indexing="ij")
    pix = torch.stack([u, v, torch.ones_like(u)], dim=-1)
    return torch.einsum("kij,hwj->khwi", torch.linalg.inv(K), pix)


def static_points(depth, sky, owned, rays, cam2ego, boxes, far, box_margin):
    """One frame: depth / sky (6, h, w), owned (6, h, w) bool, rays (6, h, w, 3), cam2ego (6, 4, 4), boxes =
    (centres, rotations, half sizes) in this frame's ego.  Returns static mask (6, h, w) and ego points (6, h, w, 3)."""
    pts = torch.einsum("kij,khwj->khwi", cam2ego[:, :3, :3], rays * depth[..., None]) + cam2ego[:, None, None, :3, 3]
    rng = depth * rays.norm(dim=-1)
    mask = owned & (sky < SKY_THRESHOLD) & (rng > 0.5) & (rng < far)
    centres, rotations, half = boxes
    if len(centres) and bool(mask.any()):
        dynamic = inside_boxes(pts[mask], centres, rotations, half, box_margin)
        keep = mask.clone()
        keep[mask] = ~dynamic
        mask = keep
    return mask, pts


def zbuffer(points, world_to_cam, K, h, w):
    """points (N, 3) in the reference frame; world_to_cam (6, 4, 4); K (6, 3, 3) -> (6, h, w) nearest camera z,
    NaN where no point lands."""
    out = torch.full((K.shape[0], h * w), float("inf"), device=points.device, dtype=points.dtype)
    for k in range(K.shape[0]):
        p = points @ world_to_cam[k, :3, :3].T + world_to_cam[k, :3, 3]
        z = p[:, 2]
        ok = z > 0.1
        p, z = p[ok], z[ok]
        u = torch.floor(K[k, 0, 0] * p[:, 0] / z + K[k, 0, 2]).long()
        v = torch.floor(K[k, 1, 1] * p[:, 1] / z + K[k, 1, 2]).long()
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        out[k].scatter_reduce_(0, v[inside] * w + u[inside], z[inside], reduce="amin", include_self=True)
    out = out.view(K.shape[0], h, w)
    return torch.where(torch.isfinite(out), out, torch.full_like(out, float("nan")))


class ClipGeometry:
    """Static masks and reference-frame points of every frame of a clip (computed once)."""

    def __init__(self, depth, sky, K, owned, cam2ego, ego_to_ref, boxes, far, box_margin):
        f, c, h, w = depth.shape
        if c != 6 or K.shape != (6, 3, 3) or owned.shape != (6, h, w) or len(boxes) != f:
            raise ValueError(f"shapes: depth {tuple(depth.shape)} K {tuple(K.shape)} owned {tuple(owned.shape)} "
                             f"boxes {len(boxes)}")
        self.depth, self.K, self.h, self.w = depth.float(), K.float(), h, w
        self.cam2ego = cam2ego.float()
        self.ego_to_ref = ego_to_ref.float()
        rays = crop_rays(self.K, h, w)
        self.mask, self.points = [], []
        for i in range(f):
            m, p = static_points(self.depth[i], sky[i].float(), owned, rays, self.cam2ego, boxes[i], far, box_margin)
            p = p @ self.ego_to_ref[i, :3, :3].T + self.ego_to_ref[i, :3, 3]
            self.mask.append(m)
            self.points.append(p[m])
        self.world_to_cam = torch.linalg.inv(self.ego_to_ref[:, None] @ self.cam2ego[None])     # (F, 6, 4, 4)

    def candidate(self, s, t):
        """Static points of frame s z-buffered into the six cameras of frame t: (6, h, w), NaN where none."""
        return zbuffer(self.points[s], self.world_to_cam[t], self.K, self.h, self.w)


@torch.no_grad()
def fuse_static_depth(geo: ClipGeometry, offsets, tol_rel):
    """Fused depth (F, 6, h, w) and stats (fraction of static pixels with at least one accepted candidate, mean
    accepted candidates per static pixel, mean |fused - own| / own over static pixels)."""
    f = geo.depth.shape[0]
    fused = geo.depth.clone()
    replaced, accepted_mean, change = [], [], []
    for t in range(f):
        own = geo.depth[t]
        static = geo.mask[t]
        if not bool(static.any()):
            continue
        stack = [own]
        for o in offsets:
            s = t + int(o)
            if s < 0 or s >= f or s == t:
                continue
            cand = geo.candidate(s, t)
            ok = static & torch.isfinite(cand) & ((cand - own).abs() <= tol_rel * own)
            stack.append(torch.where(ok, cand, torch.full_like(cand, float("nan"))))
        stack = torch.stack(stack)
        med = torch.nanmedian(stack, dim=0).values
        fused[t] = torch.where(static, med, own)
        n_acc = torch.isfinite(stack[1:]).sum(dim=0)
        replaced.append(float((n_acc[static] > 0).float().mean()))
        accepted_mean.append(float(n_acc[static].float().mean()))
        change.append(float(((fused[t] - own).abs() / own.clamp_min(1e-3))[static].mean()))
    mean = lambda x: float(sum(x) / len(x)) if x else None
    return fused, {"fuse_replaced_fraction": mean(replaced), "fuse_accepted_per_pixel": mean(accepted_mean),
                   "fuse_mean_rel_change": mean(change)}


@torch.no_grad()
def temporal_consistency(geo: ClipGeometry, depth=None):
    """Median over consecutive frame pairs of the median |z_{t-1 -> t} - z_t| / z_t (static pixels of frame t hit by
    a static point of frame t - 1) and the mean fraction of those pixels within 5 %.  `depth` (F, 6, h, w) replaces
    the depth the static points were built from when given (same masks; used to score fused maps)."""
    if depth is not None:
        geo = _with_depth(geo, depth)
    medians, within = [], []
    for t in range(1, geo.depth.shape[0]):
        cand = geo.candidate(t - 1, t)
        sel = geo.mask[t] & torch.isfinite(cand)
        if int(sel.sum()) < 100:
            continue
        rel = ((cand - geo.depth[t]).abs() / geo.depth[t].clamp_min(1e-3))[sel]
        medians.append(float(rel.median()))
        within.append(float((rel < 0.05).float().mean()))
    if not medians:
        return {"temporal_rel_median": None, "temporal_within5": None, "temporal_pairs": 0}
    return {"temporal_rel_median": float(torch.tensor(medians).median()),
            "temporal_within5": float(sum(within) / len(within)), "temporal_pairs": len(medians)}


def _with_depth(geo, depth):
    """A shallow copy of geo whose depth and points follow `depth` (static masks unchanged)."""
    out = ClipGeometry.__new__(ClipGeometry)
    out.__dict__.update(geo.__dict__)
    out.depth = depth.float()
    rays = crop_rays(geo.K, geo.h, geo.w)
    out.points = []
    for i in range(depth.shape[0]):
        p = torch.einsum("kij,khwj->khwi", geo.cam2ego[:, :3, :3], rays * out.depth[i][..., None]) \
            + geo.cam2ego[:, None, None, :3, 3]
        p = p @ geo.ego_to_ref[i, :3, :3].T + geo.ego_to_ref[i, :3, 3]
        out.points.append(p[geo.mask[i]])
    return out
