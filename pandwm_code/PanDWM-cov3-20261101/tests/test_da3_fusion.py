"""CPU tests for dwm_pano/da3_fusion.py (2026-10-26, GS_CLIPDEPTH_PLAN_20261026.md section 5).

Synthetic rig: six pinhole cameras at 1.6 m on the ego origin, yaw 0, 60, ..., 300 deg (camera z forward, x right,
y down), a scene of four vertical walls 25 m from the start, the ego driving +x at 1 m / frame.

    PYTHONPATH=src python3 -m pytest -q tests/test_da3_fusion.py
"""

import math
import sys
import types

import numpy as np
import torch

try:
    import dwm_pano.export_gs_4d_demo  # noqa: F401
except Exception:                                   # heavy deps missing locally: provide inside_boxes only
    stub = types.ModuleType("dwm_pano.export_gs_4d_demo")

    def inside_boxes(means, centres, rotations, half, margin):
        inside = torch.zeros(means.shape[0], dtype=torch.bool)
        for c, r, h in zip(centres, rotations, half):
            local = (means - torch.as_tensor(c, dtype=means.dtype)) @ torch.as_tensor(r, dtype=means.dtype)
            inside |= (local.abs() <= torch.as_tensor(h, dtype=means.dtype) + margin).all(dim=-1)
        return inside
    stub.inside_boxes = inside_boxes
    sys.modules["dwm_pano.export_gs_4d_demo"] = stub

from dwm_pano.da3_fusion import ClipGeometry, crop_rays, fuse_static_depth, temporal_consistency  # noqa: E402

H, W, F_ = 24, 40, 9
NO_BOXES = (np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3)))


def rig():
    K = torch.tensor([[20.0, 0, W / 2], [0, 20.0, H / 2], [0, 0, 1]]).repeat(6, 1, 1)
    cam2ego = torch.zeros(6, 4, 4)
    base = torch.tensor([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])          # camera axes (x, y, z) in ego for yaw 0
    for k in range(6):
        a = math.radians(60 * k)
        rz = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        cam2ego[k, :3, :3] = rz @ base
        cam2ego[k, :3, 3] = torch.tensor([0.0, 0.0, 1.6])
        cam2ego[k, 3, 3] = 1.0
    return K, cam2ego


def poses():
    e2r = torch.eye(4).repeat(F_, 1, 1)
    e2r[:, 0, 3] = torch.arange(F_, dtype=torch.float32)
    return e2r


def true_depth(K, cam2ego, e2r):
    """Exact z of the walls x = +-25, y = +-25 (reference frame) for every frame / camera / pixel."""
    rays = crop_rays(K, H, W)                                           # (6, h, w, 3)
    out = torch.empty(F_, 6, H, W)
    for f in range(F_):
        c2r = e2r[f] @ cam2ego
        d = torch.einsum("kij,khwj->khwi", c2r[:, :3, :3], rays)        # direction per unit z
        o = c2r[:, None, None, :3, 3].expand_as(d)
        best = torch.full(d.shape[:-1], float("inf"))
        for axis, plane in ((0, 25.0), (0, -25.0), (1, 25.0), (1, -25.0)):
            t = (plane - o[..., axis]) / d[..., axis]
            best = torch.where((t > 0) & (t < best), t, best)
        out[f] = best
    return out


def geometry(depth, boxes=None, tol_owned=True):
    K, cam2ego = rig()
    sky = torch.zeros_like(depth)
    owned = torch.ones(6, H, W, dtype=torch.bool) if tol_owned else torch.zeros(6, H, W, dtype=torch.bool)
    return ClipGeometry(depth, sky, K, owned, cam2ego, poses(), boxes or [NO_BOXES] * F_, far=99.0, box_margin=0.3)


def test_consistent_depth_is_kept_and_reprojects_exactly():
    K, cam2ego = rig()
    z = true_depth(K, cam2ego, poses())
    geo = geometry(z)
    fused, stats = fuse_static_depth(geo, (-2, -1, 1, 2), 0.1)
    assert stats["fuse_replaced_fraction"] > 0.8
    rel = ((fused - z).abs() / z)
    assert float(rel.max()) < 0.06                     # nearest-pixel z-buffer error only
    tc = temporal_consistency(geo)
    assert tc["temporal_pairs"] == F_ - 1 and tc["temporal_rel_median"] < 0.01


def test_fusion_removes_per_frame_scale_jitter():
    K, cam2ego = rig()
    z = true_depth(K, cam2ego, poses())
    g = torch.Generator().manual_seed(0)                # per-frame scale jitter, +-5 % uniform, no parity pattern
    jitter = (1.0 + (torch.rand(F_, generator=g) - 0.5) * 0.1)[:, None, None, None]
    noisy = z * jitter
    geo = geometry(noisy)
    before = temporal_consistency(geo)["temporal_rel_median"]
    fused, _ = fuse_static_depth(geo, (-8, -6, -4, -2, 2, 4, 6, 8), 0.1)
    after = temporal_consistency(geo, fused)["temporal_rel_median"]
    # even offsets: frames t and t - 1 take medians of disjoint parity sets (~5 values each), so the consecutive-
    # frame difference shrinks by roughly 1 / sqrt(2) to 1 / 2, not to zero
    assert after < 0.8 * before, (before, after)
    err_before = float(((noisy - z).abs() / z).mean())
    err_after = float(((fused - z).abs() / z).mean())
    assert err_after < err_before


def test_occlusion_tolerance_rejects_far_candidates():
    K, cam2ego = rig()
    z = true_depth(K, cam2ego, poses())
    bad = z.clone()
    bad[4] = z[4] * 1.5                               # one frame 50 % off: outside the 10 % tolerance everywhere
    geo = geometry(bad)
    fused, _ = fuse_static_depth(geo, (-1, 1), 0.1)
    assert torch.allclose(fused[4], bad[4])           # its neighbours are all rejected, own depth kept
    assert float(((fused[3] - z[3]).abs() / z[3]).max()) < 0.06   # frame 4 never pollutes frame 3


def test_box_pixels_and_non_owned_pixels_keep_own_depth():
    K, cam2ego = rig()
    z = true_depth(K, cam2ego, poses())
    noisy = z * torch.tensor([1.0, 1.05, 0.95, 1.05, 0.95, 1.05, 0.95, 1.05, 0.95])[:, None, None, None]
    # a box over the x = 25 wall, given in each frame's own ego (the ego is at x = f in the reference frame)
    boxes = [(np.array([[25.0 - f, 0.0, 1.6]]), np.eye(3)[None], np.array([[2.0, 30.0, 5.0]])) for f in range(F_)]
    geo = geometry(noisy, boxes=boxes)
    fused, _ = fuse_static_depth(geo, (-2, 2), 0.1)
    front = 0                                          # camera 0 looks at +x
    hit = ~geo.mask[4][front]
    assert bool(hit.any())
    assert torch.equal(fused[4, front][hit], noisy[4, front][hit])
    geo_none = geometry(noisy, tol_owned=False)
    fused_none, stats = fuse_static_depth(geo_none, (-2, 2), 0.1)
    assert torch.equal(fused_none, noisy) and stats["fuse_replaced_fraction"] is None
