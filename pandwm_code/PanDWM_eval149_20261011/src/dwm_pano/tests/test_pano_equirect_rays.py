"""Pin the analytic pano ray field against the authoritative forward projection.

`get_pano_equirect_rays` must be the exact inverse of
`pano_projection.project_ego_point_to_pano` -- that is the only thing that
makes the ray map agree with the *pixel content* of the rendered panorama.
These tests also record the geometric facts that motivated the fix, so a
regression back to the pinhole path fails loudly rather than silently.
"""
import math

import numpy as np
import pytest
import torch

from dwm_pano.datasets.pano_projection import project_ego_point_to_pano
from dwm_pano.models.crossview_temporal_dit_pano import (
    _PANO_HALF_RANGES, _PANO_SRC_H, _PANO_SRC_W, get_pano_equirect_rays)

TOKEN_H, TOKEN_W = 16, 144          # latent 32x288 with patch_size 2
TRUE_H_SPAN_DEG = 5376.0 / 9600.0 * 360.0        # 201.6
TRUE_V_SPAN_DEG = 600.0 / 900.0 * 50.0           # 33.33


def _angle_deg(a, b):
    cos = float(np.clip(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _expected_pano_pixel(half_index, row, col):
    half_start, half_end = _PANO_HALF_RANGES[half_index]
    tx = half_start + (col + 0.5) / TOKEN_W * float(half_end - half_start)
    ty = (row + 0.5) / TOKEN_H * _PANO_SRC_H
    return tx, ty


def test_pano_equirect_rays_invert_projection():
    """Round trip: ray direction -> real forward projection -> same pixel."""
    _, rays_d = get_pano_equirect_rays(len(_PANO_HALF_RANGES),
                                       (TOKEN_H, TOKEN_W))
    worst_tx = worst_ty = 0.0
    for half in range(len(_PANO_HALF_RANGES)):
        for row in range(0, TOKEN_H, 3):
            for col in range(0, TOKEN_W, 7):
                direction = rays_d[half, row, col].numpy().astype(np.float64)
                # Project a point 50 m along the ray (well past NEAR_CLIP).
                projected = project_ego_point_to_pano(direction * 50.0)
                assert projected is not None, (
                    f"half={half} row={row} col={col} fell outside the "
                    "vertical FOV, so the ray field disagrees with the "
                    "panorama's own projection")
                tx, ty = projected
                want_tx, want_ty = _expected_pano_pixel(half, row, col)
                # tx wraps at PANO_W; compare on the circle.
                dtx = abs((tx - want_tx + _PANO_SRC_W / 2) % _PANO_SRC_W
                          - _PANO_SRC_W / 2)
                worst_tx = max(worst_tx, dtx)
                worst_ty = max(worst_ty, abs(ty - want_ty))
    # Sub-pixel on a 9600-wide canvas: the inverse is exact up to float error.
    assert worst_tx < 1.0, f"worst |dtx| = {worst_tx}"
    assert worst_ty < 1.0, f"worst |dty| = {worst_ty}"


def test_pano_equirect_rays_sweep_azimuth_uniformly():
    """Azimuth must advance by a constant step -- that is what equirect means.

    Measured on the azimuth channel rather than the 3-D angle between the end
    directions: a half spans >180 deg, and `arccos` folds anything past 180
    back down (201.6 would read as 158.4), so the 3-D angle is the wrong
    instrument here.

    The pinhole path swept only 172.8 deg with a per-column step ranging from
    0.0509 to 12.6803 deg -- a 249x non-uniformity across one image.
    """
    _, rays_d = get_pano_equirect_rays(len(_PANO_HALF_RANGES),
                                       (TOKEN_H, TOKEN_W))
    mid_row = TOKEN_H // 2
    for half, (half_start, half_end) in enumerate(_PANO_HALF_RANGES):
        d = rays_d[half, mid_row].numpy()
        # Project's convention: ang_x = arctan2(px, py) = arctan2(-y, x).
        azimuth = np.unwrap(np.arctan2(-d[:, 1], d[:, 0]))
        step = np.degrees(np.diff(azimuth))
        expected_step = (half_end - half_start) / _PANO_SRC_W * 360.0 / TOKEN_W
        assert step.min() == pytest.approx(expected_step, abs=1e-3)
        assert step.max() == pytest.approx(expected_step, abs=1e-3)
        sweep = math.degrees(azimuth[-1] - azimuth[0])
        expected_sweep = expected_step * (TOKEN_W - 1)
        assert sweep == pytest.approx(expected_sweep, abs=1e-2), (
            f"half={half} azimuth sweep {sweep:.2f} deg, "
            f"expected {expected_sweep:.2f}")

    # Vertical span is under 180 deg, so the 3-D angle is safe there.
    for half in range(len(_PANO_HALF_RANGES)):
        d = rays_d[half].numpy()
        v_span = _angle_deg(d[0, TOKEN_W // 2], d[-1, TOKEN_W // 2])
        expected_v = TRUE_V_SPAN_DEG * (1.0 - 1.0 / TOKEN_H)
        assert v_span == pytest.approx(expected_v, abs=0.1), (
            f"half={half} vertical span {v_span:.2f} deg, "
            f"expected ~{expected_v:.2f}")


def test_pano_equirect_rays_look_at_the_road_not_the_sky():
    """The pinhole path produced a mean elevation of +54 deg (all sky)."""
    _, rays_d = get_pano_equirect_rays(len(_PANO_HALF_RANGES),
                                       (TOKEN_H, TOKEN_W))
    elevation = torch.rad2deg(torch.asin(rays_d[..., 2].clamp(-1.0, 1.0)))
    assert abs(float(elevation.mean())) < 1.0
    assert float(elevation.max()) < TRUE_V_SPAN_DEG / 2 + 1.0
    assert float(elevation.min()) > -TRUE_V_SPAN_DEG / 2 - 1.0


def test_pano_equirect_rays_agree_across_both_seams():
    """Same physical azimuth must get the same direction from either half.

    The pinhole path disagreed by 16.1 deg at the L/R seam and 12.7 deg at
    the wrap seam -- exactly the CAM_FRONT / CAM_BACK regions where the
    teacher-student gap is largest.
    """
    _, rays_d = get_pano_equirect_rays(len(_PANO_HALF_RANGES),
                                       (TOKEN_H, TOKEN_W))
    mid_row = TOKEN_H // 2
    d = rays_d.numpy()

    def column_for(half, target_tx):
        half_start, half_end = _PANO_HALF_RANGES[half]
        cols = np.arange(TOKEN_W)
        txs = half_start + (cols + 0.5) / TOKEN_W * float(half_end - half_start)
        return int(np.argmin(np.abs(txs - target_tx)))

    # L/R seam (tx = 4800, inside CAM_FRONT) lies in both halves' overlap.
    left = d[0, mid_row, column_for(0, 4800.0)]
    right = d[1, mid_row, column_for(1, 4800.0)]
    # Residual is token quantisation, not error: one token cell spans
    # 5376/144 = 37.3 source px = 1.4 deg of azimuth, and the nearest column
    # from each half can be a full cell apart. Measured 0.199 deg here.
    assert _angle_deg(left, right) < 1.5

    # Wrap seam (tx = 0 == 9600, inside CAM_BACK).
    left = d[0, mid_row, column_for(0, 0.0)]
    right = d[1, mid_row, column_for(1, _PANO_SRC_W)]
    assert _angle_deg(left, right) < 1.5   # measured 1.400 = exactly one cell


# ---------------------------------------------------------------------------
# Plucker moments with per-column source-camera origins.
# ---------------------------------------------------------------------------
from dwm_pano.datasets.pano_projection import CAM_ORDER              # noqa: E402
from dwm_pano.models.crossview_temporal_dit_pano import (             # noqa: E402
    RayEncoder, get_pano_column_camera_index, get_pano_plucker_moments)


def test_column_camera_index_matches_pano_ownership():
    index = get_pano_column_camera_index(len(_PANO_HALF_RANGES), TOKEN_W)
    assert index.shape == (len(_PANO_HALF_RANGES), TOKEN_W)

    def owner(half, target_tx):
        half_start, half_end = _PANO_HALF_RANGES[half]
        cols = np.arange(TOKEN_W)
        txs = half_start + (cols + 0.5) / TOKEN_W * float(half_end - half_start)
        return CAM_ORDER[int(index[half, int(np.argmin(np.abs(txs - target_tx)))])]

    # tx = 4800 is dead ahead; tx = 0 / 9600 is the wrap, owned by CAM_BACK.
    assert owner(0, 4800.0) == "CAM_FRONT"
    assert owner(1, 4800.0) == "CAM_FRONT"
    assert owner(0, 0.0) == "CAM_BACK"
    assert owner(1, float(_PANO_SRC_W)) == "CAM_BACK"
    # Every half must draw on more than one physical camera -- that is the
    # whole reason a single per-view origin is wrong.
    for half in range(len(_PANO_HALF_RANGES)):
        assert len(set(index[half].tolist())) >= 3


def test_plucker_moments_are_nonzero_and_orthogonal_to_direction():
    """`m = o x d` must be non-trivial and satisfy the Plucker invariant.

    With the virtual pano extrinsics (camera at the ego origin) the moment is
    identically zero -- measured max |m| = 0.000e+00 -- which is exactly why
    per-column real origins are required for Plucker to carry anything.
    """
    view_count = len(_PANO_HALF_RANGES)
    _, rays_d = get_pano_equirect_rays(view_count, (TOKEN_H, TOKEN_W))
    batch_size, sequence_length = 2, 3
    rays_d = rays_d.repeat(batch_size * sequence_length, 1, 1, 1)
    index = get_pano_column_camera_index(view_count, TOKEN_W)

    # Distinct, realistic mounting offsets (metres) for the six cameras.
    translations = torch.tensor(
        [[0.0, 0.0, 1.5], [-0.5, 0.9, 1.5], [1.4, 0.6, 1.5],
         [1.7, 0.0, 1.5], [1.4, -0.6, 1.5], [-0.5, -0.9, 1.5]])
    translations = translations.expand(
        batch_size, sequence_length, 6, 3).contiguous()

    moments = get_pano_plucker_moments(rays_d, translations, index)
    assert moments.shape == rays_d.shape
    assert float(moments.abs().max()) > 0.1, "moments collapsed to zero"
    # Plucker invariant: the moment is orthogonal to the direction.
    dot = (moments * rays_d).sum(-1).abs().max()
    assert float(dot) < 1e-4, f"m . d = {float(dot)}"

    # An origin at the ego centre must reproduce the degenerate case.
    zero_origin = torch.zeros(batch_size, sequence_length, 6, 3)
    assert float(get_pano_plucker_moments(
        rays_d, zero_origin, index).abs().max()) == 0.0


def test_plucker_moments_change_where_the_source_camera_switches():
    """The information Plucker adds is localised at the seams."""
    view_count = len(_PANO_HALF_RANGES)
    _, rays_d = get_pano_equirect_rays(view_count, (TOKEN_H, TOKEN_W))
    index = get_pano_column_camera_index(view_count, TOKEN_W)
    translations = torch.tensor(
        [[0.0, 0.0, 1.5], [-0.5, 0.9, 1.5], [1.4, 0.6, 1.5],
         [1.7, 0.0, 1.5], [1.4, -0.6, 1.5], [-0.5, -0.9, 1.5]])
    translations = translations.expand(1, 1, 6, 3).contiguous()
    moments = get_pano_plucker_moments(rays_d, translations, index)

    for half in range(view_count):
        switches = (index[half, 1:] != index[half, :-1]).nonzero().flatten()
        assert switches.numel() >= 2
        jump = (moments[half, TOKEN_H // 2, 1:]
                - moments[half, TOKEN_H // 2, :-1]).norm(dim=-1)
        at_switch = jump[switches].mean()
        smooth = jump[[i for i in range(TOKEN_W - 1)
                       if i not in switches.tolist()]].mean()
        assert at_switch > 3.0 * smooth, (
            f"half={half} switch jump {at_switch:.4f} vs smooth "
            f"{smooth:.4f}: moments do not distinguish source cameras")


def test_ray_encoder_channel_count_is_unchanged_by_per_pixel_pos():
    """Per-pixel `pos` must keep `proj` checkpoint-compatible."""
    encoder = RayEncoder(cond_proj_dim=72, in_channels=64)
    rays = torch.randn(2, TOKEN_H, TOKEN_W, 3)
    rays = rays / rays.norm(dim=-1, keepdim=True)
    broadcast = encoder(torch.randn(2, 3), rays)
    per_pixel = encoder(torch.randn(2, TOKEN_H, TOKEN_W, 3), rays)
    assert broadcast.shape == per_pixel.shape == (2, TOKEN_H, TOKEN_W, 64)
    assert encoder.proj.in_features == 72
