"""Band mask of a real nuScenes camera at any resolution (2026-10-22, tab:gs_recon).

The camera pixels whose content the panoramic band holds: the exact inverse of the OneBEV legacy warp that made the
recorded panoramas, i.e. PanDWM_eval149_20261011/cam_band_feats.py LegacyInverseGeometry (md5 993f0eef; validated
2026-10-20 against the recorded views: own-column MAE ~1/255, shift 0), with the output size a parameter instead of
the fixed 256 x 448.  For every output pixel q (native OpenCV index coordinates u = (j + 0.5) * 1600 / W - 0.5,
v likewise) Newton finds the pano point camera c's legacy warp sends to q, started at the ideal calibrated-sphere
solution of dwm_pano.nuscenes6_bridge; mask = converged, in front of the camera and inside the band (pre-crop pano
row centre in [149.5, 749.5)).  Camera order: nuscenes6_bridge.CAM_ORDER.  The code of _legacy_pixels and of the
solve is copied from cam_band_feats.py; band_geometry(..., (256, 448)) must equal it bit for bit (check_against).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from dwm_pano.datasets.pano_projection import pano_column_camera_indices
from dwm_pano.nuscenes6_bridge import (
    PANO_CANVAS_OVERLAP_COLUMNS, build_rig_sampling_grids, canvas_source_axes, ego_directions_to_pano_pixels,
    real_camera_pixel_directions_ego)

CANVAS_HW = (256, 4114)


def _legacy_pixels(T, Km, c, rows, cols, crop=None):
    """OneBEV legacy camera pixel (OpenCV index coords at 900x1600) of pano points (pre-crop row, 9600 column)."""
    from dwm_pano.calibrated_stitch import (
        ONEBEV_BACK_FOV_DEG, ONEBEV_OTHER_FOV_DEG, PANO_H_PRE_CROP, PANO_W, VERT_FOV_HALF_RAD, _onebev_angles)
    source_height, source_width = 900, 1600
    angles = _onebev_angles(T)
    angle_x, angle_y, roll_deg = angles[0, c, 0], angles[0, c, 1], angles[0, c, 2]
    tr = T[..., :3, 3]
    off = tr - tr.mean(dim=1, keepdim=True)
    du = Km[..., 0, 0] * (off[..., 0] / 5.5)
    dv = Km[..., 1, 1] * (off[..., 1] / 5.5)
    r_off = torch.sqrt(du.square() + dv.square())[0, c]
    fov = ONEBEV_BACK_FOV_DEG if c == 0 else ONEBEV_OTHER_FOV_DEG
    radius = source_width / 2.0 / math.tan(math.radians(fov) / 2.0) + r_off
    az = ((cols - PANO_W / 2.0 - 0.5) / PANO_W) * (2.0 * math.pi)
    el = -((rows - PANO_H_PRE_CROP / 2.0 - 0.5) / PANO_H_PRE_CROP) * (2.0 * VERT_FOV_HALF_RAD)
    x0 = radius * torch.cos(angle_y) * torch.sin(angle_x)
    y0 = radius * torch.cos(angle_y) * torch.cos(angle_x)
    z0 = radius * torch.sin(angle_y)
    alpha, beta, gamma = torch.cos(el) * torch.sin(az), torch.cos(el) * torch.cos(az), torch.sin(el)
    division = x0 * alpha + y0 * beta + z0 * gamma
    safe = torch.where(division.abs() > 1e-8, division, torch.ones_like(division))
    scale = radius.square() / safe
    vx, vy, vz = scale * alpha - x0, scale * beta - y0, scale * gamma - z0
    pxx, pxy = torch.cos(angle_x), -torch.sin(angle_x)
    delta_x = vx * pxx + vy * pxy
    pyx, pyy, pyz = -z0 * pxy, z0 * pxx, x0 * pxy - y0 * pxx
    norm = torch.sqrt(pyx.square() + pyy.square() + pyz.square()).clamp_min(1e-8)
    delta_y = (vx * pyx + vy * pyy + vz * pyz) / norm
    pixel_x = delta_x + (source_width + 1.0) / 2.0
    pixel_y = delta_y + (source_height + 1.0) / 2.0
    if crop is None:
        min_x = torch.maximum(torch.ones_like(pixel_x.amin()), torch.floor(pixel_x.amin()) - 1.0)
        min_y = torch.maximum(torch.ones_like(pixel_y.amin()), torch.floor(pixel_y.amin()) - 1.0)
        max_x = torch.minimum(torch.full_like(min_x, float(source_width)), torch.ceil(pixel_x.amax()) + 1.0)
        max_y = torch.minimum(torch.full_like(min_y, float(source_height)), torch.ceil(pixel_y.amax()) + 1.0)
        crop = (min_x, min_y, max_x, max_y)
    min_x, min_y, max_x, max_y = crop
    ccx, ccy = (max_x - min_x) / 2.0, (max_y - min_y) / 2.0
    dx, dy = pixel_x - min_x + 1.0 - ccx, pixel_y - min_y + 1.0 - ccy
    rr = roll_deg * (math.pi / 180.0)
    cos, sin = torch.cos(rr), torch.sin(rr)
    out_x = cos * dx - sin * dy + ccx + min_x
    out_y = sin * dx + cos * dy + ccy + min_y
    return out_x, out_y, division, crop


def band_geometry(c2e, K, size, out_hw):
    """c2e (6, 4, 4) camera-to-ego, K (6, 3, 3) at native 1600x900, size (6, 2) (w, h) -> dict of (6, H, W) arrays:
    valid (bool band mask), grids (canvas grid_sample coords), owner (canvas-column camera)."""
    from dwm_pano.calibrated_stitch import _onebev_legacy_grid_for_camera, prepare_gsplat_intrinsics
    from dwm_pano.datasets.pano_projection import PANO_CROP_TOP
    if any(tuple(int(v) for v in s) != (1600, 900) for s in size):
        raise RuntimeError("the legacy warp is defined at native 1600x900")
    h, w = int(out_hw[0]), int(out_hw[1])
    rows_c, cols_c = CANVAS_HW
    axes = canvas_source_axes(half_size=(rows_c, (cols_c + PANO_CANVAS_OVERLAP_COLUMNS) // 2))
    ideal_grids, ideal_valid, _ = build_rig_sampling_grids(c2e, K, size, CANVAS_HW, (h, w), band_limit=False,
                                                           source_axes=axes)
    T = torch.as_tensor(c2e, dtype=torch.float64)[None]
    Km = torch.stack([prepare_gsplat_intrinsics(torch.as_tensor(K[i], dtype=torch.float64), (900, 1600),
                                                (900, 1600)) for i in range(6)])[None]
    rows_pre = torch.as_tensor(axes[0] + PANO_CROP_TOP, dtype=torch.float64)
    cols = torch.as_tensor(axes[1], dtype=torch.float64)
    R, C = torch.meshgrid(rows_pre, cols, indexing="ij")
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    qx = torch.as_tensor((jj + 0.5) * 1600.0 / w - 0.5)
    qy = torch.as_tensor((ii + 0.5) * 900.0 / h - 0.5)
    grids, valids, owners = [], [], []
    for c in range(6):
        ref, ref_valid = _onebev_legacy_grid_for_camera(T, Km, c, R, C, (900, 1600))
        px, py, div, crop = _legacy_pixels(T, Km, c, R.flatten(), C.flatten())
        mine = torch.stack([(px + 0.5) * (2.0 / 1600) - 1.0, (py + 0.5) * (2.0 / 900) - 1.0], -1)
        v = ref_valid.flatten()
        if not torch.allclose(mine[v], ref.reshape(-1, 2)[v], atol=1e-9, rtol=0):
            raise RuntimeError(f"legacy warp copy differs from calibrated_stitch for camera {c}")
        d = real_camera_pixel_directions_ego(c2e[c], K[c], h, w, row_stride=900.0 / h, column_stride=1600.0 / w)
        tx, ty, _ = ego_directions_to_pano_pixels(d)
        k = torch.as_tensor(tx - 0.5)
        r = torch.as_tensor(ty - 0.5 + PANO_CROP_TOP)
        eps = 0.25
        for _ in range(40):
            fx, fy, _, _ = _legacy_pixels(T, Km, c, r, k, crop)
            ex, ey = fx - qx, fy - qy
            fxr, fyr, _, _ = _legacy_pixels(T, Km, c, r + eps, k, crop)
            fxc, fyc, _, _ = _legacy_pixels(T, Km, c, r, k + eps, crop)
            a, b = (fxr - fx) / eps, (fxc - fx) / eps
            cc, dd = (fyr - fy) / eps, (fyc - fy) / eps
            det = a * dd - b * cc
            det = torch.where(det.abs() > 1e-12, det, torch.full_like(det, 1e-12))
            r = r - (dd * ex - b * ey) / det
            k = k - (-cc * ex + a * ey) / det
            r = r.clamp(-2000.0, 2900.0)
        fx, fy, div, _ = _legacy_pixels(T, Km, c, r, k, crop)
        resid = torch.sqrt((fx - qx) ** 2 + (fy - qy) ** 2)
        kw = torch.remainder(k, 9600.0)
        valid = (resid < 1e-3) & (div > 1e-8) & (r >= PANO_CROP_TOP - 0.5) & (r < PANO_CROP_TOP + 600 - 0.5)
        row_c = np.interp(r.numpy().ravel() - PANO_CROP_TOP, axes[0], np.arange(rows_c))
        col_c = np.interp(kw.numpy().ravel(), axes[1], np.arange(cols_c))
        gx = np.where(valid.numpy().ravel(), (col_c + 1.5) / (cols_c + 2.0) * 2.0 - 1.0, 0.0)
        gy = np.where(valid.numpy().ravel(), (row_c + 0.5) / rows_c * 2.0 - 1.0, 0.0)
        grids.append(np.stack([gx, gy], -1).reshape(h, w, 2))
        valids.append(valid.numpy())
        owners.append(pano_column_camera_indices(kw.numpy()))
    return {"valid": np.stack(valids), "grids": np.stack(grids), "owner": np.stack(owners),
            "ideal_valid": np.asarray(ideal_valid)}
