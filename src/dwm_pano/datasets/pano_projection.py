"""
Cylindrical panorama projection for 2-pano driving world model.

This module provides forward projection (3D ego/world → pano pixel) that is the
mathematical inverse of the backward mapping used by OneBEV's `im2sphere`
when generating the RGB pano. Produced pano pixels are guaranteed to align
with the stitched RGB pano, so 3D bbox / HD map renders overlay correctly.

See original stitch code: ~/workspace/OneBEV/tools/stitch/nuscenes_stitch.py:91-155
See pipeline wrapper:     omninwm/utils/pano_hardseam.py

Coordinate systems:
  * nuScenes ego:  +X forward, +Y left,  +Z up
  * Pano camera:   +x right,   +y forward, +z up   (OneBEV convention)

Axis remap ego → pano:
  px_pano = -y_ego   # right = -left
  py_pano =  x_ego   # forward = forward
  pz_pano =  z_ego   # up = up

Pano geometry constants (must match the data generation spec):
  * Pano width:         9600  (pixels)
  * Pano pre-crop H:    900   (before row-crop)
  * Crop top:           150   (pano[150:750, :] in combine_views_hardseam)
  * Final H:            600
  * Vertical FOV:       ±25°  at pre-crop (= ±16.67° effective after crop)
  * Horizontal FOV:     360°  (full wrap), centered at TX=4800 (CAM_FRONT at refview=3)
"""
from __future__ import annotations

import numpy as np
from PIL import ImageDraw


# -----------------------------------------------------------------------------
# Pano geometry — MUST match data generation
# (from nuscenes_stitch.py:91-155 and pano_hardseam.py:63-78)
# -----------------------------------------------------------------------------
PANO_W = 9600
PANO_H_PRE_CROP = 900
PANO_CROP_TOP = 150
PANO_H = PANO_H_PRE_CROP - 2 * PANO_CROP_TOP  # 600
VERT_FOV_HALF_RAD = np.pi * 50.0 / 180.0 / 2.0  # 25° (pre-crop half-FOV)

# Voronoi boundaries from pano_hardseam.py:22 (pixel index in 9600 pano).
# Each cam owns [boundaries[i-1], boundaries[i]); CAM_BACK wraps around (both ends).
VORONOI_BOUNDARIES = [1229, 2625, 4069, 5557, 7038, 8421]

# Our 2-pano half slicing (from project_overview.md §2.2):
# Left half covers [0, 5376)  — starts at CAM_BACK wrap, ends mid CAM_FRONT
# Right half covers [4224, 9600) — starts mid CAM_FRONT, ends at CAM_BACK wrap
# Overlap [4224, 5376) is 1152 px around CAM_FRONT.
PANO_HALF_LEFT_RANGE = (0, 5376)
PANO_HALF_RIGHT_RANGE = (4224, 9600)

# Clipping thresholds
NEAR_CLIP_RADIUS = 0.5  # meters — points closer than this produce unreliable angles

# Cam order (matches nuscenes_stitch.py and pano_hardseam.py)
# Each cam owns a Voronoi slice of the pano. Horizontal boundaries in radians
# derived from VORONOI_BOUNDARIES (0-centered = CAM_FRONT, per im2sphere convention):
#   TX - PANO_W/2 → angle in [-π, π)
_VORONOI_ANG = [(b - PANO_W / 2.0) / PANO_W * 2.0 * np.pi for b in VORONOI_BOUNDARIES]
# _VORONOI_ANG = [-2.34, -1.42, -0.48, +0.50, +1.47, +2.37] rad (≈ -134°, -81°, -27°, +29°, +84°, +136°)
CAM_ORDER = ["CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
             "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"]


def _cam_idx_for_angle(ang_x_rad):
    """Given horizontal angle from ego in [-π, π], return cam index that owns it
    per Voronoi boundaries. CAM_BACK wraps around both ends."""
    # CAM_BACK: [_VORONOI_ANG[5], π] ∪ [-π, _VORONOI_ANG[0]]
    if ang_x_rad < _VORONOI_ANG[0] or ang_x_rad >= _VORONOI_ANG[5]:
        return 0  # CAM_BACK
    if ang_x_rad < _VORONOI_ANG[1]:
        return 1  # CAM_BACK_LEFT
    if ang_x_rad < _VORONOI_ANG[2]:
        return 2  # CAM_FRONT_LEFT
    if ang_x_rad < _VORONOI_ANG[3]:
        return 3  # CAM_FRONT
    if ang_x_rad < _VORONOI_ANG[4]:
        return 4  # CAM_FRONT_RIGHT
    return 5  # CAM_BACK_RIGHT


def project_ego_point_to_pano_with_cams(xyz_ego, cam_translations):
    """Forward projection accounting for per-camera translation offset (parallax).

    For each 3D ego-frame point, determine which Voronoi cam would see it based on
    the rough angle from ego, then recompute ray from THAT cam's real translation.
    This matches the hardseam pano's per-region perspective.

    Args:
        xyz_ego: (3,) sequence.
        cam_translations: dict mapping CAM name → (3,) translation in ego frame.
            Typically from info["cams"][name]["sensor2ego_translation"].

    Returns:
        (TX, TY) in final 9600×600 pano, or None.
    """
    x_ego, y_ego, z_ego = float(xyz_ego[0]), float(xyz_ego[1]), float(xyz_ego[2])

    # Step 1: rough angle from ego origin (for Voronoi owner)
    px0 = -y_ego
    py0 =  x_ego
    ang_x_from_ego = np.arctan2(px0, py0)  # [-π, π)
    cam_i = _cam_idx_for_angle(ang_x_from_ego)
    cam_name = CAM_ORDER[cam_i]
    if cam_name not in cam_translations:
        # fallback to ego origin
        return project_ego_point_to_pano(xyz_ego)
    cam_t = cam_translations[cam_name]

    # Step 2: ray from cam i's position → redo projection
    dx = x_ego - cam_t[0]
    dy = y_ego - cam_t[1]
    dz = z_ego - cam_t[2]
    px = -dy
    py = dx
    pz = dz

    r = np.sqrt(px * px + py * py + pz * pz)
    if r < NEAR_CLIP_RADIUS:
        return None

    ang_x = np.arctan2(px, py)
    ang_y = np.arcsin(np.clip(pz / r, -1.0, 1.0))

    tx = ang_x / (2.0 * np.pi) * PANO_W + PANO_W / 2.0 + 0.5
    ty_pre = -ang_y / VERT_FOV_HALF_RAD * (PANO_H_PRE_CROP / 2.0) + PANO_H_PRE_CROP / 2.0 + 0.5
    ty = ty_pre - PANO_CROP_TOP
    tx = tx % PANO_W

    if ty < 0.0 or ty >= PANO_H:
        return None
    return tx, ty


def draw_line_on_pano_with_cams(draw, p_a_ego, p_b_ego, cam_translations,
                                pen_color, pen_width=4):
    """Draw a 3D line with per-camera parallax correction.
    Each endpoint uses the cam's translation as ray origin (based on Voronoi)."""
    a = project_ego_point_to_pano_with_cams(p_a_ego, cam_translations)
    b = project_ego_point_to_pano_with_cams(p_b_ego, cam_translations)
    if a is None or b is None:
        return
    tx_a, ty_a = a
    tx_b, ty_b = b
    if not _segment_crosses_wrap(tx_a, tx_b):
        draw.line([(tx_a, ty_a), (tx_b, ty_b)], fill=pen_color, width=pen_width)
        return
    # wrap
    if tx_a < tx_b:
        t = (PANO_W - tx_b) / ((tx_a + PANO_W) - tx_b)
        ty_seam = ty_b + t * (ty_a - ty_b)
        draw.line([(tx_b, ty_b), (PANO_W - 0.5, ty_seam)], fill=pen_color, width=pen_width)
        draw.line([(0.5, ty_seam), (tx_a, ty_a)], fill=pen_color, width=pen_width)
    else:
        t = (PANO_W - tx_a) / ((tx_b + PANO_W) - tx_a)
        ty_seam = ty_a + t * (ty_b - ty_a)
        draw.line([(tx_a, ty_a), (PANO_W - 0.5, ty_seam)], fill=pen_color, width=pen_width)
        draw.line([(0.5, ty_seam), (tx_b, ty_b)], fill=pen_color, width=pen_width)

# -----------------------------------------------------------------------------
# Point projection
# -----------------------------------------------------------------------------
def project_ego_point_to_pano(xyz_ego):
    """Project a single 3D point in ego frame to full 9600×600 pano pixel.

    Args:
        xyz_ego: (3,) or (4,) numpy array/sequence. Either cartesian (x, y, z)
            or homogeneous (x, y, z, 1). Must be in nuScenes ego frame:
            +X forward, +Y left, +Z up.

    Returns:
        (TX, TY) floats in the FINAL 9600×600 pano, or None if point is
        too close or out of vertical FOV. TX is always in [0, 9600);
        TY in [0, 600) for valid points (no clipping at wrap — horizontal
        always projects).
    """
    x_ego, y_ego, z_ego = float(xyz_ego[0]), float(xyz_ego[1]), float(xyz_ego[2])

    # Axis remap ego → pano
    px = -y_ego
    py = x_ego
    pz = z_ego

    r = np.sqrt(px * px + py * py + pz * pz)
    if r < NEAR_CLIP_RADIUS:
        return None

    # Direction → angles (inverse of im2sphere's alpha/beta/gamma)
    ang_x = np.arctan2(px, py)           # [-π, π],  0 = CAM_FRONT direction
    ang_y = np.arcsin(np.clip(pz / r, -1.0, 1.0))  # elevation

    # Angles → pre-crop pixel (inverse of im2sphere:96-97)
    tx = ang_x / (2.0 * np.pi) * PANO_W + PANO_W / 2.0 + 0.5
    ty_pre = -ang_y / VERT_FOV_HALF_RAD * (PANO_H_PRE_CROP / 2.0) + PANO_H_PRE_CROP / 2.0 + 0.5

    # Apply [150:750] crop
    ty = ty_pre - PANO_CROP_TOP

    # Wrap TX into [0, 9600) (np.arctan2 may give boundary cases)
    tx = tx % PANO_W

    # Vertical FOV clip (final 600 H corresponds to ±16.67° elevation)
    if ty < 0.0 or ty >= PANO_H:
        return None

    return tx, ty


def project_ego_points_to_pano(xyz_ego_array):
    """Vectorized version. Input: (N, 3) or (N, 4) numpy array in ego frame.
    Returns: (TX, TY, valid_mask) where valid_mask is a (N,) bool array
    indicating successfully projected points (within vertical FOV + not too close).
    TX, TY arrays are (N,) float; invalid entries hold NaN.
    """
    arr = np.asarray(xyz_ego_array, dtype=np.float64)
    if arr.shape[-1] == 4:  # homogeneous
        arr = arr[..., :3]
    x_ego = arr[..., 0]
    y_ego = arr[..., 1]
    z_ego = arr[..., 2]

    px = -y_ego
    py = x_ego
    pz = z_ego

    r = np.sqrt(px * px + py * py + pz * pz)
    ang_x = np.arctan2(px, py)
    ang_y = np.arcsin(np.clip(pz / np.maximum(r, 1e-9), -1.0, 1.0))

    tx = ang_x / (2.0 * np.pi) * PANO_W + PANO_W / 2.0 + 0.5
    ty_pre = -ang_y / VERT_FOV_HALF_RAD * (PANO_H_PRE_CROP / 2.0) + PANO_H_PRE_CROP / 2.0 + 0.5
    ty = ty_pre - PANO_CROP_TOP
    tx = tx % PANO_W

    valid = (r >= NEAR_CLIP_RADIUS) & (ty >= 0.0) & (ty < PANO_H)
    tx_out = np.where(valid, tx, np.nan)
    ty_out = np.where(valid, ty, np.nan)
    return tx_out, ty_out, valid


# -----------------------------------------------------------------------------
# Line segment drawing — handles wrap-around at TX=0/9600
# -----------------------------------------------------------------------------
def _segment_crosses_wrap(tx_a, tx_b):
    """Return True if line TX_a → TX_b crosses the seam at TX=0/9600
    (i.e., going the long way around would be shorter than wrapping).
    Heuristic: if |TX_a - TX_b| > W/2, they're on opposite sides of the wrap."""
    return abs(tx_a - tx_b) > PANO_W / 2.0


def draw_line_on_pano(draw: ImageDraw.ImageDraw, p_a_ego, p_b_ego,
                      pen_color, pen_width=4):
    """Draw a 3D line from p_a_ego to p_b_ego onto the full 9600×600 pano.
    Handles wrap-around by splitting the line at TX=0/9600.

    Args:
        draw: PIL ImageDraw on a full 9600×600 image.
        p_a_ego, p_b_ego: 3D ego-frame endpoints (tuples/arrays, length 3 or 4).
        pen_color: tuple color for PIL.
        pen_width: stroke width in pixels.
    """
    a = project_ego_point_to_pano(p_a_ego)
    b = project_ego_point_to_pano(p_b_ego)
    if a is None or b is None:
        return  # at least one endpoint out of FOV — skip conservatively

    tx_a, ty_a = a
    tx_b, ty_b = b

    if not _segment_crosses_wrap(tx_a, tx_b):
        draw.line([(tx_a, ty_a), (tx_b, ty_b)], fill=pen_color, width=pen_width)
        return

    # Wrap: split into two segments via virtual TX shift
    # Example: TX_a=100, TX_b=9500 → short way is through seam. Shift the one
    # closer to TX=0 by +W, draw, then shift the other by -W, draw.
    if tx_a < tx_b:
        # tx_a is near seam-left (small TX); extend it to tx_a + W
        t = (PANO_W - tx_b) / ((tx_a + PANO_W) - tx_b)
        ty_seam_right = ty_b + t * (ty_a - ty_b)
        draw.line([(tx_b, ty_b), (PANO_W - 0.5, ty_seam_right)],
                  fill=pen_color, width=pen_width)
        draw.line([(0.5, ty_seam_right), (tx_a, ty_a)],
                  fill=pen_color, width=pen_width)
    else:
        t = (PANO_W - tx_a) / ((tx_b + PANO_W) - tx_a)
        ty_seam_right = ty_a + t * (ty_b - ty_a)
        draw.line([(tx_a, ty_a), (PANO_W - 0.5, ty_seam_right)],
                  fill=pen_color, width=pen_width)
        draw.line([(0.5, ty_seam_right), (tx_b, ty_b)],
                  fill=pen_color, width=pen_width)


def draw_polyline_on_pano(draw: ImageDraw.ImageDraw, points_ego,
                          pen_color, pen_width=4, closed=False):
    """Draw a polyline (or closed polygon outline) on the full 9600×600 pano.
    Each consecutive pair of points is drawn via draw_line_on_pano.

    Args:
        points_ego: list of (3,) or (4,) ego-frame points.
        closed: if True, add edge from last back to first.
    """
    n = len(points_ego)
    if n < 2:
        return
    for i in range(n - 1):
        draw_line_on_pano(draw, points_ego[i], points_ego[i + 1],
                          pen_color, pen_width)
    if closed:
        draw_line_on_pano(draw, points_ego[-1], points_ego[0],
                          pen_color, pen_width)


# -----------------------------------------------------------------------------
# Half slicing
# -----------------------------------------------------------------------------
def slice_full_pano_to_halves(full_image):
    """Slice a full 9600×600 PIL Image (or numpy array) into 2 halves matching
    our RGB pipeline convention.

    Returns:
        (left_half, right_half): each ~5376 × 600.
    """
    if hasattr(full_image, "crop"):  # PIL
        left = full_image.crop((PANO_HALF_LEFT_RANGE[0], 0,
                                PANO_HALF_LEFT_RANGE[1], PANO_H))
        right = full_image.crop((PANO_HALF_RIGHT_RANGE[0], 0,
                                 PANO_HALF_RIGHT_RANGE[1], PANO_H))
        return left, right
    else:  # numpy (H, W, C) or (H, W)
        left = full_image[:, PANO_HALF_LEFT_RANGE[0]:PANO_HALF_LEFT_RANGE[1]]
        right = full_image[:, PANO_HALF_RIGHT_RANGE[0]:PANO_HALF_RIGHT_RANGE[1]]
        return left, right


# -----------------------------------------------------------------------------
# Self-check
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick sanity test — 3 points at known positions
    tests = [
        ("5m front, ground level",       (5.0, 0.0, 0.0),  (4800.5, 300.5)),
        ("5m back, ground level",        (-5.0, 0.0, 0.0), (0.5, 300.5)),   # or 9600.5 wraps to 0.5
        ("5m right, ground level",       (0.0, -5.0, 0.0), (7200.5, 300.5)),
        ("5m left, ground level",        (0.0, 5.0, 0.0),  (2400.5, 300.5)),
        ("5m front 1m up",               (5.0, 0.0, 1.0),  "ty < 300.5"),
        ("5m front 1m down",             (5.0, 0.0, -1.0), "ty > 300.5"),
        ("0.1m front (too close)",       (0.1, 0.0, 0.0),  None),
        ("5m front, 10m up (out of FOV)", (5.0, 0.0, 10.0), None),
    ]
    print("=== project_ego_point_to_pano self-test ===")
    for name, xyz, expected in tests:
        r = project_ego_point_to_pano(xyz)
        print(f"  {name:35s}: ego={xyz} → {r}  (expected: {expected})")
