"""Cylindrical panorama projection for the 2-pano driving world model.

Forward projection (3D ego/world to pano pixel) that is the inverse of the backward mapping used by OneBEV's
`im2sphere` when generating the RGB pano, so projected points align with the stitched RGB pano and 3D bbox / HD map
renders overlay correctly.

Coordinate systems:
  * nuScenes ego: +X forward, +Y left, +Z up
  * Pano camera:  +x right, +y forward, +z up (OneBEV convention)

Axis remap ego to pano: px_pano = -y_ego, py_pano = x_ego, pz_pano = z_ego.

Pano geometry constants (must match the data generation):
  * Pano width 9600 px; pre-crop height 900; crop top 150 (pano[150:750, :]); final height 600
  * Vertical FOV +-25 degrees at pre-crop (+-16.67 degrees effective after the crop)
  * Horizontal FOV 360 degrees (full wrap), centered at TX=4800 (CAM_FRONT)
"""
from __future__ import annotations

import numpy as np
from PIL import ImageDraw


# Pano geometry (must match the data generation).
PANO_W = 9600
PANO_H_PRE_CROP = 900
PANO_CROP_TOP = 150
PANO_H = PANO_H_PRE_CROP - 2 * PANO_CROP_TOP  # 600
VERT_FOV_HALF_RAD = np.pi * 50.0 / 180.0 / 2.0  # 25 degrees (pre-crop half-FOV)

# Voronoi boundaries of the hard-seam stitch (pixel index in the 9600 pano).
# Each cam owns [boundaries[i-1], boundaries[i]); CAM_BACK wraps around both ends.
VORONOI_BOUNDARIES = [1229, 2625, 4069, 5557, 7038, 8421]

# 2-pano half slicing:
# Left half covers [0, 5376): starts at the CAM_BACK wrap, ends mid CAM_FRONT.
# Right half covers [4224, 9600): starts mid CAM_FRONT, ends at the CAM_BACK wrap.
# The overlap [4224, 5376) is 1152 px around CAM_FRONT.
PANO_HALF_LEFT_RANGE = (0, 5376)
PANO_HALF_RIGHT_RANGE = (4224, 9600)

# Clipping thresholds
NEAR_CLIP_RADIUS = 0.5  # meters; closer points produce unreliable angles

# Camera order of the stitch. Each cam owns a Voronoi slice of the pano; the horizontal boundaries in radians are
# derived from VORONOI_BOUNDARIES (0 = CAM_FRONT, im2sphere convention): angle = (TX - PANO_W / 2) / PANO_W * 2 pi.
_VORONOI_ANG = [(b - PANO_W / 2.0) / PANO_W * 2.0 * np.pi for b in VORONOI_BOUNDARIES]
CAM_ORDER = ["CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
             "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"]


def pano_column_camera_indices(column_coordinates):
    """Return hard-seam source-camera indices for full-pano columns.

    ``column_coordinates`` may be a scalar or an arbitrary numpy-shaped array.
    Returned indices follow :data:`CAM_ORDER`.  Both wrapped tails of the
    panorama correctly map to ``CAM_BACK``.
    """
    columns = np.asarray(column_coordinates, dtype=np.float64) % PANO_W
    cells = np.searchsorted(
        np.asarray(VORONOI_BOUNDARIES, dtype=np.float64),
        columns,
        side="right",
    )
    return np.where(cells == len(VORONOI_BOUNDARIES), 0, cells).astype(
        np.int64)


def _cam_idx_for_angle(ang_x_rad):
    """Return the index of the camera that owns a horizontal angle from ego in [-pi, pi] (Voronoi boundaries).
    CAM_BACK wraps around both ends."""
    # CAM_BACK covers [_VORONOI_ANG[5], pi] and [-pi, _VORONOI_ANG[0]].
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
    """Forward projection with the per-camera translation offset (parallax).

    The owning Voronoi camera is chosen from the rough angle from the ego origin, then the ray is recomputed from that
    camera's real translation, which matches the hard-seam pano's per-region perspective.

    Args:
        xyz_ego: (3,) sequence.
        cam_translations: dict mapping camera name to (3,) translation in the ego frame
            (info["cams"][name]["sensor2ego_translation"]).

    Returns:
        (TX, TY) in the final 9600x600 pano, or None.
    """
    x_ego, y_ego, z_ego = float(xyz_ego[0]), float(xyz_ego[1]), float(xyz_ego[2])

    # Rough angle from the ego origin picks the owning camera.
    px0 = -y_ego
    py0 =  x_ego
    ang_x_from_ego = np.arctan2(px0, py0)  # [-pi, pi)
    cam_i = _cam_idx_for_angle(ang_x_from_ego)
    cam_name = CAM_ORDER[cam_i]
    if cam_name not in cam_translations:
        # Fall back to the ego origin.
        return project_ego_point_to_pano(xyz_ego)
    cam_t = cam_translations[cam_name]

    # Recompute the ray from the owning camera's position.
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
    """Draw a 3D line with per-camera parallax correction (each endpoint uses its owning camera's translation as ray origin)."""
    a = project_ego_point_to_pano_with_cams(p_a_ego, cam_translations)
    b = project_ego_point_to_pano_with_cams(p_b_ego, cam_translations)
    if a is None or b is None:
        return
    tx_a, ty_a = a
    tx_b, ty_b = b
    if not _segment_crosses_wrap(tx_a, tx_b):
        draw.line([(tx_a, ty_a), (tx_b, ty_b)], fill=pen_color, width=pen_width)
        return
    # Segment crosses the wrap seam.
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

# Point projection
def project_ego_point_to_pano(xyz_ego):
    """Project a single 3D point in the ego frame to a pixel of the full 9600x600 pano.

    Args:
        xyz_ego: (3,) or (4,) array/sequence, cartesian (x, y, z) or homogeneous (x, y, z, 1), in the nuScenes ego
            frame (+X forward, +Y left, +Z up).

    Returns:
        (TX, TY) floats in the final pano, or None if the point is too close or out of the vertical FOV. TX is in
        [0, 9600) (horizontal always projects); TY is in [0, 600) for valid points.
    """
    x_ego, y_ego, z_ego = float(xyz_ego[0]), float(xyz_ego[1]), float(xyz_ego[2])

    # Axis remap ego to pano.
    px = -y_ego
    py = x_ego
    pz = z_ego

    r = np.sqrt(px * px + py * py + pz * pz)
    if r < NEAR_CLIP_RADIUS:
        return None

    # Direction to angles (inverse of im2sphere's alpha/beta/gamma).
    ang_x = np.arctan2(px, py)           # [-pi, pi], 0 = CAM_FRONT direction
    ang_y = np.arcsin(np.clip(pz / r, -1.0, 1.0))  # elevation

    # Angles to pre-crop pixel (inverse of im2sphere).
    tx = ang_x / (2.0 * np.pi) * PANO_W + PANO_W / 2.0 + 0.5
    ty_pre = -ang_y / VERT_FOV_HALF_RAD * (PANO_H_PRE_CROP / 2.0) + PANO_H_PRE_CROP / 2.0 + 0.5

    # Apply the [150:750] crop.
    ty = ty_pre - PANO_CROP_TOP

    # Wrap TX into [0, 9600).
    tx = tx % PANO_W

    # Vertical FOV clip (the final 600 px height corresponds to +-16.67 degrees elevation).
    if ty < 0.0 or ty >= PANO_H:
        return None

    return tx, ty


def project_ego_points_to_pano(xyz_ego_array):
    """Vectorized version. Input: (N, 3) or (N, 4) array in the ego frame.

    Returns (TX, TY, valid_mask): TX, TY are (N,) float arrays (NaN for invalid entries) and valid_mask is an (N,) bool
    array marking points that projected successfully (within the vertical FOV and not too close).
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


# Line segment drawing (handles wrap-around at TX=0/9600)
def _segment_crosses_wrap(tx_a, tx_b):
    """Return True if the line from TX_a to TX_b crosses the seam at TX=0/9600 (|TX_a - TX_b| > W / 2)."""
    return abs(tx_a - tx_b) > PANO_W / 2.0


def draw_line_on_pano(draw: ImageDraw.ImageDraw, p_a_ego, p_b_ego,
                      pen_color, pen_width=4):
    """Draw a 3D line from p_a_ego to p_b_ego onto the full 9600x600 pano, splitting it at the TX=0/9600 seam.

    Args:
        draw: PIL ImageDraw on a full 9600x600 image.
        p_a_ego, p_b_ego: 3D ego-frame endpoints (tuples/arrays, length 3 or 4).
        pen_color: tuple color for PIL.
        pen_width: stroke width in pixels.
    """
    a = project_ego_point_to_pano(p_a_ego)
    b = project_ego_point_to_pano(p_b_ego)
    if a is None or b is None:
        return  # an endpoint is out of the FOV: skip

    tx_a, ty_a = a
    tx_b, ty_b = b

    if not _segment_crosses_wrap(tx_a, tx_b):
        draw.line([(tx_a, ty_a), (tx_b, ty_b)], fill=pen_color, width=pen_width)
        return

    # Wrap: split into two segments at the seam (e.g. TX_a=100, TX_b=9500 is shorter through the seam).
    if tx_a < tx_b:
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
    """Draw a polyline (or closed polygon outline) on the full 9600x600 pano, one draw_line_on_pano call per segment.

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


# Half slicing
def slice_full_pano_to_halves(full_image):
    """Slice a full 9600x600 PIL Image (or numpy array) into the two halves of the RGB pipeline.

    Returns:
        (left_half, right_half): each 5376 px wide and 600 px high.
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


# Self-check
if __name__ == "__main__":
    # Quick sanity test with points at known positions.
    tests = [
        ("5m front, ground level",       (5.0, 0.0, 0.0),  (4800.5, 300.5)),
        ("5m back, ground level",        (-5.0, 0.0, 0.0), (0.5, 300.5)),   # 9600.5 wraps to 0.5
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
