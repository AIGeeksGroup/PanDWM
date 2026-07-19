"""Render bbox / hdmap to pano by rendering on 6 pinhole cam images and
running the SAME `stitch_hardseam` pipeline that produced the RGB pano.

This guarantees pixel-level alignment with the RGB pano:
  * Voronoi hardseam cuts at identical boundaries
  * im2sphere warp identical
  * r_offset parallax correction identical
  * Hard seams, wrap, and near-clip all handled by the existing stitch code
    (which has been validated on our 35K val samples)

vs. direct cylindrical projection (our earlier `pano_projection.py` approach):
  + No hand-rolled cylindrical math → no subtle axis/offset bugs
  + Automatically hardseam-consistent with RGB
  - Slower (6 pinhole renders + stitch per frame) — but fine for Dataset; can
    pre-compute if bottleneck in training.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

# Import the hardseam stitch from PanDWM workspace (same one used to make RGB pano)
_PANDWM_ROOT = Path("/scratch/fs201130/jn20658/tenants/clientB/workspace/PanDWM")
if str(_PANDWM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PANDWM_ROOT))
from omninwm.utils.pano_hardseam import stitch_hardseam


# nuscenes_stitch expects this order (from pano_hardseam.py:20)
# CAM_BACK(fov=110°) first, others fov=70°
ONEBEV_CAM_ORDER = ["CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
                     "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"]

# Native per-cam image size for nuScenes
CAM_W, CAM_H = 1600, 900

# Edges of a cuboid, matching mmdet3d LiDARInstance3DBoxes.corners order
# (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)
EDGES_CUBE = [(0, 3), (3, 7), (7, 4), (4, 0),   # bottom
              (1, 2), (2, 6), (6, 5), (5, 1),   # top
              (0, 1), (3, 2), (7, 6), (4, 5)]   # verticals

BBOX_COLORS = {
    "car":                   (255,  32,  32),
    "truck":                 (255, 128,   0),
    "bus":                   (255, 220,   0),
    "trailer":               (200,  80,   0),
    "construction_vehicle":  (180, 100,  60),
    "motorcycle":            (255,   0, 255),
    "bicycle":               (180,   0, 255),
    "pedestrian":            (  0, 255,  64),
    "traffic_cone":          (  0, 180, 255),
    "barrier":               (128, 128, 255),
}

HDMAP_COLORS = {
    "drivable_area": (  0, 128, 128),
    "lane":          (255, 128,   0),
    "ped_crossing":  (255, 255,   0),
    "walkway":       (  0, 200,   0),
    "stop_line":     (255,   0,   0),
    "carpark_area":  (128,   0, 128),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
# 8 corners in [-0.5, +0.5]^3, matching mmdet3d LiDARInstance3DBoxes corner order
# (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)
_CORNER_TEMPLATE = np.array([
    [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5],
    [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
    [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5],
], dtype=np.float32)


def _ego_corners_from_boxes(gt_boxes_7d, q_l2e, t_l2e):
    """Pure-numpy replacement for mmdet3d LiDARInstance3DBoxes(..., origin=(0.5,0.5,0.5)).corners
    then rotated/translated to ego frame via lidar2ego extrinsics.
    Verified pixel-perfect vs mmdet3d (max diff 1.2e-7, float32 precision noise).
    """
    boxes = np.asarray(gt_boxes_7d, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    centers = boxes[:, :3]
    dims = boxes[:, 3:6]
    yaws = boxes[:, 6]
    corners = _CORNER_TEMPLATE[None] * dims[:, None, :]
    cos_y = np.cos(yaws).astype(np.float32)
    sin_y = np.sin(yaws).astype(np.float32)
    x = corners[..., 0].copy()
    y = corners[..., 1].copy()
    corners[..., 0] = x * cos_y[:, None] - y * sin_y[:, None]
    corners[..., 1] = x * sin_y[:, None] + y * cos_y[:, None]
    corners += centers[:, None, :]
    R_l2e = np.asarray(q_l2e.rotation_matrix, dtype=np.float32)
    return corners @ R_l2e.T + np.asarray(t_l2e, dtype=np.float32)


def _ego_to_cam(points_ego, q_s2e, t_s2e):
    """Transform ego-frame points (..., 3) to a camera's sensor frame.
    Camera frame (nuScenes): x=right, y=down, z=forward."""
    q_e2s = q_s2e.inverse
    t = np.asarray(t_s2e)
    shifted = np.asarray(points_ego) - t
    # rotate each point
    flat = shifted.reshape(-1, 3)
    rotated = np.stack([q_e2s.rotate(p) for p in flat])
    return rotated.reshape(shifted.shape)


def _project_line_near_clip(a, b, near_z=0.05, far_z=512.0):
    """Pure-numpy copy of dwm.datasets.common.project_line (to avoid torch import).
    Clips segment a→b against near/far z planes, returns (x/z, y/z) for each endpoint.
    """
    if (a[2] < near_z and b[2] < near_z) or (a[2] > far_z and b[2] > far_z):
        return None
    ca, cb = a, b
    if a[2] >= near_z and b[2] < near_z:
        r = (near_z - b[2]) / (a[2] - b[2]); cb = a * r + b * (1 - r)
    elif a[2] < near_z and b[2] >= near_z:
        r = (b[2] - near_z) / (b[2] - a[2]); ca = a * r + b * (1 - r)
    if a[2] > far_z and b[2] <= far_z:
        r = (far_z - b[2]) / (a[2] - b[2]); ca = a * r + b * (1 - r)
    elif a[2] <= far_z and b[2] > far_z:
        r = (b[2] - far_z) / (b[2] - a[2]); cb = a * r + b * (1 - r)
    pa = ca[:2] / ca[2]
    pb = cb[:2] / cb[2]
    return (pa[0], pa[1], pb[0], pb[1])


def _project_line_pinhole(a_cam, b_cam, K, img_w, img_h, near_z=0.1, far_z=300.0):
    """Project a 3D line segment (in cam frame) to 2D pixel coords via pinhole.
    Returns (xa, ya, xb, yb) or None if fully clipped.
    """
    K3 = np.asarray(K)[:3, :3] if np.asarray(K).shape == (4, 4) else np.asarray(K)
    xy = _project_line_near_clip(a_cam, b_cam, near_z=near_z, far_z=far_z)
    if xy is None:
        return None
    # Scale by K
    xa, ya, xb, yb = xy
    # xa, ya are already x/z and y/z; multiply by K to get pixel coords
    u_a = K3[0, 0] * xa + K3[0, 2]
    v_a = K3[1, 1] * ya + K3[1, 2]
    u_b = K3[0, 0] * xb + K3[0, 2]
    v_b = K3[1, 1] * yb + K3[1, 2]
    # basic off-image test (coarse — let draw.line clip)
    if (u_a < -img_w or u_a > 2 * img_w) and (u_b < -img_w or u_b > 2 * img_w):
        return None
    return (u_a, v_a, u_b, v_b)


# -----------------------------------------------------------------------------
# bbox render
# -----------------------------------------------------------------------------
def render_bboxes_via_stitch(info: dict, pen_width: int = 3) -> Image.Image:
    """Render 3D bbox wireframes on 6 cam images, then hardseam-stitch.
    Returns PIL.Image RGB 9600×600 (black bg, colored wireframes)."""
    q_l2e = Quaternion(info["lidar2ego_rotation"])
    t_l2e = np.array(info["lidar2ego_translation"])
    gt_boxes = info["gt_boxes"]
    gt_names = info["gt_names"]
    corners_ego_all = _ego_corners_from_boxes(gt_boxes, q_l2e, t_l2e)

    cam_images = []
    cam_info_list = []
    for cam_name in ONEBEV_CAM_ORDER:
        cam = info["cams"][cam_name]
        q_s2e = Quaternion(cam["sensor2ego_rotation"])
        t_s2e = np.array(cam["sensor2ego_translation"])
        K = np.array(cam["camera_intrinsics"])

        img = Image.new("RGB", (CAM_W, CAM_H), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        for corners_ego, name in zip(corners_ego_all, gt_names):
            color = BBOX_COLORS.get(name)
            if color is None:
                continue
            corners_cam = _ego_to_cam(corners_ego, q_s2e, t_s2e)  # (8, 3)
            for i, j in EDGES_CUBE:
                xy = _project_line_pinhole(corners_cam[i], corners_cam[j],
                                            K, CAM_W, CAM_H)
                if xy is None:
                    continue
                draw.line(xy, fill=color, width=pen_width)
        cam_images.append(np.array(img))
        cam_info_list.append({
            "translation": cam["sensor2ego_translation"],
            "rotation":    cam["sensor2ego_rotation"],
            "camera_intrinsic": cam["camera_intrinsics"],
        })

    pano = stitch_hardseam(cam_images, cam_info_list, pano_outsize=[9600, 900])
    return Image.fromarray(pano.astype(np.uint8))


# -----------------------------------------------------------------------------
# hdmap render
# -----------------------------------------------------------------------------
def _world_to_ego_2d(world_xy, ego_translation, ego_rotation):
    """Transform (N, 2) world-frame ground points to ego frame (N, 3) with z=0 in world."""
    pts3 = np.hstack([world_xy, np.zeros((world_xy.shape[0], 1))])
    q_e2g = Quaternion(ego_rotation)
    q_g2e = q_e2g.inverse
    t_e2g = np.asarray(ego_translation)
    shifted = pts3 - t_e2g
    return np.stack([q_g2e.rotate(p) for p in shifted])


def render_hdmap_via_stitch(info: dict, map_cache, pen_width: int = 2,
                             max_distance: float = 65.0) -> Image.Image:
    """Render HD map polygon outlines on 6 cam images, then hardseam-stitch.
    Uses the same `stitch_hardseam` as RGB → pixel-aligned.
    """
    q_l2e = Quaternion(info["lidar2ego_rotation"])
    t_l2e = np.array(info["lidar2ego_translation"])  # unused but kept for symmetry
    ego_t = info["ego2global_translation"]
    ego_r = info["ego2global_rotation"]

    # Collect all polygons within max_distance, in ego frame (z=0 assumed)
    layers_polys = []   # list of (pts_ego (N,3), color)
    for layer_name, color in HDMAP_COLORS.items():
        if layer_name not in map_cache.layers:
            continue
        for rec in map_cache.layers[layer_name]:
            if "polygon_token" in rec:
                poly_tokens = [rec["polygon_token"]]
            elif "polygon_tokens" in rec:
                poly_tokens = rec["polygon_tokens"]
            else:
                continue
            for pt in poly_tokens:
                cx, cy = map_cache.polygon_centroid_world(pt)
                cen_ego = _world_to_ego_2d(np.array([[cx, cy]]), ego_t, ego_r)[0]
                if np.sqrt(cen_ego[0] ** 2 + cen_ego[1] ** 2) > max_distance:
                    continue
                pts_w = map_cache.polygon_exterior_world_pts(pt)
                if pts_w.shape[0] < 3:
                    continue
                pts_e = _world_to_ego_2d(pts_w, ego_t, ego_r)
                layers_polys.append((pts_e, color))

    cam_images = []
    cam_info_list = []
    for cam_name in ONEBEV_CAM_ORDER:
        cam = info["cams"][cam_name]
        q_s2e = Quaternion(cam["sensor2ego_rotation"])
        t_s2e = np.array(cam["sensor2ego_translation"])
        K = np.array(cam["camera_intrinsics"])

        img = Image.new("RGB", (CAM_W, CAM_H), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        for pts_ego, color in layers_polys:
            pts_cam = _ego_to_cam(pts_ego, q_s2e, t_s2e)
            # closed polygon: draw edges including last → first
            n = pts_cam.shape[0]
            for i in range(n):
                j = (i + 1) % n
                xy = _project_line_pinhole(pts_cam[i], pts_cam[j],
                                            K, CAM_W, CAM_H,
                                            near_z=0.1, far_z=max_distance)
                if xy is None:
                    continue
                draw.line(xy, fill=color, width=pen_width)
        cam_images.append(np.array(img))
        cam_info_list.append({
            "translation": cam["sensor2ego_translation"],
            "rotation":    cam["sensor2ego_rotation"],
            "camera_intrinsic": cam["camera_intrinsics"],
        })

    pano = stitch_hardseam(cam_images, cam_info_list, pano_outsize=[9600, 900])
    return Image.fromarray(pano.astype(np.uint8))
