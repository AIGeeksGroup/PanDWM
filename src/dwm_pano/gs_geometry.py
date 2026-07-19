"""Shared pano-Gaussian geometry: rays, virtual cameras, resampled targets.

Functions here are the validated core of debug_pano_gs_geometry.py (Check A
round-trip 1.8e-12 px; Check C gsplat PSNR 30-38 dB) extracted for reuse by
the GS decoder training pipeline. All directions are unit vectors in the
nuScenes ego frame; ranges are Euclidean meters along those unit rays.
"""
import numpy as np
import torch

from dwm_pano.datasets.pano_projection import (
    PANO_CROP_TOP, PANO_H, PANO_H_PRE_CROP, PANO_W, VERT_FOV_HALF_RAD,
    VORONOI_BOUNDARIES, project_ego_points_to_pano,
)

# Elevation half-extent of the FINAL 600-row pano (pre-crop FOV scaled).
PANO_LAT_HALF_RAD = VERT_FOV_HALF_RAD * PANO_H / PANO_H_PRE_CROP


def pano_pixel_centers_to_ego_directions(u, v):
    """Inverse of `project_ego_point_to_pano` for 0-based pixel indices."""
    azimuth = (u.astype(np.float64) - PANO_W / 2.0) / PANO_W * 2.0 * np.pi
    row_pre_crop = v.astype(np.float64) + PANO_CROP_TOP
    row_offset = row_pre_crop - PANO_H_PRE_CROP / 2.0
    elevation = -row_offset / (PANO_H_PRE_CROP / 2.0) * VERT_FOV_HALF_RAD
    pano_x = np.sin(azimuth) * np.cos(elevation)
    pano_y = np.cos(azimuth) * np.cos(elevation)
    pano_z = np.sin(elevation)
    # Axis remap pano -> ego is the inverse of (px, py, pz) = (-y, x, z).
    return np.stack([pano_y, -pano_x, pano_z], axis=-1)


def pano_source_camera_cosine_map():
    """Per-pixel cos(ray, source-camera axis) over the full 600x9600 pano.

    The stitched pano depth stores each Voronoi source camera's z-depth;
    dividing by this map converts it to Euclidean range along the pano ray.
    Camera axes are the angular midpoints between adjacent Voronoi boundaries.
    """
    boundary_azimuth = [
        (boundary - PANO_W / 2.0) / PANO_W * 2.0 * np.pi
        for boundary in VORONOI_BOUNDARIES
    ]
    column_azimuth = (np.arange(PANO_W) + 0.5 - PANO_W / 2.0) \
        / PANO_W * 2.0 * np.pi
    cell_axis = np.empty(PANO_W)
    for cell_index in range(len(boundary_azimuth)):
        next_index = (cell_index + 1) % len(boundary_azimuth)
        start = boundary_azimuth[cell_index]
        stop = boundary_azimuth[next_index]
        span = (stop - start) % (2.0 * np.pi)
        center = start + span / 2.0
        inside = ((column_azimuth - start) % (2.0 * np.pi)) < span
        cell_axis[inside] = center
    row_pre_crop = np.arange(PANO_H, dtype=np.float64) + PANO_CROP_TOP
    row_offset = row_pre_crop + 0.5 - PANO_H_PRE_CROP / 2.0
    elevation = -row_offset / (PANO_H_PRE_CROP / 2.0) * VERT_FOV_HALF_RAD
    azimuth_cosine = np.cos(column_azimuth - cell_axis)
    return np.cos(elevation)[:, None] * azimuth_cosine[None, :]


def pano_half_directions(column_range, output_height, output_width):
    """Unit ego rays at pixel centers of a resized pano-half grid."""
    column_start, column_stop = column_range
    column_scale = (column_stop - column_start) / output_width
    row_scale = PANO_H / output_height
    column_axis = np.arange(output_width, dtype=np.float64)
    row_axis = np.arange(output_height, dtype=np.float64)
    u_axis = column_start + (column_axis + 0.5) * column_scale - 0.5
    v_axis = (row_axis + 0.5) * row_scale - 0.5
    u_grid, v_grid = np.meshgrid(u_axis, v_axis, indexing="xy")
    return pano_pixel_centers_to_ego_directions(u_grid, v_grid)


def build_virtual_camera(yaw_deg, height, width, fov_deg, lateral_shift_m):
    """OpenCV-convention c2w in ego frame plus pinhole K for one yaw."""
    yaw = np.deg2rad(yaw_deg)
    forward = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    right = np.array([-np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[1, 3] = lateral_shift_m
    focal = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    intrinsics = np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ])
    return c2w, intrinsics


def virtual_pixel_directions_ego(c2w, intrinsics, height, width):
    j_grid, i_grid = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64), indexing="ij")
    x_cam = (i_grid + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0]
    y_cam = (j_grid + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1]
    directions_cam = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=-1)
    return directions_cam @ c2w[:3, :3].T


def sample_pano_as_ground_truth(pano_float, directions_ego, yaw_deg,
                                sample_radius_m):
    """Equirect-resample the pano along given ego directions (center view)."""
    height, width = directions_ego.shape[:2]
    points = directions_ego.reshape(-1, 3) * sample_radius_m
    tx, ty, valid = project_ego_points_to_pano(points)
    tx = np.where(valid, tx, 0.0)
    ty = np.where(valid, ty, 0.0)
    # Roll the pano so this view's azimuth sits at the center, avoiding wrap.
    shift_px = int(round(yaw_deg / 360.0 * PANO_W))
    pano_rolled = np.roll(pano_float, -shift_px, axis=1)
    tx_rolled = (tx - shift_px) % PANO_W
    grid_x = tx_rolled / PANO_W * 2.0 - 1.0
    grid_y = ty / PANO_H * 2.0 - 1.0
    grid_array = np.stack([grid_x, grid_y], axis=-1)
    grid = torch.from_numpy(grid_array.reshape(1, height, width, 2))
    source = torch.from_numpy(
        pano_rolled.transpose(2, 0, 1)[None]).to(torch.float64)
    sampled = torch.nn.functional.grid_sample(
        source, grid, mode="bilinear", padding_mode="zeros",
        align_corners=False)
    ground_truth = sampled[0].numpy().transpose(1, 2, 0)
    valid_mask = valid.reshape(height, width)
    ground_truth[~valid_mask] = 0.0
    return ground_truth, valid_mask
