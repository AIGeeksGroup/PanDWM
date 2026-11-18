"""Calibrated nuScenes camera geometry and differentiable hard-seam stitch.

The RGB panoramas used by :mod:`dwm_pano.datasets.nuscenes_pano` are not
central-camera equirectangular images.  They were made from the six nuScenes
cameras and a fixed, per-column hard seam.  Consequently a Gaussian scene
rendered from the six real cameras must be put through the same 2-D stitch
before it can be compared to the panorama RGB target.

This module deliberately keeps two projection models separate:

``"onebev_legacy"``
    Reproduces the geometric mapping in public OneBEV
    ``nuscenes_stitch.py``: relative Euler angles, a 110 degree CAM_BACK FOV /
    70 degree other-camera FOV, and its calibration-dependent ``r_offset``
    heuristic.  This is the model to use when supervising against the
    existing OneBEV-derived panoramas.  PyTorch's continuous bilinear sampler
    is not bit-identical to OpenCV's table-quantized two-pass remap, but the
    source coordinates and fixed owner cuts match.

``"calibrated"``
    Projects the panorama's ego-frame spherical ray through each camera's
    real sensor-to-ego rotation and intrinsic matrix.  This is physically
    cleaner and useful for geometry diagnostics or newly generated data, but
    it is intentionally *not* presented as pixel-identical to legacy RGB.

Both paths use the fixed Voronoi owner boundaries from
``datasets/pano_projection.py`` and ``torch.nn.functional.grid_sample``.  The
stitch is differentiable with respect to the six input images.  The hard
owner choice and optional seam erosion are fixed masks and do not need a
gradient.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from dwm_pano.datasets.pano_projection import (
    CAM_ORDER,
    PANO_CROP_TOP,
    PANO_H,
    PANO_H_PRE_CROP,
    PANO_W,
    VERT_FOV_HALF_RAD,
    VORONOI_BOUNDARIES,
)


NUSCENES_CAMERA_COUNT = 6
NUSCENES_IMAGE_SIZE = (900, 1600)  # (height, width)
ONEBEV_FRONT_CAMERA_INDEX = CAM_ORDER.index("CAM_FRONT")
ONEBEV_BACK_FOV_DEG = 110.0
ONEBEV_OTHER_FOV_DEG = 70.0
SUPPORTED_PROJECTION_MODELS = ("onebev_legacy", "calibrated")


def quaternion_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a nuScenes ``(w, x, y, z)`` quaternion to a 3x3 matrix.

    The implementation is local so dataset construction does not depend on
    SciPy and demo/training code can share exactly the same convention.
    Non-unit quaternions are normalized defensively.
    """
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError(
            "sensor2ego_rotation must have shape (4,), got {}".format(
                values.shape))
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("sensor2ego_rotation must be finite and non-zero")
    w, x, y, z = values / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z),
         2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),
         1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),
         2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float32)


def camera_records_to_arrays(
    cameras: Mapping[str, Mapping[str, object]],
    camera_order: Sequence[str] = CAM_ORDER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack one frame's nuScenes camera records in hard-seam order.

    Returns:
        ``(camera_to_ego, intrinsics, image_sizes)`` with shapes
        ``[6,4,4]``, ``[6,3,3]`` and ``[6,2]``.  Image sizes are ``(W,H)`` to
        match the dataset's existing ``image_size`` convention.  Public
        interpolated info pkls do not carry width/height, so 1600x900 is used
        unless a record explicitly supplies both fields.  The returned K is
        the raw nuScenes/OpenCV pixel-index matrix; call
        :func:`prepare_gsplat_intrinsics` before a gsplat render.
    """
    if len(camera_order) != NUSCENES_CAMERA_COUNT:
        raise ValueError(
            "hard-seam calibration requires exactly six cameras, got {}"
            .format(len(camera_order)))

    transforms = []
    intrinsics = []
    image_sizes = []
    for camera_name in camera_order:
        if camera_name not in cameras:
            raise KeyError(
                "camera calibration is missing {}; available cameras: {}"
                .format(camera_name, sorted(cameras)))
        record = cameras[camera_name]
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = quaternion_wxyz_to_matrix(
            record["sensor2ego_rotation"])
        translation = np.asarray(
            record["sensor2ego_translation"], dtype=np.float32)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError(
                "{}.sensor2ego_translation must be finite shape (3,), got {}"
                .format(camera_name, translation.shape))
        transform[:3, 3] = translation

        intrinsic = np.asarray(record["camera_intrinsics"], dtype=np.float32)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError(
                "{}.camera_intrinsics must be finite shape (3,3), got {}"
                .format(camera_name, intrinsic.shape))
        width = int(record.get("width", NUSCENES_IMAGE_SIZE[1]))
        height = int(record.get("height", NUSCENES_IMAGE_SIZE[0]))
        if width <= 0 or height <= 0:
            raise ValueError(
                "{} has invalid camera image size {}x{}".format(
                    camera_name, width, height))

        transforms.append(transform)
        intrinsics.append(intrinsic)
        image_sizes.append(np.asarray([width, height], dtype=np.int64))

    return (
        np.stack(transforms),
        np.stack(intrinsics),
        np.stack(image_sizes),
    )


def resize_intrinsics(
    intrinsics: torch.Tensor,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Resize K rows while preserving its existing coordinate convention."""
    source_height, source_width = [int(value) for value in source_size]
    target_height, target_width = [int(value) for value in target_size]
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("source_size and target_size entries must be positive")
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(
            "intrinsics must end in (3,3), got {}".format(intrinsics.shape))
    scale = intrinsics.new_tensor([
        target_width / float(source_width),
        target_height / float(source_height),
        1.0,
    ])
    # Left multiplication scales complete image-coordinate rows, including a
    # non-zero skew term, while keeping the homogeneous row unchanged.
    return intrinsics * scale[..., :, None]


def opencv_intrinsics_to_gsplat(intrinsics: torch.Tensor) -> torch.Tensor:
    """Convert OpenCV/nuScenes pixel-index K to gsplat edge-coordinate K.

    nuScenes projects with OpenCV coordinates, where integer ``(u,v)`` is the
    center of array element ``image[v,u]``.  gsplat evaluates raster pixels at
    ``(u+0.5,v+0.5)``.  Adding 0.5 to the principal point preserves the same
    physical camera ray under that change of coordinate origin.
    """
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(
            "intrinsics must end in (3,3), got {}".format(intrinsics.shape))
    result = intrinsics.clone()
    result[..., 0, 2] += 0.5
    result[..., 1, 2] += 0.5
    return result


def prepare_gsplat_intrinsics(
    opencv_intrinsics: torch.Tensor,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Convert native OpenCV K and resize it for a gsplat render.

    Conversion happens before resize, yielding
    ``cx_gs=(cx_cv+0.5)*sx`` and ``cy_gs=(cy_cv+0.5)*sy``.  Directly scaling
    raw nuScenes ``cx/cy`` misses ``0.5*s`` output pixels.
    """
    return resize_intrinsics(
        opencv_intrinsics_to_gsplat(opencv_intrinsics),
        source_size, target_size)


def scale_camera_intrinsics(
    intrinsics: torch.Tensor,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Backward-compatible name for :func:`prepare_gsplat_intrinsics`.

    Inputs are raw nuScenes/OpenCV intrinsics; outputs are gsplat intrinsics.
    New code should prefer the explicit function name.
    """
    return prepare_gsplat_intrinsics(intrinsics, source_size, target_size)


def hard_seam_owner_map(
    output_size: tuple[int, int],
    *,
    column_range: tuple[float, float] = (0.0, float(PANO_W)),
    seam_erosion_px: int = 0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return owner index and non-seam mask for a resized panorama region.

    Args:
        output_size: ``(height, width)`` of the requested output.
        column_range: half-open coordinates in the canonical 9600-wide pano.
            This directly supports the dataset's left ``(0,5376)`` and right
            ``(4224,9600)`` halves as well as a full panorama.
        seam_erosion_px: remove this many *output* pixels on both sides of
            each of the six internal hard seams.  The horizontal wrap is not
            removed because both tails are owned by CAM_BACK.

    Returns:
        ``owner[H,W]`` in :data:`CAM_ORDER`, and ``valid[H,W]`` which is false
        only inside an eroded seam band.
    """
    output_height, output_width = [int(value) for value in output_size]
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output_size entries must be positive")
    column_start, column_stop = [float(value) for value in column_range]
    if not (0.0 <= column_start < column_stop <= float(PANO_W)):
        raise ValueError(
            "column_range must satisfy 0 <= start < stop <= {}, got {}"
            .format(PANO_W, column_range))
    if int(seam_erosion_px) != seam_erosion_px or seam_erosion_px < 0:
        raise ValueError("seam_erosion_px must be a non-negative integer")

    columns = column_start + (
        torch.arange(output_width, dtype=torch.float64, device=device) + 0.5
    ) * ((column_stop - column_start) / output_width) - 0.5
    owners, valid_columns = hard_seam_owners_from_columns(
        columns, seam_erosion_px=seam_erosion_px)

    return (
        owners[None, :].expand(output_height, output_width),
        valid_columns[None, :].expand(output_height, output_width),
    )


def hard_seam_owners_from_columns(
    pano_columns: torch.Tensor,
    *,
    seam_erosion_px: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Owner/valid vectors for an explicit canonical-pano column axis.

    ``pano_columns`` may be non-uniform.  This is important for the training
    canvas, which concatenates two independently resized 5376-pixel halves
    after removing a rounded overlap and therefore is not a global 9600->4114
    resize.  Seam erosion is performed in output-index space and remains exact
    for such a non-uniform axis.
    """
    if pano_columns.ndim != 1 or pano_columns.numel() == 0:
        raise ValueError("pano_columns must be a non-empty 1-D tensor")
    if int(seam_erosion_px) != seam_erosion_px or seam_erosion_px < 0:
        raise ValueError("seam_erosion_px must be a non-negative integer")
    if not bool(torch.all(pano_columns[1:] > pano_columns[:-1])):
        raise ValueError("pano_columns must be strictly increasing")
    boundaries = torch.as_tensor(
        VORONOI_BOUNDARIES, dtype=pano_columns.dtype,
        device=pano_columns.device)
    owners = torch.bucketize(pano_columns, boundaries, right=True)
    owners = torch.where(
        owners == len(VORONOI_BOUNDARIES),
        torch.zeros_like(owners),
        owners,
    ).to(torch.long)
    valid = torch.ones_like(owners, dtype=torch.bool)
    if seam_erosion_px:
        changes = torch.nonzero(
            owners[1:] != owners[:-1], as_tuple=False).flatten()
        for change in changes.cpu().tolist():
            # change indexes the last pixel on the left.  Remove exactly E
            # output pixels on each side of the ownership transition.
            valid[max(0, change - seam_erosion_px + 1):change + 1] = False
            valid[
                change + 1:min(owners.numel(), change + 1 + seam_erosion_px)
            ] = False
    return owners, valid


def half_canvas_pano_axes(
    *,
    half_size: tuple[int, int],
    right_trim_px: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact canonical ``(v,u)`` axes of ``_stitch_canvas``.

    The dataset first resizes ``[0,5376)`` and ``[4224,9600)`` independently
    to ``half_size=(H,W)``.  The pipeline then keeps all of the left half and
    drops ``right_trim_px`` columns from the right half before concatenation.
    This helper preserves those two resize maps and their rounded overlap; a
    single global full-panorama resize does not.

    Returned row coordinates are in the 900-row *pre-crop* OneBEV canvas,
    while column coordinates are in the canonical 9600-wide canvas.
    """
    half_height, half_width = [int(value) for value in half_size]
    right_trim_px = int(right_trim_px)
    if half_height <= 0 or half_width <= 0:
        raise ValueError("half_size entries must be positive")
    if not 0 <= right_trim_px < half_width:
        raise ValueError(
            "right_trim_px must satisfy 0 <= trim < {}, got {}".format(
                half_width, right_trim_px))
    from dwm_pano.datasets.pano_projection import (
        PANO_HALF_LEFT_RANGE,
        PANO_HALF_RIGHT_RANGE,
    )

    row_axis = (
        (torch.arange(half_height, device=device, dtype=dtype) + 0.5)
        * (PANO_H / half_height) - 0.5 + PANO_CROP_TOP
    )

    def resized_columns(column_range):
        start, stop = column_range
        return start + (
            torch.arange(half_width, device=device, dtype=dtype) + 0.5
        ) * ((stop - start) / half_width) - 0.5

    left_columns = resized_columns(PANO_HALF_LEFT_RANGE)
    right_columns = resized_columns(PANO_HALF_RIGHT_RANGE)[right_trim_px:]
    columns = torch.cat([left_columns, right_columns])
    if not bool(torch.all(columns[1:] > columns[:-1])):
        raise ValueError(
            "right_trim_px={} leaves duplicated/reversed overlap in the "
            "stitched canvas".format(right_trim_px))
    return row_axis, columns


def _output_pano_coordinates(
    output_size: tuple[int, int],
    column_range: tuple[float, float],
    row_range: tuple[float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_height, output_width = output_size
    column_start, column_stop = column_range
    row_start, row_stop = row_range
    columns = column_start + (
        torch.arange(output_width, device=device, dtype=dtype) + 0.5
    ) * ((column_stop - column_start) / output_width) - 0.5
    rows = row_start + (
        torch.arange(output_height, device=device, dtype=dtype) + 0.5
    ) * ((row_stop - row_start) / output_height) - 0.5
    return torch.meshgrid(rows, columns, indexing="ij")


def _mesh_pano_axes(
    pano_rows: torch.Tensor,
    pano_columns: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pano_rows.ndim != 1 or pano_columns.ndim != 1:
        raise ValueError("pano_rows and pano_columns must both be 1-D")
    rows = pano_rows.to(device=device, dtype=dtype)
    columns = pano_columns.to(device=device, dtype=dtype)
    if rows.numel() == 0 or columns.numel() == 0:
        raise ValueError("pano_rows and pano_columns must be non-empty")
    if not bool(torch.all(rows[1:] > rows[:-1])):
        raise ValueError("pano_rows must be strictly increasing")
    if not bool(torch.all(columns[1:] > columns[:-1])):
        raise ValueError("pano_columns must be strictly increasing")
    return torch.meshgrid(rows, columns, indexing="ij")


def _matrix_to_euler_xyz(rotation: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of SciPy ``Rotation.as_euler('xyz')`` off gimbal lock."""
    sy = torch.sqrt(
        rotation[..., 0, 0].square() + rotation[..., 1, 0].square())
    regular = sy > 1e-7
    x_regular = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    y_regular = torch.atan2(-rotation[..., 2, 0], sy)
    z_regular = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    x_singular = torch.atan2(-rotation[..., 1, 2], rotation[..., 1, 1])
    y_singular = torch.atan2(-rotation[..., 2, 0], sy)
    z_singular = torch.zeros_like(z_regular)
    return torch.stack([
        torch.where(regular, x_regular, x_singular),
        torch.where(regular, y_regular, y_singular),
        torch.where(regular, z_regular, z_singular),
    ], dim=-1)


def _normalize_grid_coordinates(
    pixel_x: torch.Tensor,
    pixel_y: torch.Tensor,
    source_size: tuple[int, int],
    *,
    coordinate_convention: str,
) -> torch.Tensor:
    source_height, source_width = source_size
    # align_corners=False maps normalized -1/+1 to the outer image edges.
    # gsplat/K projections already use that edge-coordinate convention
    # (raster pixels are x+0.5/y+0.5). OpenCV remap instead takes zero-based
    # pixel-index coordinates, which need +0.5 before normalization.
    if coordinate_convention == "edge":
        offset = 0.0
    elif coordinate_convention == "opencv_index":
        offset = 0.5
    else:
        raise ValueError(
            "unknown pixel coordinate convention {!r}".format(
                coordinate_convention))
    grid_x = (pixel_x + offset) * (2.0 / source_width) - 1.0
    grid_y = (pixel_y + offset) * (2.0 / source_height) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


def _calibrated_grid_for_camera(
    camera_to_ego: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_index: int,
    pano_rows: torch.Tensor,
    pano_columns: torch.Tensor,
    source_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    azimuth = (
        (pano_columns - PANO_W / 2.0 - 0.5) / PANO_W
    ) * (2.0 * math.pi)
    elevation = -(
        (pano_rows - PANO_H_PRE_CROP / 2.0 - 0.5)
        / PANO_H_PRE_CROP
    ) * (2.0 * VERT_FOV_HALF_RAD)
    cos_elevation = torch.cos(elevation)
    # OneBEV pano axes (right, forward, up) -> nuScenes ego (forward,left,up).
    directions_ego = torch.stack([
        torch.cos(azimuth) * cos_elevation,
        -torch.sin(azimuth) * cos_elevation,
        torch.sin(elevation),
    ], dim=-1)

    rotation = camera_to_ego[:, camera_index, :3, :3]
    directions_camera = torch.einsum(
        "nij,hwj->nhwi", rotation.transpose(-1, -2), directions_ego)
    projected = torch.einsum(
        "nij,nhwj->nhwi", intrinsics[:, camera_index], directions_camera)
    camera_z = directions_camera[..., 2]
    safe_z = torch.where(
        camera_z.abs() > 1e-8, camera_z, torch.ones_like(camera_z))
    pixel_x = projected[..., 0] / safe_z
    pixel_y = projected[..., 1] / safe_z
    source_height, source_width = source_size
    valid = (
        (camera_z > 1e-8)
        & (pixel_x >= 0.0)
        & (pixel_x <= source_width)
        & (pixel_y >= 0.0)
        & (pixel_y <= source_height)
    )
    grid = _normalize_grid_coordinates(
        pixel_x, pixel_y, source_size, coordinate_convention="edge")
    grid = torch.where(valid[..., None], grid, torch.full_like(grid, 2.0))
    return grid, valid


def _onebev_angles(camera_to_ego: torch.Tensor) -> torch.Tensor:
    euler = _matrix_to_euler_xyz(camera_to_ego[..., :3, :3])
    reference = euler[:, ONEBEV_FRONT_CAMERA_INDEX]
    yaw = reference[:, None, 2] - euler[..., 2]
    yaw = torch.remainder(yaw + math.pi, 2.0 * math.pi) - math.pi
    pitch = euler[..., 1] - reference[:, None, 1]
    # Public OneBEV later passes this radian-valued number to an OpenCV API
    # expecting degrees.  _onebev_legacy_grid_for_camera intentionally
    # preserves that historical behavior for target compatibility.
    roll_value_passed_as_degrees = euler[..., 0] - reference[:, None, 0]
    return torch.stack([yaw, pitch, roll_value_passed_as_degrees], dim=-1)


def _onebev_legacy_grid_for_camera(
    camera_to_ego: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_index: int,
    pano_rows: torch.Tensor,
    pano_columns: torch.Tensor,
    source_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    source_height, source_width = source_size
    angles = _onebev_angles(camera_to_ego)
    angle_x = angles[:, camera_index, 0]
    angle_y = angles[:, camera_index, 1]
    legacy_roll_degrees = angles[:, camera_index, 2]

    translations = camera_to_ego[..., :3, 3]
    translation_offset = translations - translations.mean(dim=1, keepdim=True)
    delta_u = intrinsics[..., 0, 0] * (translation_offset[..., 0] / 5.5)
    delta_v = intrinsics[..., 1, 1] * (translation_offset[..., 1] / 5.5)
    r_offset = torch.sqrt(delta_u.square() + delta_v.square())
    fov_deg = (
        ONEBEV_BACK_FOV_DEG if camera_index == 0 else ONEBEV_OTHER_FOV_DEG)
    radius = (
        source_width / 2.0 / math.tan(math.radians(fov_deg) / 2.0)
        + r_offset[:, camera_index]
    )

    angle_x = angle_x[:, None, None]
    angle_y = angle_y[:, None, None]
    radius = radius[:, None, None]
    pano_azimuth = (
        (pano_columns - PANO_W / 2.0 - 0.5) / PANO_W
    ) * (2.0 * math.pi)
    pano_elevation = -(
        (pano_rows - PANO_H_PRE_CROP / 2.0 - 0.5)
        / PANO_H_PRE_CROP
    ) * (2.0 * VERT_FOV_HALF_RAD)

    x0 = radius * torch.cos(angle_y) * torch.sin(angle_x)
    y0 = radius * torch.cos(angle_y) * torch.cos(angle_x)
    z0 = radius * torch.sin(angle_y)
    alpha = torch.cos(pano_elevation) * torch.sin(pano_azimuth)
    beta = torch.cos(pano_elevation) * torch.cos(pano_azimuth)
    gamma = torch.sin(pano_elevation)
    division = x0 * alpha + y0 * beta + z0 * gamma
    safe_division = torch.where(
        division.abs() > 1e-8, division, torch.ones_like(division))
    scale = radius.square() / safe_division
    x1 = scale * alpha
    y1 = scale * beta
    z1 = scale * gamma

    vector_x = x1 - x0
    vector_y = y1 - y0
    vector_z = z1 - z0
    positive_x_x = torch.cos(angle_x)
    positive_x_y = -torch.sin(angle_x)
    delta_x = vector_x * positive_x_x + vector_y * positive_x_y
    positive_y_x = -z0 * positive_x_y
    positive_y_y = z0 * positive_x_x
    positive_y_z = x0 * positive_x_y - y0 * positive_x_x
    positive_y_norm = torch.sqrt(
        positive_y_x.square() + positive_y_y.square()
        + positive_y_z.square()).clamp_min(1e-8)
    delta_y = (
        vector_x * positive_y_x + vector_y * positive_y_y
        + vector_z * positive_y_z
    ) / positive_y_norm

    # These are the public OneBEV coordinates before warp_image_fast's crop.
    pixel_x = delta_x + (source_width + 1.0) / 2.0
    pixel_y = delta_y + (source_height + 1.0) / 2.0

    # warp_image_fast crops at minX/minY and remaps with ``Px-minX+1``.
    # Re-express that operation directly in full-image coordinates.  The
    # crop only changes the center used by its tiny historical roll.
    min_x = torch.maximum(
        torch.ones_like(pixel_x[:, 0, 0]),
        torch.floor(pixel_x.amin(dim=(-2, -1))) - 1.0,
    )
    min_y = torch.maximum(
        torch.ones_like(pixel_y[:, 0, 0]),
        torch.floor(pixel_y.amin(dim=(-2, -1))) - 1.0,
    )
    max_x = torch.minimum(
        torch.full_like(min_x, float(source_width)),
        torch.ceil(pixel_x.amax(dim=(-2, -1))) + 1.0,
    )
    max_y = torch.minimum(
        torch.full_like(min_y, float(source_height)),
        torch.ceil(pixel_y.amax(dim=(-2, -1))) + 1.0,
    )
    crop_center_x = ((max_x - min_x) / 2.0)[:, None, None]
    crop_center_y = ((max_y - min_y) / 2.0)[:, None, None]
    destination_x = pixel_x - min_x[:, None, None] + 1.0
    destination_y = pixel_y - min_y[:, None, None] + 1.0

    # cv2.getRotationMatrix2D receives the radian-valued legacy roll as if it
    # were degrees.  Invert that source->destination matrix because
    # grid_sample, like cv2.remap, needs destination->source coordinates.
    roll_radians_used_by_opencv = (
        legacy_roll_degrees * (math.pi / 180.0))[:, None, None]
    cosine = torch.cos(roll_radians_used_by_opencv)
    sine = torch.sin(roll_radians_used_by_opencv)
    centered_x = destination_x - crop_center_x
    centered_y = destination_y - crop_center_y
    source_crop_x = cosine * centered_x - sine * centered_y + crop_center_x
    source_crop_y = sine * centered_x + cosine * centered_y + crop_center_y
    pixel_x = source_crop_x + min_x[:, None, None]
    pixel_y = source_crop_y + min_y[:, None, None]

    valid = (
        (division > 1e-8)
        & (pixel_x >= -0.5)
        & (pixel_x <= source_width - 0.5)
        & (pixel_y >= -0.5)
        & (pixel_y <= source_height - 0.5)
    )
    grid = _normalize_grid_coordinates(
        pixel_x, pixel_y, source_size,
        coordinate_convention="opencv_index")
    grid = torch.where(valid[..., None], grid, torch.full_like(grid, 2.0))
    return grid, valid


def _validate_stitch_inputs(
    camera_images: torch.Tensor,
    camera_to_ego: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
    if camera_images.ndim < 4:
        raise ValueError(
            "camera_images must be [...,6,C,H,W], got {}"
            .format(camera_images.shape))
    if camera_images.shape[-4] != NUSCENES_CAMERA_COUNT:
        raise ValueError(
            "camera_images camera dimension must be 6, got {}"
            .format(camera_images.shape[-4]))
    leading_shape = tuple(camera_images.shape[:-4])
    batch_count = math.prod(leading_shape) if leading_shape else 1
    target_transform_shape = leading_shape + (
        NUSCENES_CAMERA_COUNT, 4, 4)
    target_intrinsics_shape = leading_shape + (
        NUSCENES_CAMERA_COUNT, 3, 3)
    try:
        camera_to_ego = torch.broadcast_to(
            camera_to_ego, target_transform_shape)
        intrinsics = torch.broadcast_to(intrinsics, target_intrinsics_shape)
    except RuntimeError as error:
        raise ValueError(
            "calibration leading dimensions must broadcast to {}; got {} and {}"
            .format(leading_shape, camera_to_ego.shape, intrinsics.shape)
        ) from error
    compute_dtype = (
        torch.float64 if camera_images.dtype == torch.float64
        else torch.float32)
    device = camera_images.device
    return (
        leading_shape,
        camera_images.reshape(
            batch_count, *camera_images.shape[-4:]),
        camera_to_ego.reshape(batch_count, 6, 4, 4).to(
            device=device, dtype=compute_dtype),
        intrinsics.reshape(batch_count, 6, 3, 3).to(
            device=device, dtype=compute_dtype),
    )


def _flatten_calibration(
    camera_to_ego: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
    if camera_to_ego.shape[-3:] != (6, 4, 4):
        raise ValueError(
            "camera_to_ego must end in (6,4,4), got {}".format(
                camera_to_ego.shape))
    if intrinsics.shape[-3:] != (6, 3, 3):
        raise ValueError(
            "intrinsics must end in (6,3,3), got {}".format(
                intrinsics.shape))
    try:
        leading_shape = torch.broadcast_shapes(
            camera_to_ego.shape[:-3], intrinsics.shape[:-3])
    except RuntimeError as error:
        raise ValueError(
            "calibration leading shapes do not broadcast: {} and {}".format(
                camera_to_ego.shape[:-3], intrinsics.shape[:-3])) from error
    transforms = torch.broadcast_to(
        camera_to_ego, leading_shape + (6, 4, 4)).reshape(-1, 6, 4, 4)
    matrices = torch.broadcast_to(
        intrinsics, leading_shape + (6, 3, 3)).reshape(-1, 6, 3, 3)
    return (
        tuple(leading_shape),
        transforms.to(device=device, dtype=dtype),
        matrices.to(device=device, dtype=dtype),
    )


def build_owned_pano_sampling_grid(
    camera_to_ego: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    source_size: tuple[int, int],
    pano_rows: torch.Tensor,
    pano_columns: torch.Tensor,
    projection_model: str,
    seam_erosion_px: int = 0,
    legacy_mapping_size: tuple[int, int] | None = NUSCENES_IMAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the source grid selected by the hard-seam owner at every pixel.

    ``intrinsics`` must be in gsplat edge coordinates at ``source_size``;
    build it from dataset/OpenCV K with :func:`prepare_gsplat_intrinsics`.

    Returns ``(grid, owner, valid)``.  ``grid`` has shape
    ``[...,H,W,2]`` in ``grid_sample(align_corners=False)`` coordinates,
    ``owner`` is the calibration-independent ``[H,W]`` camera-index map, and
    ``valid`` is ``[...,H,W]`` after camera-FOV and seam checks.  The explicit
    axes make this suitable for :func:`half_canvas_pano_axes`.

    ``legacy_mapping_size`` defaults to native nuScenes 900x1600.  This is
    important even when ``source_size`` is a cheaper 360x640 Gaussian render:
    public OneBEV computed its ``(W+1)/2`` and ``+1`` remap offsets at native
    resolution.  The resulting normalized grid can sample a proportionally
    resized render exactly.  Set it to ``None`` only to run the legacy formula
    natively at ``source_size`` (mainly useful for synthetic tests).
    """
    if projection_model not in SUPPORTED_PROJECTION_MODELS:
        raise ValueError(
            "projection_model must be one of {}, got {!r}".format(
                SUPPORTED_PROJECTION_MODELS, projection_model))
    source_height, source_width = [int(value) for value in source_size]
    if source_height <= 0 or source_width <= 0:
        raise ValueError("source_size entries must be positive")
    device = camera_to_ego.device
    dtype = (
        torch.float64 if camera_to_ego.dtype == torch.float64
        else torch.float32)
    leading_shape, transforms, matrices = _flatten_calibration(
        camera_to_ego, intrinsics, device=device, dtype=dtype)
    if projection_model == "onebev_legacy" \
            and legacy_mapping_size is not None:
        mapping_height, mapping_width = [
            int(value) for value in legacy_mapping_size]
        if mapping_height <= 0 or mapping_width <= 0:
            raise ValueError("legacy_mapping_size entries must be positive")
        # ``matrices`` are already in gsplat edge coordinates here; resize
        # without applying the OpenCV->gsplat +0.5 conversion a second time.
        mapping_intrinsics = resize_intrinsics(
            matrices, (source_height, source_width),
            (mapping_height, mapping_width))
        mapping_source_size = (mapping_height, mapping_width)
    else:
        mapping_intrinsics = matrices
        mapping_source_size = (source_height, source_width)
    rows, columns = _mesh_pano_axes(
        pano_rows, pano_columns, device=device, dtype=dtype)
    owner_columns, seam_valid_columns = hard_seam_owners_from_columns(
        columns[0], seam_erosion_px=seam_erosion_px)
    owner = owner_columns[None].expand(rows.shape)

    owned_grid = torch.zeros(
        transforms.shape[0], *rows.shape, 2, device=device, dtype=dtype)
    owned_valid = torch.zeros(
        transforms.shape[0], *rows.shape, device=device, dtype=torch.bool)
    for camera_index in range(NUSCENES_CAMERA_COUNT):
        if projection_model == "calibrated":
            grid, camera_valid = _calibrated_grid_for_camera(
                transforms, matrices, camera_index, rows, columns,
                (source_height, source_width))
        else:
            grid, camera_valid = _onebev_legacy_grid_for_camera(
                transforms, mapping_intrinsics, camera_index, rows, columns,
                mapping_source_size)
        select = (owner == camera_index)[None, ..., None]
        owned_grid = torch.where(select, grid, owned_grid)
        owned_valid = torch.where(
            select[..., 0], camera_valid, owned_valid)
    owned_valid &= seam_valid_columns[None, None, :]
    owned_grid = torch.where(
        owned_valid[..., None], owned_grid, torch.full_like(owned_grid, 2.0))
    return (
        owned_grid.reshape(leading_shape + tuple(rows.shape) + (2,)),
        owner,
        owned_valid.reshape(leading_shape + tuple(rows.shape)),
    )


def sampling_grid_to_ego_rays(
    sampling_grid: torch.Tensor,
    owner: torch.Tensor,
    camera_to_ego: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    source_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover exact content rays represented by an owned sampling grid.

    This converts the sampled source pixel through its real gsplat-convention
    ``K`` and ``R``;
    it therefore exposes the ray actually contributing RGB even when the
    legacy OneBEV warp differs from an ideal sphere.  Returns ego-frame
    ``(origins, unit_directions, source_camera_z_cosine)``.  For a stitched
    z-depth target, Euclidean ray range is ``depth_z / z_cosine``.
    """
    source_height, source_width = [int(value) for value in source_size]
    if sampling_grid.shape[-1] != 2 or sampling_grid.ndim < 3:
        raise ValueError(
            "sampling_grid must be [...,H,W,2], got {}".format(
                sampling_grid.shape))
    height, width = sampling_grid.shape[-3:-1]
    if owner.shape != (height, width):
        raise ValueError(
            "owner must have shape ({},{}), got {}".format(
                height, width, owner.shape))
    leading_shape = sampling_grid.shape[:-3]
    calibration_leading, transforms, matrices = _flatten_calibration(
        camera_to_ego, intrinsics, device=sampling_grid.device,
        dtype=sampling_grid.dtype)
    try:
        broadcast_leading = torch.broadcast_shapes(
            leading_shape, calibration_leading)
    except RuntimeError as error:
        raise ValueError(
            "grid/calibration leading shapes do not broadcast: {} and {}"
            .format(leading_shape, calibration_leading)) from error
    if tuple(broadcast_leading) != tuple(calibration_leading):
        # Re-run against the fully broadcast shape.  This is uncommon (a
        # shared calibration with a batched grid) but keeps the pure API sane.
        camera_to_ego = torch.broadcast_to(
            camera_to_ego, broadcast_leading + (6, 4, 4))
        intrinsics = torch.broadcast_to(
            intrinsics, broadcast_leading + (6, 3, 3))
        _, transforms, matrices = _flatten_calibration(
            camera_to_ego, intrinsics, device=sampling_grid.device,
            dtype=sampling_grid.dtype)
    grid = torch.broadcast_to(
        sampling_grid, broadcast_leading + (height, width, 2)).reshape(
            -1, height, width, 2)
    batch_count = grid.shape[0]
    # K and gsplat use image-edge coordinates; raster pixel (i,j) is sampled
    # at (i+0.5,j+0.5).  A legacy OpenCV-index grid was normalized with that
    # same +0.5, so this single inverse is correct for both projection modes.
    pixel_x = (grid[..., 0] + 1.0) * (source_width / 2.0)
    pixel_y = (grid[..., 1] + 1.0) * (source_height / 2.0)
    homogeneous = torch.stack([
        pixel_x, pixel_y, torch.ones_like(pixel_x)], dim=-1)

    origins = torch.zeros(
        batch_count, height, width, 3,
        device=grid.device, dtype=grid.dtype)
    directions = torch.zeros_like(origins)
    z_cosine = torch.zeros(
        batch_count, height, width,
        device=grid.device, dtype=grid.dtype)
    owner = owner.to(device=grid.device)
    for camera_index in range(NUSCENES_CAMERA_COUNT):
        direction_camera = torch.einsum(
            "nij,nhwj->nhwi",
            torch.linalg.inv(matrices[:, camera_index]), homogeneous)
        direction_camera = F.normalize(direction_camera, dim=-1)
        direction_ego = torch.einsum(
            "nij,nhwj->nhwi",
            transforms[:, camera_index, :3, :3], direction_camera)
        select = (owner == camera_index)[None, ..., None]
        origins = torch.where(
            select,
            transforms[:, camera_index, :3, 3][:, None, None, :],
            origins)
        directions = torch.where(select, direction_ego, directions)
        z_cosine = torch.where(
            select[..., 0], direction_camera[..., 2], z_cosine)
    output_shape = tuple(broadcast_leading) + (height, width)
    return (
        origins.reshape(output_shape + (3,)),
        directions.reshape(output_shape + (3,)),
        z_cosine.reshape(output_shape),
    )


def stitch_camera_views_from_owned_grid(
    camera_images: torch.Tensor,
    sampling_grid: torch.Tensor,
    owner: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a reusable owned grid to six views with differentiable sampling.

    This is the fast training-loop entry point when calibration is constant:
    precompute :func:`build_owned_pano_sampling_grid` once, then call this
    function for each set of six Gaussian renders.  Only the images need a
    gradient.
    """
    if camera_images.ndim < 4 or camera_images.shape[-4] != 6:
        raise ValueError(
            "camera_images must be [...,6,C,H,W], got {}".format(
                camera_images.shape))
    if sampling_grid.ndim < 3 or sampling_grid.shape[-1] != 2:
        raise ValueError(
            "sampling_grid must be broadcastable [...,H,W,2], got {}".format(
                sampling_grid.shape))
    height, width = sampling_grid.shape[-3:-1]
    if owner.shape != (height, width):
        raise ValueError(
            "owner must have shape ({},{}), got {}".format(
                height, width, owner.shape))
    leading_shape = tuple(camera_images.shape[:-4])
    grid_leading = tuple(sampling_grid.shape[:-3])
    valid_leading = tuple(valid_mask.shape[:-2])
    try:
        torch.broadcast_shapes(leading_shape, grid_leading, valid_leading)
    except RuntimeError as error:
        raise ValueError(
            "image/grid/valid leading shapes do not broadcast: {}, {}, {}"
            .format(leading_shape, grid_leading, valid_leading)) from error
    # The output follows camera_images' leading shape.  Requiring the other
    # operands to broadcast *to* it avoids silently introducing an extra dim.
    images = camera_images.reshape(-1, *camera_images.shape[-4:])
    batch_count = images.shape[0]
    try:
        grid = torch.broadcast_to(
            sampling_grid, leading_shape + (height, width, 2)).reshape(
                batch_count, height, width, 2)
        valid = torch.broadcast_to(
            valid_mask, leading_shape + (height, width)).reshape(
                batch_count, height, width)
    except RuntimeError as error:
        raise ValueError(
            "sampling_grid and valid_mask must broadcast to image leading "
            "shape {}".format(leading_shape)) from error
    grid = grid.to(device=images.device, dtype=images.dtype)
    valid = valid.to(device=images.device)
    owner = owner.to(device=images.device)

    changes = torch.nonzero(
        owner[0, 1:] != owner[0, :-1], as_tuple=False).flatten()
    # Ownership must be constant over rows for a vertical hard seam.
    if not torch.equal(owner, owner[0:1].expand_as(owner)):
        raise ValueError("owner must be constant over panorama rows")
    run_starts = [0] + [int(value) + 1 for value in changes.cpu().tolist()]
    run_stops = run_starts[1:] + [width]
    output_runs = []
    valid_runs = []
    for run_start, run_stop in zip(run_starts, run_stops):
        camera_index = int(owner[0, run_start])
        run_valid = valid[:, None, :, run_start:run_stop]
        sampled = F.grid_sample(
            images[:, camera_index], grid[:, :, run_start:run_stop],
            mode="bilinear", padding_mode="zeros", align_corners=False)
        output_runs.append(sampled * run_valid.to(dtype=sampled.dtype))
        valid_runs.append(run_valid)
    panorama = torch.cat(output_runs, dim=-1).reshape(
        leading_shape + (
            camera_images.shape[-3], height, width))
    stitched_valid = torch.cat(valid_runs, dim=-1).reshape(
        leading_shape + (1, height, width))
    return panorama, stitched_valid


def differentiable_hard_seam_stitch(
    camera_images: torch.Tensor,
    camera_to_ego: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    output_size: tuple[int, int] | None = None,
    column_range: tuple[float, float] = (0.0, float(PANO_W)),
    row_range: tuple[float, float] = (
        float(PANO_CROP_TOP), float(PANO_CROP_TOP + PANO_H)),
    pano_rows: torch.Tensor | None = None,
    pano_columns: torch.Tensor | None = None,
    projection_model: str,
    seam_erosion_px: int = 0,
    legacy_mapping_size: tuple[int, int] | None = NUSCENES_IMAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp six real-camera images and select the fixed hard-seam owner.

    Args:
        camera_images: ``[...,6,C,Hcam,Wcam]``.  These may be Gaussian
            renders; gradients flow from the stitched panorama back into them.
        camera_to_ego: broadcastable ``[...,6,4,4]`` OpenCV-convention c2w.
        intrinsics: broadcastable gsplat-convention ``[...,6,3,3]`` for the
            actual resolution of ``camera_images``.  Use
            :func:`prepare_gsplat_intrinsics` on raw dataset K.
        pano_rows, pano_columns: optional explicit 1-D pre-crop/canonical axes.
            Pass the result of :func:`half_canvas_pano_axes` to reproduce the
            non-uniform 4114-wide training canvas exactly.  When omitted,
            axes are generated from ``output_size``/ranges.
        projection_model: mandatory ``"onebev_legacy"`` or ``"calibrated"``.
            It is intentionally not defaulted because choosing the wrong one
            silently recreates the geometry/target mismatch this module fixes.
        legacy_mapping_size: native resolution at which the old OneBEV warp
            constants were evaluated.  Its normalized grid is then applied
            to ``camera_images`` at any proportional render resolution.

    Returns:
        ``(panorama, valid_mask)`` with shapes ``[...,C,Hout,Wout]`` and
        ``[...,1,Hout,Wout]``.  Invalid camera-FOV and eroded-seam pixels are
        zero in the panorama and false in the mask.

    The implementation samples only contiguous owner runs rather than all six
    cameras over the complete panorama.  This keeps the differentiable grid
    at O(HW), not O(6HW), which matters for a 9600-wide target.
    """
    if projection_model not in SUPPORTED_PROJECTION_MODELS:
        raise ValueError(
            "projection_model must be one of {}, got {!r}".format(
                SUPPORTED_PROJECTION_MODELS, projection_model))
    if (pano_rows is None) != (pano_columns is None):
        raise ValueError(
            "pano_rows and pano_columns must be supplied together")
    if pano_rows is not None:
        inferred_output_size = (int(pano_rows.numel()), int(pano_columns.numel()))
        if output_size is not None and tuple(output_size) != inferred_output_size:
            raise ValueError(
                "output_size {} disagrees with explicit axes {}".format(
                    output_size, inferred_output_size))
        output_size = inferred_output_size
    elif output_size is None:
        output_size = (PANO_H, PANO_W)
    output_height, output_width = [int(value) for value in output_size]
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output_size entries must be positive")
    row_start, row_stop = [float(value) for value in row_range]
    if not (0.0 <= row_start < row_stop <= float(PANO_H_PRE_CROP)):
        raise ValueError(
            "row_range must lie inside the {}-row pre-crop pano, got {}"
            .format(PANO_H_PRE_CROP, row_range))

    leading_shape, images, transforms, intrinsics = _validate_stitch_inputs(
        camera_images, camera_to_ego, intrinsics)
    _, _, channel_count, source_height, source_width = images.shape
    if pano_rows is None:
        row_start, row_stop = row_range
        column_start, column_stop = column_range
        explicit_rows = row_start + (
            torch.arange(
                output_height, device=images.device, dtype=transforms.dtype)
            + 0.5
        ) * ((row_stop - row_start) / output_height) - 0.5
        explicit_columns = column_start + (
            torch.arange(
                output_width, device=images.device, dtype=transforms.dtype)
            + 0.5
        ) * ((column_stop - column_start) / output_width) - 0.5
    else:
        explicit_rows = pano_rows.to(
            device=images.device, dtype=transforms.dtype)
        explicit_columns = pano_columns.to(
            device=images.device, dtype=transforms.dtype)
    sampling_grid, owner, valid = build_owned_pano_sampling_grid(
        transforms, intrinsics,
        source_size=(source_height, source_width),
        pano_rows=explicit_rows, pano_columns=explicit_columns,
        projection_model=projection_model,
        seam_erosion_px=seam_erosion_px,
        legacy_mapping_size=legacy_mapping_size)
    panorama, valid_mask = stitch_camera_views_from_owned_grid(
        images, sampling_grid, owner, valid)
    return (
        panorama.reshape(
            leading_shape + (channel_count, output_height, output_width)),
        valid_mask.reshape(
            leading_shape + (1, output_height, output_width)),
    )
