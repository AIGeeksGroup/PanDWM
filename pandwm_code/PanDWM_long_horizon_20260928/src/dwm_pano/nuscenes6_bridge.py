"""Protocol bridge: the 360x33 pano band -> the six real nuScenes cameras.

Why this exists
---------------
Every number this project reports lives in its own panorama space, so none of
them can be placed beside a published nuScenes figure.  This module maps a
pano frame into the six *real* camera frames (real ``camera_intrinsics`` and
real ``sensor2ego`` from the dataset), which is the space every public driving
video-generation number is reported in.

Two facts make it a pure resample, with no Gaussians and no depth:

1.  The hard-seam pano was stitched so that each Voronoi region holds content
    as seen from **that region's own camera translation** (see
    :func:`dwm_pano.datasets.pano_projection.project_ego_point_to_pano_with_cams`).
    A pixel ray of camera ``c`` therefore already starts at ``t_c``, so its
    azimuth/elevation -- and hence its pano column/row -- is **independent of
    the unknown scene depth**, as long as the pano region it lands in is owned
    by ``c`` itself.  Residual parallax appears only near a Voronoi seam, where
    the landing region belongs to a neighbour that saw the world from a
    different origin.  :func:`resample_pano_to_real_camera` returns that
    fraction per frame so it is measured rather than assumed.

2.  gsplat/Gaussian decoding is *not* needed for an appearance/temporal
    distribution metric.  The G2 decoder currently sits at ~19.8 dB future
    PSNR with a ~0 dB gain over its own no-GS control, so routing through it
    would add a one-sided instrument penalty that a natively-6-view external
    method never pays.

The hard limit that no resample can remove
------------------------------------------
The pano is a +-16.67 deg elevation band.  The real cameras are taller, so a
full-frame 6-camera protocol is geometrically impossible: about 16% of the rows
of the five ~39 deg cameras have no pano content at all, and **46% of CAM_BACK**
(58.1 deg vertical span) does.  Any bridged metric is therefore *band limited*
and must be tagged as such; it cannot be compared against a published
full-frame number.  Every arm has to pass through the identical crop instead.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch

from dwm_pano.datasets.pano_projection import (
    CAM_ORDER,
    NEAR_CLIP_RADIUS,
    PANO_CROP_TOP,
    PANO_H,
    PANO_H_PRE_CROP,
    PANO_W,
    VERT_FOV_HALF_RAD,
    _cam_idx_for_angle,
)

# Elevation half-extent of the FINAL 600-row pano.  Same expression as
# dwm_pano.gs_geometry.PANO_LAT_HALF_RAD; kept as one authority here so the
# bridge cannot drift from the projection it inverts.
PANO_BAND_HALF_RAD = VERT_FOV_HALF_RAD * PANO_H / PANO_H_PRE_CROP

# Columns the pipeline drops from the resized right half before concatenation,
# giving the 4114-wide V=1 canvas.  Mirrored from
# `dwm_pano.pipelines.ctsd_pano_gs_decoder.PANO_CANVAS_OVERLAP_COLUMNS` rather
# than imported, because that module pulls in diffusers and this one must stay
# importable with numpy and torch alone.  test_nuscenes6_bridge pins the two
# together by reading the pipeline source, so the copy cannot drift silently.
PANO_CANVAS_OVERLAP_COLUMNS = 494

def canvas_source_axes(half_size, right_trim_px: int | None = None):
    """Post-crop pano (row, column) coordinate of every V=1 canvas pixel.

    The evaluation pipeline never materialises a 600x9600 pano: it produces
    the 4114-wide V=1 canvas by resizing the two halves independently and
    dropping the overlap band.  Re-deriving that map with a single global
    resize is wrong, so this delegates to
    :func:`dwm_pano.calibrated_stitch.half_canvas_pano_axes`, which is the
    authority the GS pipeline already uses, and only shifts the rows from the
    900-row pre-crop frame into the 600-row post-crop frame the bridge works
    in.  Returned coordinates are 0-based pixel centres.
    """
    from dwm_pano.calibrated_stitch import half_canvas_pano_axes
    if right_trim_px is None:
        right_trim_px = PANO_CANVAS_OVERLAP_COLUMNS
    rows, columns = half_canvas_pano_axes(
        half_size=half_size, right_trim_px=int(right_trim_px),
        dtype=torch.float64)
    return (rows.numpy() - PANO_CROP_TOP, columns.numpy())


# Radius used only to pick a Voronoi owner for the parallax report.  It never
# enters the sampling grid (see the module docstring), so the resample itself
# is depth free.  50 m is the far end of typical nuScenes annotation range.
OWNER_PROBE_RADIUS_M = 50.0


def real_camera_valid_row_band(
    intrinsics, image_height: int, pitch_rad: float = 0.0
) -> tuple[float, float]:
    """Row interval of one real camera that the pano band can fill.

    Uses the camera-frame vertical ray angle, so it is exact for a camera whose
    optical axis is horizontal and a good approximation otherwise; pass
    ``pitch_rad`` to shift the band for a pitched rig.  Returned bounds are
    fractional rows, clipped to ``[0, image_height]``.
    """
    k = np.asarray(intrinsics, dtype=np.float64)
    focal_y = float(k[1, 1])
    center_y = float(k[1, 2])
    if focal_y <= 0.0:
        raise ValueError("intrinsics[1, 1] must be positive, got {}".format(
            focal_y))
    low = center_y + focal_y * math.tan(-PANO_BAND_HALF_RAD - pitch_rad)
    high = center_y + focal_y * math.tan(PANO_BAND_HALF_RAD - pitch_rad)
    return (max(0.0, min(low, high)),
            min(float(image_height), max(low, high)))


def real_camera_band_table(
    camera_to_ego, intrinsics, image_size, camera_order: Sequence[str] = CAM_ORDER
) -> list[dict]:
    """Per-camera band coverage for a whole rig, for logging and tests."""
    c2w = np.asarray(camera_to_ego, dtype=np.float64)
    k_all = np.asarray(intrinsics, dtype=np.float64)
    sizes = np.asarray(image_size)
    if not (c2w.shape[0] == k_all.shape[0] == sizes.shape[0]
            == len(camera_order)):
        raise ValueError(
            "rig arrays must all have {} entries, got {} / {} / {}".format(
                len(camera_order), c2w.shape[0], k_all.shape[0],
                sizes.shape[0]))
    rows = []
    for index, name in enumerate(camera_order):
        width, height = int(sizes[index][0]), int(sizes[index][1])
        low, high = real_camera_valid_row_band(k_all[index], height)
        focal_y = float(k_all[index][1, 1])
        center_y = float(k_all[index][1, 2])
        rows.append({
            "camera": name,
            "width": width,
            "height": height,
            "row_low": low,
            "row_high": high,
            "covered_rows": high - low,
            "covered_fraction": (high - low) / height,
            "v_fov_top_deg": math.degrees(
                math.atan((0.0 - center_y) / focal_y)),
            "v_fov_bottom_deg": math.degrees(
                math.atan((height - center_y) / focal_y)),
        })
    return rows


def real_camera_pixel_directions_ego(
    camera_to_ego, intrinsics, height: int, width: int,
    row_offset: float = 0.0, row_stride: float = 1.0,
    column_offset: float = 0.0, column_stride: float = 1.0,
):
    """Unit ego-frame ray directions at the pixel centres of one real camera.

    ``camera_to_ego`` is the dataset's OpenCV-convention 4x4; ``intrinsics`` is
    the raw nuScenes pixel-index K.  The offset/stride pair lets a caller
    address a cropped and/or resized output grid in *native* pixel
    coordinates, which is how the band crop stays exact instead of being
    re-derived from a resized K.
    """
    c2w = np.asarray(camera_to_ego, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    column_index = np.arange(width, dtype=np.float64)
    row_index = np.arange(height, dtype=np.float64)
    u_native = column_offset + (column_index + 0.5) * column_stride
    v_native = row_offset + (row_index + 0.5) * row_stride
    u_grid, v_grid = np.meshgrid(u_native, v_native, indexing="xy")
    x_camera = (u_grid - k[0, 2]) / k[0, 0]
    y_camera = (v_grid - k[1, 2]) / k[1, 1]
    directions_camera = np.stack(
        [x_camera, y_camera, np.ones_like(x_camera)], axis=-1)
    directions_ego = directions_camera @ c2w[:3, :3].T
    norm = np.linalg.norm(directions_ego, axis=-1, keepdims=True)
    return directions_ego / np.maximum(norm, 1e-12)


def ego_directions_to_pano_pixels(directions_ego):
    """Pano (tx, ty, in_band) for ego-frame ray *directions*.

    This is :func:`project_ego_points_to_pano` specialised to a direction: the
    azimuth/elevation of a ray does not depend on how far along it a point
    sits, which is exactly why the bridge needs no depth.
    """
    directions = np.asarray(directions_ego, dtype=np.float64)
    x_ego = directions[..., 0]
    y_ego = directions[..., 1]
    z_ego = directions[..., 2]
    pano_x = -y_ego
    pano_y = x_ego
    pano_z = z_ego
    radius = np.sqrt(pano_x ** 2 + pano_y ** 2 + pano_z ** 2)
    azimuth = np.arctan2(pano_x, pano_y)
    elevation = np.arcsin(
        np.clip(pano_z / np.maximum(radius, 1e-9), -1.0, 1.0))
    tx = (azimuth / (2.0 * np.pi) * PANO_W + PANO_W / 2.0 + 0.5) % PANO_W
    ty_pre_crop = -elevation / VERT_FOV_HALF_RAD * (PANO_H_PRE_CROP / 2.0) \
        + PANO_H_PRE_CROP / 2.0 + 0.5
    ty = ty_pre_crop - PANO_CROP_TOP
    in_band = (ty >= 0.0) & (ty < PANO_H)
    return tx, ty, in_band


def voronoi_owner_indices(directions_ego, camera_translation,
                          radius_m: float = OWNER_PROBE_RADIUS_M):
    """Which pano region a camera's rays land in, at a probe radius.

    The pano's own owner rule takes the azimuth of the point measured from the
    **ego origin**, so a ray leaving ``camera_translation`` can land in a
    neighbour's region.  Where it does, that region's content was rendered
    from a different origin and the depth-free resample carries a parallax
    error.  Reporting this is the point of the function.
    """
    directions = np.asarray(directions_ego, dtype=np.float64)
    translation = np.asarray(camera_translation, dtype=np.float64).reshape(3)
    points = translation + directions * float(radius_m)
    azimuth_from_ego = np.arctan2(-points[..., 1], points[..., 0])
    flat = azimuth_from_ego.reshape(-1)
    owners = np.fromiter(
        (_cam_idx_for_angle(float(angle)) for angle in flat),
        dtype=np.int64, count=flat.size)
    return owners.reshape(azimuth_from_ego.shape)


def build_camera_sampling_grid(
    camera_to_ego, intrinsics, image_size, source_size: tuple[int, int],
    output_size: tuple[int, int], band_limit: bool = True,
    camera_index: int | None = None,
    owner_probe_radius_m: float = OWNER_PROBE_RADIUS_M,
    source_axes=None,
):
    """Normalised ``grid_sample`` grid taking a pano source into one camera.

    Split out of the sampler because the grid depends only on calibration and
    sizes, never on pixels: a clip's 32 frames share one rig, and the whole
    dataset has only a couple of distinct rigs, so a caller that caches on a
    calibration signature builds this once instead of per frame.

    Returns ``(grid, valid, report)``; ``grid`` is ``[H, W, 2]`` addressing a
    source that has been padded by one column on each side (see
    :func:`apply_camera_sampling_grid`).
    """
    source_height, source_width = [int(value) for value in source_size]
    native_width, native_height = int(image_size[0]), int(image_size[1])
    output_height, output_width = [int(value) for value in output_size]
    if min(output_height, output_width) <= 0:
        raise ValueError("output_size must be positive, got {}".format(
            output_size))

    if source_axes is None:
        if (source_height, source_width) != (PANO_H, PANO_W):
            raise ValueError(
                "source must be {}x{} without source_axes, got {}x{}".format(
                    PANO_H, PANO_W, source_height, source_width))
        source_rows = None
        source_columns = None
    else:
        source_rows, source_columns = [
            np.asarray(axis, dtype=np.float64) for axis in source_axes]
        if (source_rows.shape[0], source_columns.shape[0]) \
                != (source_height, source_width):
            raise ValueError(
                "source_axes describe a {}x{} grid but the source is {}x{}"
                .format(source_rows.shape[0], source_columns.shape[0],
                        source_height, source_width))

    if band_limit:
        row_low, row_high = real_camera_valid_row_band(
            intrinsics, native_height)
        if row_high - row_low <= 0.0:
            raise ValueError(
                "the pano band does not intersect this camera at all")
    else:
        row_low, row_high = 0.0, float(native_height)
    row_stride = (row_high - row_low) / output_height
    column_stride = native_width / output_width

    directions = real_camera_pixel_directions_ego(
        camera_to_ego, intrinsics, output_height, output_width,
        row_offset=row_low, row_stride=row_stride,
        column_offset=0.0, column_stride=column_stride)
    tx, ty, in_band = ego_directions_to_pano_pixels(directions)

    # align_corners=False maps a normalised x to pixel centre t-0.5, which is
    # the convention `tx`/`ty` already carry (see project_ego_points_to_pano's
    # trailing +0.5).  Shifting by the pad gives (tx+1)/(W_src+2)*2-1.
    if source_rows is None:
        column_centre = tx - 0.5
        row_centre = ty - 0.5
    else:
        # Invert the canvas map.  `source_columns` is strictly increasing
        # (half_canvas_pano_axes asserts it), so np.interp is exact at the
        # sample points and linear between them -- the same assumption the
        # forward resize made.
        column_centre = np.interp(
            tx - 0.5, source_columns, np.arange(source_width, dtype=np.float64))
        row_centre = np.interp(
            ty - 0.5, source_rows, np.arange(source_height, dtype=np.float64))
    grid_x = np.where(
        in_band,
        (column_centre + 1.5) / (source_width + 2.0) * 2.0 - 1.0, 0.0)
    grid_y = np.where(
        in_band, (row_centre + 0.5) / source_height * 2.0 - 1.0, 0.0)
    grid = np.stack([grid_x, grid_y], axis=-1)

    report = {
        "row_low": float(row_low),
        "row_high": float(row_high),
        "native_rows_covered": float(row_high - row_low),
        "native_row_fraction": float((row_high - row_low) / native_height),
        "in_band_pixel_fraction": float(in_band.mean()),
    }
    if camera_index is not None:
        translation = np.asarray(camera_to_ego, dtype=np.float64)[:3, 3]
        owners = voronoi_owner_indices(
            directions, translation, owner_probe_radius_m)
        foreign = owners != int(camera_index)
        report["foreign_region_fraction"] = float(
            foreign[in_band].mean() if in_band.any() else 0.0)
        report["owner_probe_radius_m"] = float(owner_probe_radius_m)
    return grid, in_band, report


def apply_camera_sampling_grid(source, grid, valid):
    """Sample ``source`` with a grid from :func:`build_camera_sampling_grid`.

    ``source`` is ``[N, C, H, W]`` (N may be batch*time); the same grid is
    applied to every entry, which is the whole point of caching it.
    """
    source_tensor = torch.as_tensor(source)
    if source_tensor.ndim != 4:
        raise ValueError(
            "source must be [N, C, H, W], got shape {}".format(
                tuple(source_tensor.shape)))
    # grid_sample cannot wrap, but a ray landing at tx just below the source
    # width has to blend with column 0.  Pad one column from each end so the
    # whole azimuth circle stays continuous.
    padded = torch.cat(
        [source_tensor[..., -1:], source_tensor, source_tensor[..., :1]],
        dim=-1)
    grid_tensor = torch.as_tensor(
        grid, dtype=source_tensor.dtype, device=source_tensor.device)
    grid_batch = grid_tensor[None].expand(
        source_tensor.shape[0], -1, -1, -1)
    sampled = torch.nn.functional.grid_sample(
        padded, grid_batch, mode="bilinear", padding_mode="border",
        align_corners=False)
    mask = torch.as_tensor(valid, device=sampled.device)[None, None]
    return sampled * mask.to(sampled.dtype)


def resample_pano_to_real_camera(
    pano, camera_to_ego, intrinsics, image_size,
    output_size: tuple[int, int], band_limit: bool = True,
    camera_index: int | None = None,
    owner_probe_radius_m: float = OWNER_PROBE_RADIUS_M,
    source_axes=None,
):
    """Sample a single pano frame into one real camera's frame.

    Thin wrapper over build/apply, kept because it reads well for one frame.

    Args:
        pano: ``[C, H, W]`` float tensor -- either the 600x9600 pano, or any
            grid described by ``source_axes`` (e.g. the V=1 canvas).
        camera_to_ego / intrinsics: that camera's real 4x4 and raw 3x3.
        image_size: native ``(width, height)`` the intrinsics belong to.
        output_size: ``(height, width)`` of the returned image.
        band_limit: crop the source row range to the pano band before
            resizing, so the output has no dead rows.  With ``False`` the full
            native frame is addressed and out-of-band pixels come back zero.
        camera_index: index in ``CAM_ORDER``; when given, the returned report
            includes the fraction of pixels landing in another camera's region.

    Returns:
        ``(image, valid, report)`` with ``image`` ``[C, H, W]``, ``valid``
        ``[H, W]`` bool, and ``report`` a dict of measured coverage.
    """
    pano_tensor = torch.as_tensor(pano)
    if pano_tensor.ndim != 3:
        raise ValueError(
            "pano must be [C, H, W], got shape {}".format(
                tuple(pano_tensor.shape)))
    grid, valid, report = build_camera_sampling_grid(
        camera_to_ego, intrinsics, image_size,
        tuple(int(v) for v in pano_tensor.shape[-2:]), output_size,
        band_limit=band_limit, camera_index=camera_index,
        owner_probe_radius_m=owner_probe_radius_m, source_axes=source_axes)
    image = apply_camera_sampling_grid(pano_tensor[None], grid, valid)[0]
    return image, torch.from_numpy(valid), report


def build_rig_sampling_grids(
    camera_to_ego, intrinsics, image_size, source_size: tuple[int, int],
    output_size: tuple[int, int], band_limit: bool = True,
    camera_order: Sequence[str] = CAM_ORDER,
    owner_probe_radius_m: float = OWNER_PROBE_RADIUS_M,
    source_axes=None,
):
    """Per-camera grids for a whole rig; cache this on a calibration key."""
    grids = []
    valids = []
    reports = []
    for index, name in enumerate(camera_order):
        grid, valid, report = build_camera_sampling_grid(
            camera_to_ego[index], intrinsics[index], image_size[index],
            source_size, output_size, band_limit=band_limit,
            camera_index=index, owner_probe_radius_m=owner_probe_radius_m,
            source_axes=source_axes)
        report["camera"] = name
        grids.append(grid)
        valids.append(valid)
        reports.append(report)
    return np.stack(grids), np.stack(valids), reports


def resample_pano_to_rig(
    pano, camera_to_ego, intrinsics, image_size,
    output_size: tuple[int, int], band_limit: bool = True,
    camera_order: Sequence[str] = CAM_ORDER,
    owner_probe_radius_m: float = OWNER_PROBE_RADIUS_M,
    source_axes=None,
):
    """Sample one pano frame into all six real cameras.

    Returns ``(images, valid, reports)`` with ``images`` ``[6, C, H, W]``,
    ``valid`` ``[6, H, W]`` and one report dict per camera.  Output size is
    common across cameras on purpose: the six native bands have different
    heights, and a per-camera output size would reintroduce exactly the
    per-camera squeeze that makes PanoPatch-FVD's per-camera values
    incomparable across cameras.
    """
    pano_tensor = torch.as_tensor(pano)
    if pano_tensor.ndim != 3:
        raise ValueError(
            "pano must be [C, H, W], got shape {}".format(
                tuple(pano_tensor.shape)))
    grids, valids, reports = build_rig_sampling_grids(
        camera_to_ego, intrinsics, image_size,
        tuple(int(v) for v in pano_tensor.shape[-2:]), output_size,
        band_limit=band_limit, camera_order=camera_order,
        owner_probe_radius_m=owner_probe_radius_m, source_axes=source_axes)
    images = torch.stack([
        apply_camera_sampling_grid(pano_tensor[None], grids[index],
                                   valids[index])[0]
        for index in range(grids.shape[0])])
    return images, torch.from_numpy(valids), reports
