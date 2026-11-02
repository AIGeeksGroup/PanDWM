"""Geometry-only instance tracks for temporal/cross-view training losses.

nuScenes already assigns a persistent ``gt_box_ids`` token to each annotated
3D box.  This module projects those boxes onto PanDWM's two overlapping pano
halves and packs one clip into fixed-size tensors that the normal DataLoader
can collate.  RGB pixels are deliberately not touched here: the track is a
piece of supervision, not a new image condition for the frozen teacher.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from pyquaternion import Quaternion

from dwm_pano.datasets.pano_projection import (
    PANO_H, PANO_W, PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE,
    project_ego_points_to_pano,
)
from dwm_pano.datasets.pano_via_stitch import _ego_corners_from_boxes


PANO_HALF_RANGES = (PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE)


def circular_extent_segments(columns, period=PANO_W):
    """Return the shortest circular interval covering ``columns``.

    The result contains one segment for ordinary boxes and two segments when
    a box crosses the 0/``period`` wrap seam.  Finding the complement of the
    largest circular gap is robust to which cuboid corner happens to be first.
    """
    values = np.sort(np.asarray(columns, dtype=np.float64) % period)
    if values.size == 0:
        return []
    if values.size == 1:
        value = float(values[0])
        return [(value, value)]
    gaps = np.diff(np.concatenate((values, values[:1] + period)))
    gap_index = int(np.argmax(gaps))
    start = float(values[(gap_index + 1) % values.size])
    end = float(values[gap_index])
    if start <= end:
        return [(start, end)]
    return [(start, float(period)), (0.0, end)]


def _largest_intersection(segments, interval):
    candidates = []
    for start, end in segments:
        low = max(float(start), float(interval[0]))
        high = min(float(end), float(interval[1]))
        if high > low:
            candidates.append((low, high))
    return max(candidates, key=lambda item: item[1] - item[0]) \
        if candidates else None


def frame_instance_half_boxes(
    info,
    half_size,
    min_pixels=4,
    min_visibility=2,
    excluded_classes=("barrier", "traffic_cone"),
):
    """Project one frame's GT boxes to both resized pano halves.

    Returns ``(ids, boxes, valid)``. ``boxes`` is ``[N, 2, 4]`` in normalized
    ``[x0, y0, x1, y1]`` coordinates; ``valid`` is ``[N, 2]``.  A partially
    visible box is clipped to the relevant half instead of being discarded.
    """
    gt_boxes = np.asarray(info["gt_boxes"], dtype=np.float64)
    if gt_boxes.size == 0:
        return [], np.zeros((0, 2, 4), dtype=np.float32), \
            np.zeros((0, 2), dtype=np.bool_)
    box_ids = list(info["gt_box_ids"])
    if len(box_ids) != len(gt_boxes):
        raise ValueError("gt_box_ids and gt_boxes must have identical length")
    names = list(info.get("gt_names", [""] * len(gt_boxes)))
    visibility = list(info.get("visibility", [4] * len(gt_boxes)))
    if len(names) != len(gt_boxes) or len(visibility) != len(gt_boxes):
        raise ValueError("gt_names/visibility must align with gt_boxes")
    excluded_classes = set(excluded_classes or ())

    corners = _ego_corners_from_boxes(
        gt_boxes,
        Quaternion(info["lidar2ego_rotation"]),
        np.asarray(info["lidar2ego_translation"]),
    )
    half_w, half_h = (float(half_size[0]), float(half_size[1]))
    output_ids, output_boxes, output_valid = [], [], []
    for token, name, visible_level, box_corners in zip(
        box_ids, names, visibility, corners
    ):
        if str(name) in excluded_classes:
            continue
        if int(visible_level) < int(min_visibility):
            continue
        tx, ty, visible = project_ego_points_to_pano(box_corners)
        if int(visible.sum()) < 4:
            continue
        tx, ty = tx[visible], ty[visible]
        y0 = max(0.0, float(np.min(ty)))
        y1 = min(float(PANO_H), float(np.max(ty)))
        if (y1 - y0) * half_h / PANO_H < min_pixels:
            continue
        segments = circular_extent_segments(tx)
        per_view_boxes = np.zeros((2, 4), dtype=np.float32)
        per_view_valid = np.zeros((2,), dtype=np.bool_)
        for view, half_range in enumerate(PANO_HALF_RANGES):
            intersection = _largest_intersection(segments, half_range)
            if intersection is None:
                continue
            x0, x1 = intersection
            source_width = float(half_range[1] - half_range[0])
            pixel_width = (x1 - x0) * half_w / source_width
            if pixel_width < min_pixels:
                continue
            # Store normalized coordinates so this metadata remains valid if
            # the VAE latent resolution changes.
            per_view_boxes[view] = (
                (x0 - half_range[0]) / source_width,
                y0 / PANO_H,
                (x1 - half_range[0]) / source_width,
                y1 / PANO_H,
            )
            per_view_valid[view] = True
        if per_view_valid.any():
            output_ids.append(str(token))
            output_boxes.append(per_view_boxes)
            output_valid.append(per_view_valid)

    return (
        output_ids,
        np.asarray(output_boxes, dtype=np.float32).reshape(-1, 2, 4),
        np.asarray(output_valid, dtype=np.bool_).reshape(-1, 2),
    )


def pack_clip_instance_tracks(
    window,
    half_size,
    max_tracks=64,
    min_pixels=4,
    min_visibility=2,
    excluded_classes=("barrier", "traffic_cone"),
):
    """Pack a variable number of 3D-box tracks into fixed-size clip arrays.

    Tracks are ranked by visible observation count and projected area.  The
    ranking only matters in the rare case that a frame window contains more
    than ``max_tracks`` identities; it keeps long, measurable tracks first.
    Returned local IDs are stable within the clip and padded with ``-1``.
    """
    if max_tracks < 1:
        raise ValueError("max_tracks must be positive")
    frames = []
    scores = defaultdict(lambda: [0, 0.0])
    for info in window:
        ids, boxes, valid = frame_instance_half_boxes(
            info, half_size=half_size, min_pixels=min_pixels,
            min_visibility=min_visibility,
            excluded_classes=excluded_classes)
        frames.append((ids, boxes, valid))
        for token, token_boxes, token_valid in zip(ids, boxes, valid):
            widths = token_boxes[:, 2] - token_boxes[:, 0]
            heights = token_boxes[:, 3] - token_boxes[:, 1]
            scores[token][0] += int(token_valid.sum())
            scores[token][1] += float(
                (widths * heights * token_valid.astype(np.float32)).sum())

    selected = sorted(
        scores,
        key=lambda token: (-scores[token][0], -scores[token][1], token),
    )[:max_tracks]
    slots = {token: index for index, token in enumerate(selected)}
    t = len(window)
    track_ids = np.full((t, max_tracks), -1, dtype=np.int64)
    boxes = np.zeros((t, max_tracks, 2, 4), dtype=np.float32)
    valid = np.zeros((t, max_tracks, 2), dtype=np.bool_)
    for frame_index, (ids, frame_boxes, frame_valid) in enumerate(frames):
        for token, token_boxes, token_valid in zip(
            ids, frame_boxes, frame_valid
        ):
            if token not in slots:
                continue
            slot = slots[token]
            track_ids[frame_index, slot] = slot
            boxes[frame_index, slot] = token_boxes
            valid[frame_index, slot] = token_valid
    return track_ids, boxes, valid
