"""Project GT 3D boxes to 2D boxes on the V=1 stitched panorama.

Motivation: `gt_box_ids` in the interp pkl are persistent nuScenes instance
tokens, but nothing downstream uses them -- `render_bboxes_via_stitch` colours
wireframes by *class*, so the model is never told which box is which object.
To measure whether the same object keeps its appearance across frames we first
need, per frame, that object's pixel region on the same V=1 canvas the FVD
metric already scores.

The V=1 geometry mirrors `fvd_pano.PanoPatchFrechetVideoDistance`:
9600-wide pano -> two 2304-wide halves -> stitched to 4114 by dropping the
494-px overlap.
"""
import numpy as np
from pyquaternion import Quaternion

from dwm_pano.datasets.pano_projection import (
    PANO_H, PANO_W, project_ego_points_to_pano)
from dwm_pano.datasets.pano_via_stitch import _ego_corners_from_boxes

PANO_W_V1 = 4114
_HALF_SRC_W = 5376.0
_HALF_PX_W = 2304.0
_RIGHT_START = 4224.0
_OVERLAP = 494.0
_SCALE = _HALF_PX_W / _HALF_SRC_W          # 9600-axis -> half-pixel axis


def pano_column_to_v1(tx):
    """Map a column on the 9600 pano axis to the 4114 stitched axis.

    Left half [0, 5376) lands at V=1 [0, 2304); the right half contributes its
    columns [494, 2304) to V=1 [2304, 4114). The two branches agree to within
    a fraction of a pixel at the boundary, which is what makes the seam
    continuous.
    """
    tx = np.asarray(tx, dtype=np.float64)
    left = tx * _SCALE
    right = (_HALF_PX_W - _OVERLAP) + (tx - _RIGHT_START) * _SCALE
    return np.where(tx < _HALF_SRC_W, left, right)


def frame_instance_boxes(info, min_pixels=8):
    """2D boxes on the V=1 canvas for every GT instance in one frame.

    Returns:
        ``(ids, boxes, skipped_wrap)`` where ``ids`` is a list of instance
        token strings, ``boxes`` is ``(N, 4)`` float64 ``[x0, y0, x1, y1]`` on
        the 4114x600 canvas, and ``skipped_wrap`` counts boxes dropped for
        straddling the wrap seam (their pixels are split across both canvas
        edges, so a single rectangle cannot describe them).
    """
    gt_boxes = np.asarray(info["gt_boxes"], dtype=np.float64)
    if gt_boxes.size == 0:
        return [], np.zeros((0, 4)), 0
    box_ids = list(info["gt_box_ids"])
    corners = _ego_corners_from_boxes(
        gt_boxes,
        Quaternion(info["lidar2ego_rotation"]),
        np.asarray(info["lidar2ego_translation"]))            # (N, 8, 3)

    ids, boxes, skipped = [], [], 0
    for i in range(corners.shape[0]):
        tx, ty, valid = project_ego_points_to_pano(corners[i])
        if valid.sum() < 4:            # too few corners inside the vertical FOV
            continue
        tx, ty = tx[valid], ty[valid]
        # A box straddling the wrap has corners at both ends of the 9600 axis.
        if tx.max() - tx.min() > PANO_W / 2.0:
            skipped += 1
            continue
        x0, x1 = pano_column_to_v1(tx.min()), pano_column_to_v1(tx.max())
        y0, y1 = float(ty.min()), float(ty.max())
        if x1 - x0 < min_pixels or y1 - y0 < min_pixels:
            continue
        ids.append(box_ids[i])
        boxes.append([x0, y0, x1, y1])
    return ids, np.asarray(boxes, dtype=np.float64).reshape(-1, 4), skipped
