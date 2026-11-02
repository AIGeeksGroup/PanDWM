"""PanoNuScenesDataset plus per-frame 3D boxes in ego coordinates.

Extra fields (fixed shapes so default collation works):
  box_centers     [T, M, 3]    float32  box centre, ego frame of that frame
  box_rotations   [T, M, 3, 3] float32  box-to-ego rotation
  box_half_sizes  [T, M, 3]    float32  half extents along the box axes
  box_valid       [T, M]       bool
  box_track       [T, M]       int64    clip-local track id (-1 = padding)

Only classes listed in ``dynamic_classes`` are emitted: they decide which
Gaussians the 4D decoder treats as dynamic.  Boxes come from the same pkl
(``gt_boxes`` in LiDAR coordinates, mmdet3d origin (0.5, 0.5, 0.5), and
``lidar2ego``) used for the rendered box conditions.

2026-10-24 (round-2 fine-tune, PanDWM-gs-ft2-20261024/GS_FT2_PLAN_20261024.md), both off by default:
  pose_source "camera_time"   ego_transforms = the mean ego pose of the six cameras at their exposure times
                              (recon.targets.camera_time_pose) instead of the pkl's frame-level pose, which for the
                              12 Hz interpolated frames is the pose of a LiDAR sweep up to 250 ms before the cameras
                              (report section 44); the boxes follow through the adjusted lidar2ego of
                              recon.targets.camera_time_infos, so their global placement is unchanged;
  box_reading "mmdet3d_v0x"   gt_boxes read as the pkl stores them, (x, y, z, w, l, h, -yaw - pi/2): length along the
                              heading.  "legacy" reads (x, y, z, dx, dy, dz, yaw), which mirrors diagonal boxes.
"""

from __future__ import annotations

import numpy as np
import torch
from pyquaternion import Quaternion

from dwm_pano.datasets.nuscenes_pano import PanoNuScenesDataset

DEFAULT_DYNAMIC_CLASSES = (
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle")


def frame_boxes_in_ego(info, dynamic_classes, box_reading="legacy"):
    """(tokens, centers (N,3), rotations (N,3,3), half sizes (N,3)) in ego."""
    if box_reading not in ("legacy", "mmdet3d_v0x"):
        raise ValueError(f"box_reading {box_reading!r}")
    boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float64)
    if boxes.size == 0:
        return [], np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3))
    boxes = boxes.reshape(boxes.shape[0], -1)[:, :7]
    tokens = list(info.get("gt_box_ids", range(len(boxes))))
    names = list(info.get("gt_names", [""] * len(boxes)))
    if len(tokens) != len(boxes) or len(names) != len(boxes):
        raise ValueError("gt_box_ids / gt_names must align with gt_boxes")
    keep = [i for i, name in enumerate(names) if str(name) in dynamic_classes]
    if not keep:
        return [], np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3))
    boxes = boxes[keep]
    tokens = [tokens[i] for i in keep]
    l2e_rotation = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    l2e_translation = np.asarray(info["lidar2ego_translation"], dtype=np.float64)
    centers = boxes[:, :3] @ l2e_rotation.T + l2e_translation
    if box_reading == "mmdet3d_v0x":
        heading, half = -boxes[:, 6] - np.pi / 2, boxes[:, [4, 3, 5]] / 2.0
    else:
        heading, half = boxes[:, 6], boxes[:, 3:6] / 2.0
    cos, sin = np.cos(heading), np.sin(heading)
    yaw = np.zeros((boxes.shape[0], 3, 3))
    yaw[:, 0, 0], yaw[:, 0, 1] = cos, -sin
    yaw[:, 1, 0], yaw[:, 1, 1] = sin, cos
    yaw[:, 2, 2] = 1.0
    rotations = l2e_rotation[None] @ yaw
    return tokens, centers, rotations, half


class PanoNuScenesBoxDataset(PanoNuScenesDataset):
    def __init__(self, *args, max_boxes: int = 128,
                 dynamic_classes=DEFAULT_DYNAMIC_CLASSES, pose_source: str = "pkl",
                 box_reading: str = "legacy", **kwargs):
        super().__init__(*args, **kwargs)
        if pose_source not in ("pkl", "camera_time"):
            raise ValueError(f"pose_source {pose_source!r}")
        if box_reading not in ("legacy", "mmdet3d_v0x"):
            raise ValueError(f"box_reading {box_reading!r}")
        self.pose_source, self.box_reading = pose_source, box_reading
        if max_boxes < 1:
            raise ValueError("max_boxes must be positive")
        self.max_boxes = int(max_boxes)
        self.dynamic_classes = set(dynamic_classes)

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        window = self.clips[idx]
        if self.pose_source == "camera_time":
            from dwm_pano.recon.targets import camera_time_infos
            window = camera_time_infos(window)
            if "ego_transforms" in result:
                e2g = np.zeros((len(window), 1, 4, 4), dtype=np.float32)
                for t, info in enumerate(window):
                    e2g[t, 0, :3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
                    e2g[t, 0, :3, 3] = np.asarray(info["ego2global_translation"], dtype=np.float64)
                    e2g[t, 0, 3, 3] = 1.0
                old = result["ego_transforms"]
                if tuple(old.shape) != e2g.shape:
                    raise RuntimeError(f"ego_transforms {tuple(old.shape)} vs {e2g.shape}")
                result["ego_transforms"] = torch.from_numpy(e2g).to(old.dtype) if torch.is_tensor(old) else e2g
        frames = len(window)
        m = self.max_boxes
        centers = np.zeros((frames, m, 3), dtype=np.float32)
        rotations = np.tile(np.eye(3, dtype=np.float32), (frames, m, 1, 1))
        half = np.zeros((frames, m, 3), dtype=np.float32)
        valid = np.zeros((frames, m), dtype=np.bool_)
        track = np.full((frames, m), -1, dtype=np.int64)
        track_ids: dict = {}
        for t, info in enumerate(window):
            tokens, c, r, h = frame_boxes_in_ego(info, self.dynamic_classes, self.box_reading)
            # Nearest boxes first if a frame exceeds the cap: they dominate
            # the panorama and the static/dynamic split near the ego vehicle.
            order = np.argsort(np.linalg.norm(c[:, :2], axis=-1)) if len(tokens) else []
            for slot, i in enumerate(list(order)[:m]):
                centers[t, slot], rotations[t, slot], half[t, slot] = c[i], r[i], h[i]
                valid[t, slot] = True
                track[t, slot] = track_ids.setdefault(tokens[i], len(track_ids))
        result["box_centers"] = torch.from_numpy(centers)
        result["box_rotations"] = torch.from_numpy(rotations)
        result["box_half_sizes"] = torch.from_numpy(half)
        result["box_valid"] = torch.from_numpy(valid)
        result["box_track"] = torch.from_numpy(track)
        return result
