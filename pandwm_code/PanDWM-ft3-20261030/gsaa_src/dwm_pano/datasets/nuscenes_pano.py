"""PanoNuScenesDataset — 2-pano driving dataset for OpenDWM CTSD pipeline.

Standalone torch Dataset (NOT subclass of MotionDataset) because we work off
pre-stitched pano jpg/h5/png files + a 12Hz interpolation pkl, rather than
raw nuScenes keyframe sensor data.

Produces batches matching OpenDWM's expected schema
(see `src/dwm/datasets/README.md`):
  - images:            [t, v=2] PIL RGB panos (each 448×4032, slice of 9600×600 stitched)
  - 3dbox_images:      [t, v=2] PIL, same size, colored bbox wireframes on black
  - hdmap_images:      [t, v=2] PIL, same size, colored HD-map polygon outlines
  - camera_transforms: [t, v=2, 4, 4] virtual cam2ego (identity for pano)
  - camera_intrinsics: [t, v=2, 3, 3] pseudo K (based on pano half resolution)
  - image_size:        [t, v=2, 2] (W, H) per view
  - ego_transforms:    [t, 1, 4, 4] ego2global
  - pano_camera_translations: [t, 6, 3] calibrated camera origins in ego
  - pano_camera_transforms: [t, 6, 4, 4] real sensor-to-ego c2w
  - pano_camera_intrinsics: [t, 6, 3, 3] raw nuScenes/OpenCV per-camera K
  - pano_camera_image_size: [t, 6, 2] native (W, H), normally 1600x900
  - image_description: [t, v=2] caption strings
  - fps:               (1,) float tensor
  - pts:               [t, v=2] relative timestamps (ms)

Pano half slicing (matches pano_projection.PANO_HALF_*_RANGE):
  - left half  = pano[:, 0:5376]       (covers ~202° centered on CAM_BACK)
  - right half = pano[:, 4224:9600]    (covers ~202° centered on CAM_FRONT)
  - 1152 px overlap around CAM_FRONT

Output resolution per half (configurable):
  - default 448×4032 (H=448, aspect 9:1 to preserve pano geometry)
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from dwm_pano.datasets.pano_projection import (
    CAM_ORDER, PANO_W, PANO_H,
    PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE,
)
from dwm_pano.datasets.pano_hdmap import MapCache
from dwm_pano.datasets.pano_via_stitch import (
    render_bboxes_via_stitch, render_hdmap_via_stitch,
)
from dwm_pano.calibrated_stitch import camera_records_to_arrays


# Default output resolution per pano half
DEFAULT_HALF_SIZE = (4032, 448)   # (W, H)
# (BBOX_COLORS, EDGES now live in pano_via_stitch.py; imported by the renderer)


def _render_bboxes_on_full_pano(info, pen_width=3, bbox_dir: Optional[str] = None):
    """Returns PIL 9600×600 bbox wireframe pano.
    If `bbox_dir` is set AND `{bbox_dir}/{ts}.png` exists, LOAD from disk (fast).
    Otherwise render on-the-fly via 6-cam pinhole + stitch_hardseam (slow, ~15s)."""
    if bbox_dir:
        ts = int(info["timestamp"])
        path = os.path.join(bbox_dir, f"{ts}.png")
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
    return render_bboxes_via_stitch(info, pen_width=pen_width)


def _render_hdmap_on_full_pano(info, pen_width=2, hdmap_dir: Optional[str] = None):
    """Returns PIL 9600×600 hdmap polygon pano.
    If `hdmap_dir` is set AND `{hdmap_dir}/{ts}.png` exists, LOAD from disk.
    Otherwise render on-the-fly."""
    if hdmap_dir:
        ts = int(info["timestamp"])
        path = os.path.join(hdmap_dir, f"{ts}.png")
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
    try:
        map_cache = MapCache.get(info["location"])
    except FileNotFoundError:
        return Image.new("RGB", (PANO_W, PANO_H), (0, 0, 0))
    return render_hdmap_via_stitch(info, map_cache, pen_width=pen_width)


def _slice_full_pano_to_halves(full_img, out_w, out_h):
    """Slice 9600×600 PIL to 2 halves and resize each to (out_w, out_h).
    Returns (left, right) PIL images."""
    left_full = full_img.crop((PANO_HALF_LEFT_RANGE[0], 0,
                               PANO_HALF_LEFT_RANGE[1], PANO_H))
    right_full = full_img.crop((PANO_HALF_RIGHT_RANGE[0], 0,
                                PANO_HALF_RIGHT_RANGE[1], PANO_H))
    left = left_full.resize((out_w, out_h), Image.LANCZOS)
    right = right_full.resize((out_w, out_h), Image.LANCZOS)
    return left, right


def _slice_full_pano_to_halves_nearest(full_img, out_w, out_h):
    """Same but NEAREST resize — used for bbox/hdmap overlays to keep lines sharp."""
    left_full = full_img.crop((PANO_HALF_LEFT_RANGE[0], 0,
                               PANO_HALF_LEFT_RANGE[1], PANO_H))
    right_full = full_img.crop((PANO_HALF_RIGHT_RANGE[0], 0,
                                PANO_HALF_RIGHT_RANGE[1], PANO_H))
    left = left_full.resize((out_w, out_h), Image.NEAREST)
    right = right_full.resize((out_w, out_h), Image.NEAREST)
    return left, right


def _make_virtual_pano_K(half_w, half_h):
    """Pseudo K for a pano half. Values not physically pinhole-correct; implicit
    conditioning just embeds these numbers as 'pano-half signature'."""
    fx = float(half_w) / 2.0
    fy = float(half_h) / 2.0
    cx = float(half_w) / 2.0
    cy = float(half_h) / 2.0
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def _make_virtual_pano_extrinsics(half_idx: int):
    """Virtual cam2ego 4x4 for each pano half.
    Half 0 (left, covers CAM_BACK..CAM_FRONT_LEFT) — cam pose = ego origin looking
      towards the CENTER of left half's Voronoi range (roughly ~-80° from +X_ego).
    Half 1 (right, covers CAM_FRONT..CAM_BACK_RIGHT) — centered at ~+80° from +X.
    For simplicity and stability: virtual cam at ego origin, rotation = yaw only
    toward half center. Half center angle in pano = (half_range_center - PANO_W/2) / PANO_W * 2π.

    Left half center (TX=2688): angle ≈ -80.4°
    Right half center (TX=6912): angle ≈ +80.4°
    """
    if half_idx == 0:
        center_tx = (PANO_HALF_LEFT_RANGE[0] + PANO_HALF_LEFT_RANGE[1]) / 2.0
    else:
        center_tx = (PANO_HALF_RIGHT_RANGE[0] + PANO_HALF_RIGHT_RANGE[1]) / 2.0
    ang_rad = (center_tx - PANO_W / 2.0) / PANO_W * 2.0 * np.pi
    # Rotate around z by -ang_rad (camera +z should point to the half's center
    # direction in ego frame; camera frame +z is forward)
    cos, sin = np.cos(ang_rad), np.sin(ang_rad)
    T = np.eye(4, dtype=np.float32)
    # Rotate such that camera +z (forward-looking in cam frame) points to angle ang_rad in ego
    # In ego frame, direction at angle ang_rad (measured from +X CCW toward +Y in pano
    # coord) is approximately (cos(ang_rad)*X + sin(ang_rad)*(-Y)) where pano +x = ego -y.
    # To keep things simple and deterministic: just encode a yaw rotation.
    T[:3, :3] = np.array([
        [ cos, -sin, 0.0],
        [ sin,  cos, 0.0],
        [ 0.0,  0.0, 1.0],
    ], dtype=np.float32)
    return T


# -----------------------------------------------------------------------------
# Main Dataset class
# -----------------------------------------------------------------------------
def _load_token_to_scene(meta_dir: str) -> dict:
    """Build sample_token → scene_token map from raw nuScenes JSON tables.
    `meta_dir` should be the v1.0-trainval directory containing sample.json.
    """
    import json
    with open(os.path.join(meta_dir, "sample.json"), "r") as f:
        samples = json.load(f)
    return {s["token"]: s["scene_token"] for s in samples}


def _group_infos_by_scene(infos, token2scene: Optional[dict] = None,
                          strict: bool = True):
    """Group pkl infos by **official nuScenes scene_token**.

    Strategy (verified 100% recovery on 12Hz ASAP-interp pkls):
      - 32-char tokens are real nuScenes sample_tokens → direct lookup
      - 33-char tokens are synthetic ASAP-interp tokens of form
        `<source_keyframe_token>+suffix`; the first 32 chars are the parent
        sample_token. So `token[:32]` always lands in `token2scene`.

    `strict=True` (default): raise on any unrecovered token. This is the
    safe default — silent fallback to `description` would mask metadata
    misconfiguration and silently degrade clip quality (description is
    not unique per scene; the train pkl has 2 colliding pairs).

    `strict=False`: try `description` then `(location, 20s_bucket)` as
    fallback. Only use this for legacy pkls without official metadata.
    """
    by_scene = {}
    unrecovered = []
    for info in infos:
        s = info.get("scene_token")
        if s is None and token2scene is not None:
            tok = info["token"]
            s = token2scene.get(tok) or token2scene.get(tok[:32])
        if s is None:
            if strict:
                unrecovered.append(info["token"])
                continue
            s = info.get("description")
            if s is None:
                loc = info.get("location", "unknown")
                bucket = int(info["timestamp"]) // 20_000_000
                s = f"{loc}_{bucket}"
        # Persist the recovered scene_token back into the info dict so
        # downstream code (e.g. clips[idx][0]["scene_token"] in preview
        # debug CSV / Phase 1.6 A4 held-out validation) can access it
        # directly. Without this, recovered tokens live only as by_scene
        # keys and get lost in `_enumerate_pano_clips`.
        info["scene_token"] = s
        by_scene.setdefault(s, []).append(info)

    if strict and unrecovered:
        raise ValueError(
            f"_group_infos_by_scene(strict=True): {len(unrecovered)} tokens "
            f"failed to map to a scene_token. First 3: {unrecovered[:3]}. "
            f"Either pass a valid `token2scene` mapping built from official "
            f"v1.0-trainval/sample.json, or use strict=False (NOT recommended "
            f"for training)."
        )
    for s in by_scene:
        by_scene[s].sort(key=lambda i: i["timestamp"])
    return by_scene


def _enumerate_pano_clips(scene_infos, sequence_length, fps_stride_tuples,
                          emit_failures=False, bad_timestamps_us=None):
    """Timestamp-based multi-fps clip enumeration aligned with official
    OpenDWM `MotionDataset.enumerate_segments` (fps>0 branch, see
    `src/dwm/datasets/nuscenes.py:270-330`).

    Algorithm (per (fps, stride) tuple, per scene):
      1. anchor t0 (seconds) starts at scene's first frame timestamp,
         advances by `stride` seconds, until `t0 + sequence_duration <=
         scene_end_ts`, where `sequence_duration = sequence_length / fps`.
      2. target_times[i] = t0 + i / fps   (in seconds; converted to us)
      3. for each target, find_nearest pkl frame in the scene's timestamp
         list (bisect, tie-break left like official find_nearest).
      4. validate the clip:
         - max_time_error <= 0.5e6 / fps  (microseconds; tolerance ↑ as fps↓)
         - indices strictly monotonic increasing
         - len(set(indices)) == sequence_length  (no duplicate frames)
      5. yield (clip_info_list, audit_meta) for valid clips.

    When `emit_failures=True`, also yields `(None, audit_meta_with_reason)`
    for every rejected anchor; lets the audit script count failure modes
    without re-running enumeration.

    Why timestamp-based and NOT index-based round(native_fps/target_fps) skip:
      native_fps=12, target_fps=5: round(12/5)=2 → effective 6Hz, NOT 5Hz.
      target_fps=10: round(12/10)=1 → effective 12Hz, NOT 10Hz.
      Only target_fps=2 works correctly (round(12/2)=6 → 2Hz exact).
      Index-based skip silently aliases mid-fps targets to wrong frequencies.
      Timestamp-based with find_nearest + tolerance respects the actual target
      fps regardless of pkl native rate.

    Audit metadata per clip:
      - fps:                actual fps the clip was sampled at
      - stride_s:           stride for this anchor walk
      - max_time_error_us:  max abs(target_us - nearest_actual_us) over clip frames
      - actual_duration_us: timestamps[indices[-1]] - timestamps[indices[0]]
      - scene_token, indices, valid, reason (latter only when invalid)
    """
    import bisect

    # Bad-timestamp blacklist: any clip whose actual sampled frames
    # (sdl[indices[*]]["timestamp"]) intersects this set is rejected
    # with reason="bad_timestamp_in_window". Filtering uses len(indices)
    # (== sequence_length post-validation) — no hardcoded window length.
    bad_set = set(int(ts) for ts in (bad_timestamps_us or []))

    for fps, stride in fps_stride_tuples:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0 seconds, got {stride}")

        sequence_duration_s = sequence_length / float(fps)
        tolerance_us = 500_000.0 / float(fps)

        for scene_token, sdl in scene_infos.items():
            n = len(sdl)
            if n < 2:
                if emit_failures:
                    yield (None, {
                        "fps": float(fps), "stride_s": float(stride),
                        "scene_token": scene_token, "valid": False,
                        "reason": "scene_too_short_n<2",
                    })
                continue

            timestamps_us = [int(i["timestamp"]) for i in sdl]
            scene_start_s = timestamps_us[0] / 1e6
            scene_end_s = timestamps_us[-1] / 1e6
            scene_span_s = scene_end_s - scene_start_s

            if scene_span_s < sequence_duration_s:
                if emit_failures:
                    yield (None, {
                        "fps": float(fps), "stride_s": float(stride),
                        "scene_token": scene_token, "valid": False,
                        "reason": "scene_span_lt_sequence_duration",
                        "scene_span_s": scene_span_s,
                        "sequence_duration_s": sequence_duration_s,
                    })
                continue

            last_valid_anchor_s = scene_end_s - sequence_duration_s
            t = scene_start_s
            while t <= last_valid_anchor_s + 1e-9:
                target_us = [
                    int(round((t + i / fps) * 1e6))
                    for i in range(sequence_length)
                ]

                # find_nearest indices via bisect (matches dwm.datasets.common.find_nearest:
                # tie-break left, clamp to valid range)
                indices = []
                for tgt in target_us:
                    idx = bisect.bisect_left(timestamps_us, tgt)
                    if idx == 0:
                        pass
                    elif idx >= len(timestamps_us):
                        idx = len(timestamps_us) - 1
                    else:
                        if (tgt - timestamps_us[idx - 1]) <= (timestamps_us[idx] - tgt):
                            idx -= 1
                    indices.append(idx)

                errors_us = [
                    abs(timestamps_us[idx] - tgt)
                    for idx, tgt in zip(indices, target_us)
                ]
                max_err_us = max(errors_us)
                actual_duration_us = (
                    timestamps_us[indices[-1]] - timestamps_us[indices[0]]
                )
                monotonic = all(
                    indices[i] < indices[i + 1]
                    for i in range(len(indices) - 1))
                no_dup = len(set(indices)) == sequence_length

                base_meta = {
                    "fps": float(fps),
                    "stride_s": float(stride),
                    "scene_token": scene_token,
                    "indices": list(indices),
                    "max_time_error_us": int(max_err_us),
                    "actual_duration_us": int(actual_duration_us),
                }

                if max_err_us <= tolerance_us and monotonic and no_dup:
                    # Bad-timestamp filter: walks all len(indices) sampled
                    # frame ts (NOT a hardcoded 16); reject if any in bad_set.
                    clip_ts = [int(timestamps_us[i]) for i in indices]
                    if bad_set and any(ts in bad_set for ts in clip_ts):
                        if emit_failures:
                            base_meta["valid"] = False
                            base_meta["reason"] = "bad_timestamp_in_window"
                            base_meta["bad_ts_hit"] = [
                                ts for ts in clip_ts if ts in bad_set]
                            yield (None, base_meta)
                    else:
                        base_meta["valid"] = True
                        yield ([sdl[i] for i in indices], base_meta)
                elif emit_failures:
                    base_meta["valid"] = False
                    if max_err_us > tolerance_us:
                        base_meta["reason"] = "max_time_error_exceeds_tolerance"
                    elif not monotonic:
                        base_meta["reason"] = "indices_not_monotonic"
                    elif not no_dup:
                        base_meta["reason"] = "indices_duplicated"
                    else:
                        base_meta["reason"] = "unknown"
                    yield (None, base_meta)

                t += stride


class PanoNuScenesDataset(torch.utils.data.Dataset):
    """2-pano dataset for OpenDWM CTSD. Reads pre-stitched pano jpg + 12Hz pkl.

    Args:
        pkl_path: path to interpolated 12Hz pkl (train or val).
        rgb_dir: directory with {timestamp}.jpg pano RGBs (full 9600×600).
        depth_dir, seg_dir: optional, same keyed-by-timestamp layout.
        sequence_length: frames per clip (e.g., 9).
        fps_stride_tuples: iterable of (fps, stride_seconds). Default [(2.0, 2.0)]:
            sample at 2 Hz (every 6th frame of 12Hz), 2s stride between clips.
        half_size: (W, H) of each pano half after resize.
        split: label for debugging (e.g., 'train' / 'val').
    """
    def __init__(
        self,
        pkl_path: str,
        rgb_dir: str,
        depth_dir: Optional[str] = None,
        seg_dir: Optional[str] = None,
        bbox_dir: Optional[str] = None,       # NEW: if set, read precomputed PNG
        hdmap_dir: Optional[str] = None,      # NEW: if set, read precomputed PNG
        sequence_length: int = 9,
        fps_stride_tuples: list = ((2.0, 2.0),),
        half_size: tuple = DEFAULT_HALF_SIZE,
        split: str = "unknown",
        enable_3dbox: bool = True,
        enable_hdmap: bool = True,
        enable_camera_transforms: bool = True,
        enable_ego_transforms: bool = True,
        enable_description: bool = True,
        hdmap_pen_width: int = 2,     # matches scripts/precompute_bbox_hdmap_pano.py default
        bbox_pen_width: int = 3,      # matches precompute default
        nuscenes_meta_dir: Optional[str] = "/path/to/pandwm_root/nuscenes/v1.0-trainval",
        stub_key_data_dict: Optional[dict] = None,
        output_fps: Optional[float] = None,
        bad_timestamps_us: Optional[list] = None,
        emit_instance_tracks: bool = False,
        max_instance_tracks: int = 64,
        instance_track_min_pixels: int = 4,
        instance_track_min_visibility: int = 2,
        instance_track_excluded_classes: tuple = ("barrier", "traffic_cone"),
    ):
        # output_fps: scalar fps reported in result["fps"] (used by model
        # added_time_ids). If None, derive from fps_stride_tuples[0][0].
        # Avoid hardcoding 2.0 — must match the actual sampling rate of the
        # clip (e.g. 12.0 for our 12Hz DFoT clips). Wrong fps silently hurts
        # generation quality via wrong condition embedding.
        # stub_key_data_dict: per OpenDWM convention, inject extra fields the
        # pipeline expects but this dataset doesn't naturally produce (e.g.
        # crossview_mask = [[1,1],[1,1]] for 2-pano).
        with open(pkl_path, "rb") as f:
            pkl = pickle.load(f)
        self.infos = pkl["infos"]
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.seg_dir = seg_dir
        self.bbox_dir = bbox_dir
        self.hdmap_dir = hdmap_dir
        self.sequence_length = sequence_length
        self.half_size = half_size
        self.split = split
        self.enable_3dbox = enable_3dbox
        self.enable_hdmap = enable_hdmap
        self.enable_camera_transforms = enable_camera_transforms
        self.enable_ego_transforms = enable_ego_transforms
        self.enable_description = enable_description
        self.hdmap_pen_width = hdmap_pen_width
        self.bbox_pen_width = bbox_pen_width
        self.stub_key_data_dict = stub_key_data_dict
        self.output_fps = (
            float(output_fps) if output_fps is not None
            else float(fps_stride_tuples[0][0])
        )
        self.emit_instance_tracks = bool(emit_instance_tracks)
        self.max_instance_tracks = int(max_instance_tracks)
        self.instance_track_min_pixels = int(instance_track_min_pixels)
        self.instance_track_min_visibility = int(instance_track_min_visibility)
        self.instance_track_excluded_classes = tuple(
            instance_track_excluded_classes or ())
        if self.emit_instance_tracks and self.max_instance_tracks < 1:
            raise ValueError("max_instance_tracks must be positive")

        if nuscenes_meta_dir is None or not os.path.exists(
                os.path.join(nuscenes_meta_dir, "sample.json")):
            raise FileNotFoundError(
                f"nuscenes_meta_dir must point to an extracted v1.0-trainval/ "
                f"directory containing sample.json (for scene_token recovery). "
                f"Got: {nuscenes_meta_dir!r}. Download "
                f"v1.0-trainval_meta.tgz and extract."
            )
        token2scene = _load_token_to_scene(nuscenes_meta_dir)
        scene_infos = _group_infos_by_scene(
            self.infos, token2scene=token2scene, strict=True)
        # Multi-fps timestamp-based enumeration; each clip carries its own
        # `fps` (in audit meta) for per-clip __getitem__ exposure to the model.
        _pairs = list(_enumerate_pano_clips(
            scene_infos, sequence_length, fps_stride_tuples,
            bad_timestamps_us=bad_timestamps_us))
        self.clips = [p[0] for p in _pairs]
        self.clip_audit = [p[1] for p in _pairs]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int):
        window = self.clips[idx]
        t = self.sequence_length
        v = 2
        half_w, half_h = self.half_size

        # Allocate result
        result = {}
        images = []
        bboxes = []
        hdmaps = []
        camera_transforms = []
        camera_intrinsics = []
        image_size = []
        ego_transforms = []
        pano_camera_translations = []
        pano_camera_transforms = []
        pano_camera_intrinsics = []
        pano_camera_image_size = []
        pts = []
        frame_timestamps = []
        descriptions = []

        # Use first frame's timestamp as anchor for pts
        t0 = window[0]["timestamp"]

        for info in window:
            ts = int(info["timestamp"])
            # --- images ---
            rgb_path = os.path.join(self.rgb_dir, f"{ts}.jpg")
            if not os.path.exists(rgb_path):
                raise FileNotFoundError(
                    f"RGB pano missing: {rgb_path}. /local is per-node — if "
                    f"training runs on a node without RGB, the previous silent "
                    f"black-image fallback would let loss converge to 0 on "
                    f"black 'GT'. Pin srun --nodelist=<node-with-rgb> or "
                    f"replicate rgb_dir to a shared filesystem.")
            rgb_full = Image.open(rgb_path).convert("RGB")
            left_rgb, right_rgb = _slice_full_pano_to_halves(rgb_full, half_w, half_h)
            images.append([left_rgb, right_rgb])

            # --- 3dbox_images ---
            if self.enable_3dbox:
                bbox_full = _render_bboxes_on_full_pano(
                    info, pen_width=self.bbox_pen_width, bbox_dir=self.bbox_dir)
                lb, rb = _slice_full_pano_to_halves_nearest(bbox_full, half_w, half_h)
                bboxes.append([lb, rb])

            # --- hdmap_images ---
            if self.enable_hdmap:
                hdmap_full = _render_hdmap_on_full_pano(
                    info, pen_width=self.hdmap_pen_width, hdmap_dir=self.hdmap_dir)
                lh, rh = _slice_full_pano_to_halves_nearest(hdmap_full, half_w, half_h)
                hdmaps.append([lh, rh])

            # --- camera_transforms / camera_intrinsics ---
            if self.enable_camera_transforms:
                K = _make_virtual_pano_K(half_w, half_h)
                E_left = _make_virtual_pano_extrinsics(0)
                E_right = _make_virtual_pano_extrinsics(1)
                camera_transforms.append(np.stack([E_left, E_right]))
                camera_intrinsics.append(np.stack([K, K]))
                image_size.append(np.array([[half_w, half_h], [half_w, half_h]],
                                            dtype=np.int64))

            # --- ego_transforms ---
            if self.enable_ego_transforms:
                ego_r = Quaternion(info["ego2global_rotation"]).rotation_matrix
                ego_t = np.array(info["ego2global_translation"])
                T_e2g = np.eye(4, dtype=np.float32)
                T_e2g[:3, :3] = ego_r
                T_e2g[:3, 3] = ego_t
                ego_transforms.append(T_e2g[None, :, :])  # (1, 4, 4), 1 sensor

            # Full real-camera calibration in the exact hard-seam camera
            # order.  Keep the legacy translations key for existing decoder
            # configs; the new c2w/K keys let a calibrated target render and
            # differentiable OneBEV stitch share one authoritative packing.
            real_c2w, real_k, real_image_size = camera_records_to_arrays(
                info["cams"], camera_order=CAM_ORDER)
            pano_camera_transforms.append(real_c2w)
            pano_camera_intrinsics.append(real_k)
            pano_camera_image_size.append(real_image_size)
            pano_camera_translations.append(real_c2w[:, :3, 3])

            # --- pts ---
            dt_ms = (info["timestamp"] - t0) // 1000
            pts.append([dt_ms, dt_ms])
            frame_timestamps.append(ts)

            # --- description ---
            if self.enable_description:
                # Scene-level text from pkl metadata. Upgrade later to DriveMLM caption.
                time = "day"
                weather = "clear"
                environment = info.get("description", "urban road")
                desc_str = f"{time}, {weather}, {environment}"
                descriptions.append([desc_str, desc_str])

        # Assemble result dict
        result["images"] = images
        if self.enable_3dbox:
            result["3dbox_images"] = bboxes
        if self.enable_hdmap:
            result["hdmap_images"] = hdmaps
        if self.enable_camera_transforms:
            result["camera_transforms"] = torch.from_numpy(np.stack(camera_transforms))
            result["camera_intrinsics"] = torch.from_numpy(np.stack(camera_intrinsics))
            result["image_size"] = torch.from_numpy(np.stack(image_size))
        if self.enable_ego_transforms:
            result["ego_transforms"] = torch.from_numpy(np.stack(ego_transforms))
        result["pano_camera_translations"] = torch.from_numpy(
            np.stack(pano_camera_translations))
        result["pano_camera_transforms"] = torch.from_numpy(
            np.stack(pano_camera_transforms))
        result["pano_camera_intrinsics"] = torch.from_numpy(
            np.stack(pano_camera_intrinsics))
        result["pano_camera_image_size"] = torch.from_numpy(
            np.stack(pano_camera_image_size))
        if self.enable_description:
            result["image_description"] = descriptions
        # scalar (shape ()) per sample → DataLoader collate → (B,) per batch.
        # Pipeline expects (B,)-shape: ctsd.py:329 does fps.unsqueeze(-1)×3 + repeat
        # which assumes (B,) input. Don't wrap in [list] (would give (1,) → (B,1)).
        # Per-clip fps from audit metadata (multi-fps support); falls back to
        # self.output_fps for clips with no audit (defensive).
        _clip_fps = self.clip_audit[idx]["fps"] if hasattr(self, "clip_audit") \
            and idx < len(self.clip_audit) else self.output_fps
        result["fps"] = torch.tensor(_clip_fps, dtype=torch.float32)
        result["pts"] = torch.tensor(pts, dtype=torch.float32)
        # Exact frame file-name timestamps (pts floors us->ms, irreversible).
        result["frame_timestamps_us"] = torch.tensor(
            frame_timestamps, dtype=torch.int64)
        # Identification fields for preview/debug:
        #   dataset_index: __getitem__ argument. Under torch.utils.data.Subset,
        #     this is the ORIGINAL index into the base PanoNuScenesDataset
        #     clip enumeration (Subset(ds, [0,1,50])[2] calls ds.__getitem__(50)).
        #   anchor_timestamp_us: window[0]["timestamp"] of the clip — fully
        #     unique per clip, survives any sub-batching, useful in preview CSV.
        result["dataset_index"] = torch.tensor(idx, dtype=torch.int64)
        result["anchor_timestamp_us"] = torch.tensor(
            int(self.clips[idx][0]["timestamp"]), dtype=torch.int64)
        # scene_token (string) — set by `_group_infos_by_scene` and persisted
        # back into info dict. Critical for held-out scene verification (Phase
        # 1.6 A4) and preview CSV. Default-collated to List[str] of len B.
        result["scene_token"] = self.clips[idx][0].get("scene_token", "")
        # split (string: "train" / "val") — disambiguates train pkl vs val pkl
        # within a ConcatDataset preview pool (A5.2+). Without this, two clips
        # from different splits with the same dataset_index would be visually
        # ambiguous in preview_debug.csv.
        result["split"] = self.split
        if self.emit_instance_tracks:
            from dwm_pano.datasets.instance_tracks import \
                pack_clip_instance_tracks
            track_ids, track_boxes, track_valid = pack_clip_instance_tracks(
                window,
                half_size=self.half_size,
                max_tracks=self.max_instance_tracks,
                min_pixels=self.instance_track_min_pixels,
                min_visibility=self.instance_track_min_visibility,
                excluded_classes=self.instance_track_excluded_classes,
            )
            result["instance_track_ids"] = torch.from_numpy(track_ids)
            result["instance_track_boxes"] = torch.from_numpy(track_boxes)
            result["instance_track_valid"] = torch.from_numpy(track_valid)
        if self.stub_key_data_dict is not None:
            from dwm.datasets.common import add_stub_key_data
            add_stub_key_data(self.stub_key_data_dict, result)
        return result


if __name__ == "__main__":
    # Smoke test both train and val
    PKL_BASE = "/path/to/pandwm_root/workspace/OpenDWM/data/pickle"
    TEST_CASES = [
        ("val", f"{PKL_BASE}/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl",
         "/local/data/val/rgb"),
        ("train", f"{PKL_BASE}/nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl",
         "/path/to/pandwm_root/nuscenes-360/train"),
    ]
    for split, pkl, rgb in TEST_CASES:
        if not os.path.exists(pkl):
            print(f"[{split}] pkl missing: {pkl} — skip"); continue
        if not os.path.exists(rgb):
            print(f"[{split}] rgb dir missing: {rgb} — skip"); continue
        ds = PanoNuScenesDataset(
            pkl_path=pkl, rgb_dir=rgb,
            sequence_length=9,
            fps_stride_tuples=[(2.0, 2.0)],
            half_size=(2016, 224),
            split=split,
        )
        print(f"\n[{split}] len(ds) = {len(ds)}")
        if len(ds) == 0:
            continue
        sample = ds[0]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {tuple(v.shape)} {v.dtype}")
            elif isinstance(v, list):
                n = len(v)
                m = len(v[0]) if v else 0
                el = v[0][0] if v and v[0] else None
                extra = f" (PIL size: {el.size})" if hasattr(el, "size") else ""
                print(f"  {k}: list[{n} × {m}] {type(el).__name__}{extra}")
