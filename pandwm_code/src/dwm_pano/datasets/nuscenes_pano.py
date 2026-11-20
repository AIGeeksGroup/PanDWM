"""PanoNuScenesDataset: 2-pano driving dataset for the OpenDWM CTSD pipeline.

A standalone torch Dataset (not a MotionDataset subclass) that works off pre-stitched pano jpg/png files and a 12 Hz
interpolation pkl instead of raw nuScenes sensor data. Each clip returns the OpenDWM batch schema:
  - images, 3dbox_images, hdmap_images: [t, v=2] PIL panos (each half 448x4032, a slice of the 9600x600 pano)
  - camera_transforms [t, 2, 4, 4]: virtual cam2ego (yaw only); camera_intrinsics [t, 2, 3, 3]: pseudo K
  - image_size [t, 2, 2] (W, H); ego_transforms [t, 1, 4, 4] ego2global
  - pano_camera_translations [t, 6, 3], pano_camera_transforms [t, 6, 4, 4], pano_camera_intrinsics [t, 6, 3, 3],
    pano_camera_image_size [t, 6, 2]: the real six-camera calibration (sensor-to-ego, OpenCV K, native size)
  - image_description [t, 2] caption strings; fps scalar; pts [t, 2] relative timestamps in ms

Pano half slicing (matches pano_projection.PANO_HALF_*_RANGE): left half = pano[:, 0:5376] (about 202 degrees
centered on CAM_BACK), right half = pano[:, 4224:9600] (centered on CAM_FRONT), 1152 px overlap around CAM_FRONT.
The output size per half is configurable (default 448x4032).
"""
from __future__ import annotations

import json
import os
import re
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


# Default output resolution per pano half.
DEFAULT_HALF_SIZE = (4032, 448)   # (W, H)


def _render_bboxes_on_full_pano(info, pen_width=3, bbox_dir: Optional[str] = None):
    """Return the 9600x600 PIL bbox wireframe pano.

    If `bbox_dir` is set and `{bbox_dir}/{ts}.png` exists it is loaded from disk, otherwise it is rendered on the fly
    with the six pinhole cameras and the hard-seam stitch (slow).
    """
    if bbox_dir:
        ts = int(info["timestamp"])
        path = os.path.join(bbox_dir, f"{ts}.png")
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
    return render_bboxes_via_stitch(info, pen_width=pen_width)


def _render_hdmap_on_full_pano(info, pen_width=2, hdmap_dir: Optional[str] = None):
    """Return the 9600x600 PIL hdmap polygon pano, loaded from `hdmap_dir` if available, else rendered on the fly."""
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
    """Slice a 9600x600 PIL pano into two halves and resize each to (out_w, out_h). Returns (left, right)."""
    left_full = full_img.crop((PANO_HALF_LEFT_RANGE[0], 0,
                               PANO_HALF_LEFT_RANGE[1], PANO_H))
    right_full = full_img.crop((PANO_HALF_RIGHT_RANGE[0], 0,
                                PANO_HALF_RIGHT_RANGE[1], PANO_H))
    left = left_full.resize((out_w, out_h), Image.LANCZOS)
    right = right_full.resize((out_w, out_h), Image.LANCZOS)
    return left, right


def _slice_full_pano_to_halves_nearest(full_img, out_w, out_h):
    """Same as above with NEAREST resize, used for bbox/hdmap overlays to keep lines sharp."""
    left_full = full_img.crop((PANO_HALF_LEFT_RANGE[0], 0,
                               PANO_HALF_LEFT_RANGE[1], PANO_H))
    right_full = full_img.crop((PANO_HALF_RIGHT_RANGE[0], 0,
                                PANO_HALF_RIGHT_RANGE[1], PANO_H))
    left = left_full.resize((out_w, out_h), Image.NEAREST)
    right = right_full.resize((out_w, out_h), Image.NEAREST)
    return left, right


def _make_virtual_pano_K(half_w, half_h):
    """Pseudo intrinsics for a pano half. Not physically pinhole-correct; the conditioning only uses them as a per-half signature."""
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

    The virtual camera sits at the ego origin with a yaw-only rotation toward the center of the half (left half about
    -80.4 degrees from +X_ego, right half about +80.4 degrees). The center angle is
    (half_range_center - PANO_W / 2) / PANO_W * 2 * pi.
    """
    if half_idx == 0:
        center_tx = (PANO_HALF_LEFT_RANGE[0] + PANO_HALF_LEFT_RANGE[1]) / 2.0
    else:
        center_tx = (PANO_HALF_RIGHT_RANGE[0] + PANO_HALF_RIGHT_RANGE[1]) / 2.0
    ang_rad = (center_tx - PANO_W / 2.0) / PANO_W * 2.0 * np.pi
    cos, sin = np.cos(ang_rad), np.sin(ang_rad)
    T = np.eye(4, dtype=np.float32)
    # Yaw rotation about z toward the half center (pano +x corresponds to ego -y).
    T[:3, :3] = np.array([
        [ cos, -sin, 0.0],
        [ sin,  cos, 0.0],
        [ 0.0,  0.0, 1.0],
    ], dtype=np.float32)
    return T


# Main dataset class
def _load_token_to_scene(meta_dir: str) -> dict:
    """Build a sample_token to scene_token map from the raw nuScenes JSON tables (`meta_dir` is the v1.0-trainval directory containing sample.json)."""
    import json
    with open(os.path.join(meta_dir, "sample.json"), "r") as f:
        samples = json.load(f)
    return {s["token"]: s["scene_token"] for s in samples}


def _group_infos_by_scene(infos, token2scene: Optional[dict] = None,
                          strict: bool = True):
    """Group pkl infos by the official nuScenes scene_token.

    32-char tokens are real nuScenes sample tokens (direct lookup). 33-char tokens are synthetic interpolated tokens
    `<source_keyframe_token>+suffix` whose first 32 chars are the parent sample token, so `token[:32]` always lands in
    `token2scene`.

    `strict=True` (default) raises on any unrecovered token, because falling back to `description` would silently degrade
    clip quality (descriptions are not unique per scene). `strict=False` falls back to `description`, then to
    `(location, 20 s bucket)`, for legacy pkls without official metadata.
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
        # Store the recovered scene_token in the info dict so that it survives `_enumerate_pano_clips`.
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
    """Timestamp-based multi-fps clip enumeration, aligned with OpenDWM `MotionDataset.enumerate_segments`.

    Per (fps, stride) tuple and per scene:
      1. the anchor t0 (seconds) starts at the scene's first frame timestamp and advances by `stride` seconds while
         `t0 + sequence_length / fps <= scene_end_ts`;
      2. target_times[i] = t0 + i / fps (converted to microseconds);
      3. each target is matched to the nearest pkl frame of the scene (bisect, ties go left like the official
         find_nearest);
      4. the clip is valid if max_time_error <= 0.5e6 / fps microseconds, the indices are strictly increasing and
         there are `sequence_length` distinct frames;
      5. valid clips are yielded as (clip_info_list, audit_meta).

    With `emit_failures=True`, rejected anchors are also yielded as `(None, audit_meta_with_reason)`.

    Sampling by timestamp rather than by skipping round(native_fps / target_fps) frames keeps the effective rate exact
    (for a 12 Hz source, index skipping would turn 5 Hz into 6 Hz and 10 Hz into 12 Hz).

    Audit metadata per clip: fps, stride_s, max_time_error_us, actual_duration_us (last minus first timestamp),
    scene_token, indices, valid, and reason (only when invalid).
    """
    import bisect

    # Clips containing any blacklisted timestamp are rejected with reason="bad_timestamp_in_window".
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

                # Nearest frame via bisect (ties go left, indices clamped to the valid range).
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
                    # Reject clips with any blacklisted frame timestamp.
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


_NIGHT_RE = re.compile(r"\bnight\b", re.IGNORECASE)
_RAIN_RE = re.compile(r"\brain(?:y|ing)?\b", re.IGNORECASE)


def scene_time_weather(description: str):
    """(time, weather) for the text condition from an official nuScenes scene description: ("night" | "day", "rainy" | "clear")."""
    description = description or ""
    time = "night" if _NIGHT_RE.search(description) else "day"
    weather = "rainy" if _RAIN_RE.search(description) else "clear"
    return time, weather


class PanoNuScenesDataset(torch.utils.data.Dataset):
    """2-pano dataset for OpenDWM CTSD. Reads pre-stitched pano jpg files and a 12 Hz pkl.

    Args:
        pkl_path: path to the interpolated 12 Hz pkl (train or val).
        rgb_dir: directory with {timestamp}.jpg pano RGBs (full 9600x600).
        depth_dir, seg_dir: optional, same keyed-by-timestamp layout.
        sequence_length: frames per clip (e.g. 9).
        fps_stride_tuples: iterable of (fps, stride_seconds). Default [(2.0, 2.0)]:
            sample at 2 Hz (every 6th frame of 12 Hz), 2 s stride between clips.
        half_size: (W, H) of each pano half after resize.
        split: label for debugging (e.g. 'train' / 'val').
    """
    def __init__(
        self,
        pkl_path: str,
        rgb_dir: str,
        depth_dir: Optional[str] = None,
        seg_dir: Optional[str] = None,
        bbox_dir: Optional[str] = None,       # if set, read precomputed PNG
        hdmap_dir: Optional[str] = None,      # if set, read precomputed PNG
        sequence_length: int = 9,
        fps_stride_tuples: list = ((2.0, 2.0),),
        half_size: tuple = DEFAULT_HALF_SIZE,
        split: str = "unknown",
        enable_3dbox: bool = True,
        enable_hdmap: bool = True,
        enable_camera_transforms: bool = True,
        enable_ego_transforms: bool = True,
        enable_description: bool = True,
        hdmap_pen_width: int = 2,     # default of the bbox/hdmap precompute script
        bbox_pen_width: int = 3,      # default of the bbox/hdmap precompute script
        nuscenes_meta_dir: Optional[str] = "/path/to/pandwm_root/nuscenes/v1.0-trainval",
        stub_key_data_dict: Optional[dict] = None,
        output_fps: Optional[float] = None,
        bad_timestamps_us: Optional[list] = None,
        emit_instance_tracks: bool = False,
        max_instance_tracks: int = 64,
        instance_track_min_pixels: int = 4,
        instance_track_min_visibility: int = 2,
        instance_track_excluded_classes: tuple = ("barrier", "traffic_cone"),
        time_weather_source: str = "fixed",
    ):
        # output_fps: scalar fps reported in result["fps"] (used for the model's added_time_ids); if None it is
        # derived from fps_stride_tuples[0][0]. It must match the actual sampling rate of the clip.
        # stub_key_data_dict: extra fields the pipeline expects but this dataset does not produce, injected per
        # OpenDWM convention (e.g. crossview_mask = [[1,1],[1,1]] for 2-pano).
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
        # Multi-fps timestamp-based enumeration; each clip carries its own fps in the audit meta.
        _pairs = list(_enumerate_pano_clips(
            scene_infos, sequence_length, fps_stride_tuples,
            bad_timestamps_us=bad_timestamps_us))
        self.clips = [p[0] for p in _pairs]
        self.clip_audit = [p[1] for p in _pairs]
        # time_weather_source: "fixed" uses "day, clear" for every clip; "scene_description" takes the time and
        # weather from the official nuScenes scene description ("night" if it says night, "rainy" if it says rain).
        if time_weather_source not in ("fixed", "scene_description"):
            raise ValueError(f"time_weather_source {time_weather_source!r}")
        self.time_weather_source = time_weather_source
        self._scene_time_weather = {}
        if time_weather_source == "scene_description":
            with open(os.path.join(nuscenes_meta_dir, "scene.json")) as f:
                self._scene_time_weather = {
                    s["token"]: scene_time_weather(s.get("description", ""))
                    for s in json.load(f)}

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
                    f"RGB pano missing: {rgb_path}. Check rgb_dir; when it is on node-local "
                    f"storage, run on a node that holds it or copy it to a "
                    f"shared filesystem.")
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

            # Full real-camera calibration in the hard-seam camera order. The translations key is kept for
            # existing decoder configs; the c2w/K keys let a calibrated target render and the differentiable
            # stitch share one packing.
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
                # Scene-level text from pkl metadata.
                time = "day"
                weather = "clear"
                if self.time_weather_source == "scene_description":
                    time, weather = self._scene_time_weather.get(
                        info.get("scene_token"), (time, weather))
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
        # A scalar per sample, collated to (B,); the pipeline expects a (B,)-shaped fps, so do not wrap it in a list.
        # Per-clip fps comes from the audit metadata, falling back to self.output_fps.
        _clip_fps = self.clip_audit[idx]["fps"] if hasattr(self, "clip_audit") \
            and idx < len(self.clip_audit) else self.output_fps
        result["fps"] = torch.tensor(_clip_fps, dtype=torch.float32)
        result["pts"] = torch.tensor(pts, dtype=torch.float32)
        # Exact frame file-name timestamps (pts floors us to ms and cannot be inverted).
        result["frame_timestamps_us"] = torch.tensor(
            frame_timestamps, dtype=torch.int64)
        # Identification fields for preview/debug:
        #   dataset_index: the __getitem__ argument (under torch.utils.data.Subset this is the original index into
        #     the base dataset's clip enumeration).
        #   anchor_timestamp_us: window[0]["timestamp"] of the clip, unique per clip.
        result["dataset_index"] = torch.tensor(idx, dtype=torch.int64)
        result["anchor_timestamp_us"] = torch.tensor(
            int(self.clips[idx][0]["timestamp"]), dtype=torch.int64)
        # scene_token (string): set by `_group_infos_by_scene`; collated to a list of B strings.
        result["scene_token"] = self.clips[idx][0].get("scene_token", "")
        # split ("train" / "val"): distinguishes clips of different splits inside a ConcatDataset.
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
    # Smoke test on the train and val splits.
    PKL_BASE = "/path/to/pandwm_root/workspace/OpenDWM/data/pickle"
    TEST_CASES = [
        ("val", f"{PKL_BASE}/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl",
         "/local/data/val/rgb"),
        ("train", f"{PKL_BASE}/nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl",
         "/path/to/pandwm_root/nuscenes-360/train"),
    ]
    for split, pkl, rgb in TEST_CASES:
        if not os.path.exists(pkl):
            print(f"[{split}] pkl missing: {pkl}, skipped"); continue
        if not os.path.exists(rgb):
            print(f"[{split}] rgb dir missing: {rgb}, skipped"); continue
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
                print(f"  {k}: list[{n} x {m}] {type(el).__name__}{extra}")
