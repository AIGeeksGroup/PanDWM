"""Task-matched nuScenes input adapter for the official HorizonDrive code.

The public HorizonDrive evaluator normally reads raw nuScenes camera images
and enumerates its own 12 Hz scene rollouts.  That is useful for reproducing
the paper, but it is not a controlled comparison with PanDWM.  This adapter
keeps HorizonDrive's model and rollout code unchanged while replacing only
the validation data path:

* clips are enumerated with PanDWM's timestamp sampler;
* RGB, bbox and HD-map inputs come from the same 9600x600 pano caches;
* the same two-half resize/dedup canvas is resampled into CAM_FRONT's real
  calibration, restricted to the pano's available vertical band.

It deliberately lives in first-party ``dwm_pano`` rather than modifying the
read-only HorizonDrive reference checkout.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tvf

try:
    from horizondrive.datasets.nuscenes_dataset import nuScenesDataset
except ImportError as error:  # Let lightweight PanDWM modules remain usable.
    _HORIZONDRIVE_IMPORT_ERROR = error

    class nuScenesDataset:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "HorizonDrive must be on PYTHONPATH to construct "
                "HorizonDrivePanoBandDataset"
            ) from _HORIZONDRIVE_IMPORT_ERROR

from dwm_pano.datasets.nuscenes_pano import (
    _enumerate_pano_clips,
    _group_infos_by_scene,
    _slice_full_pano_to_halves,
    _slice_full_pano_to_halves_nearest,
)
from dwm_pano.nuscenes6_bridge import (
    PANO_CANVAS_OVERLAP_COLUMNS,
    apply_camera_sampling_grid,
    build_camera_sampling_grid,
    canvas_source_axes,
)


def common_clip_indices(
    data_infos: list[dict],
    scene_token_groups: Iterable[Iterable[str]],
    sequence_length: int = 35,
    fps: float = 10.0,
    stride_s: float = 20.0,
):
    """Return the exact timestamp clips produced by PanDWM's val sampler.

    HorizonDrive's annotation pickle already carries per-scene token groups,
    so official scene metadata is not needed merely to recover membership.
    Synthetic group ids are sufficient: PanDWM's sampler depends on scene
    membership and timestamps, not on the spelling of the scene token.
    """
    token_to_scene = {}
    for scene_index, tokens in enumerate(scene_token_groups):
        for token in tokens:
            if token in token_to_scene:
                raise ValueError("duplicate token in scene groups: {}".format(token))
            token_to_scene[token] = "scene_group_{:04d}".format(scene_index)

    scene_infos = _group_infos_by_scene(
        data_infos, token2scene=token_to_scene, strict=True)
    pairs = list(_enumerate_pano_clips(
        scene_infos, int(sequence_length), [(float(fps), float(stride_s))]))
    token_to_index = {info["token"]: index
                      for index, info in enumerate(data_infos)}
    clips = [
        [token_to_index[info["token"]] for info in clip]
        for clip, _audit in pairs
    ]
    audits = [audit for _clip, audit in pairs]
    return clips, audits


class HorizonDrivePanoBandDataset(nuScenesDataset):
    """HorizonDrive dataset backed by PanDWM's cached pano validation data."""

    def __init__(
        self,
        *args,
        pano_rgb_dir: str,
        pano_bbox_dir: str | None = None,
        pano_hdmap_dir: str | None = None,
        common_sequence_length: int = 35,
        common_fps: float = 10.0,
        common_stride_s: float = 20.0,
        expected_clip_count: int | None = 149,
        pano_half_size: tuple[int, int] = (2304, 256),
        right_trim_px: int = PANO_CANVAS_OVERLAP_COLUMNS,
        native_camera_size: tuple[int, int] = (1600, 900),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.nuscenes_camera_names != ["CAM_FRONT"]:
            raise ValueError(
                "the task-matched protocol currently requires exactly "
                "camera_names=[camera_front], got {}".format(
                    self.nuscenes_camera_names))

        self.pano_rgb_dir = pano_rgb_dir
        self.pano_bbox_dir = pano_bbox_dir
        self.pano_hdmap_dir = pano_hdmap_dir
        self.common_sequence_length = int(common_sequence_length)
        self.common_fps = float(common_fps)
        self.common_stride_s = float(common_stride_s)
        self.pano_half_size = tuple(int(value) for value in pano_half_size)
        self.right_trim_px = int(right_trim_px)
        self.native_camera_size = tuple(int(value)
                                        for value in native_camera_size)
        self._source_axes = canvas_source_axes(
            half_size=(self.pano_half_size[1], self.pano_half_size[0]),
            right_trim_px=self.right_trim_px)
        self._sampling_grid_cache = {}
        self._active_frames = None

        clips, audits = common_clip_indices(
            self.data_infos, self.scene_tokens,
            sequence_length=self.common_sequence_length,
            fps=self.common_fps, stride_s=self.common_stride_s)
        if expected_clip_count is not None and len(clips) != expected_clip_count:
            raise ValueError(
                "common protocol expected {} clips but enumerated {}; the "
                "annotation pickle is not the frozen PanDWM validation set"
                .format(expected_clip_count, len(clips)))
        self.clip_infos = clips
        self.common_clip_audit = audits
        self.length = len(clips)
        self.clip_token_to_clip_idx = {
            self.data_infos[clip[0]]["token"]: clip_index
            for clip_index, clip in enumerate(clips)
        }

    def get_data_info(self, index, infer_image_len=None):
        frames = super().get_data_info(index, infer_image_len=infer_image_len)
        self._active_frames = frames
        return frames

    @staticmethod
    def _cache_path(directory: str, timestamp, suffix: str):
        path = os.path.join(directory, "{}{}".format(int(timestamp), suffix))
        if not os.path.isfile(path):
            raise FileNotFoundError("missing common-protocol pano cache: {}".format(path))
        return path

    def _frame_calibration(self, frame):
        camera_index = frame["camera_names"].index("CAM_FRONT")
        camera_to_ego = np.asarray(
            frame["camera2ego"][camera_index], dtype=np.float64)
        intrinsics = np.asarray(
            frame["camera_intrinsics"][camera_index], dtype=np.float64)[:3, :3]
        return camera_to_ego, intrinsics

    def _sampling_grid(self, frame, source_size, output_size):
        camera_to_ego, intrinsics = self._frame_calibration(frame)
        key = (
            tuple(np.round(camera_to_ego, 5).ravel()),
            tuple(np.round(intrinsics, 4).ravel()),
            tuple(source_size), tuple(output_size), self.native_camera_size,
        )
        cached = self._sampling_grid_cache.get(key)
        if cached is None:
            cached = build_camera_sampling_grid(
                camera_to_ego, intrinsics, self.native_camera_size,
                source_size=source_size, output_size=output_size,
                band_limit=True, camera_index=3,
                source_axes=self._source_axes)
            self._sampling_grid_cache[key] = cached
        return cached

    def _load_canvas(self, path: str, nearest: bool):
        full = Image.open(path).convert("RGB")
        half_w, half_h = self.pano_half_size
        slicer = (_slice_full_pano_to_halves_nearest if nearest
                  else _slice_full_pano_to_halves)
        left, right = slicer(full, half_w, half_h)
        left_tensor = tvf.to_tensor(left)
        right_tensor = tvf.to_tensor(right)
        return torch.cat(
            [left_tensor, right_tensor[..., self.right_trim_px:]], dim=-1)

    def _load_band_sequence(
        self, directory: str, suffix: str, target_height: int,
        target_width: int, nearest: bool,
    ):
        if self._active_frames is None:
            raise RuntimeError("get_data_info must run before loading pano frames")
        output_size = (int(target_height), int(target_width))
        output = []
        for frame in self._active_frames:
            path = self._cache_path(directory, frame["timestamp"], suffix)
            canvas = self._load_canvas(path, nearest=nearest)
            grid, valid, _report = self._sampling_grid(
                frame, tuple(canvas.shape[-2:]), output_size)
            band = apply_camera_sampling_grid(canvas[None], grid, valid)[0]
            output.append(band)
        return torch.stack(output)

    def read_and_process_images(
        self, image_paths, target_height, target_width, image_transforms=None,
    ):
        del image_paths, image_transforms
        pixels = self._load_band_sequence(
            self.pano_rgb_dir, ".jpg", target_height, target_width,
            nearest=False)
        return pixels * 2.0 - 1.0

    def get_batch(
        self, idx, resoultion=(19, 384, 768), conditions=(),
        infer_image_len=None, valid_mode=None,
    ):
        requested = list(conditions)
        passthrough = [name for name in requested if name == "action"]
        sample = super().get_batch(
            idx, resoultion, passthrough,
            infer_image_len=infer_image_len, valid_mode=valid_mode)
        _frames, height, width = resoultion
        if "bbox" in requested:
            if self.pano_bbox_dir is None:
                raise ValueError("bbox requested but pano_bbox_dir is unset")
            bbox = self._load_band_sequence(
                self.pano_bbox_dir, ".png", height, width, nearest=True)
            sample["conditions"]["bbox"] = bbox * 2.0 - 1.0
        if "hdmap" in requested:
            if self.pano_hdmap_dir is None:
                raise ValueError("hdmap requested but pano_hdmap_dir is unset")
            hdmap = self._load_band_sequence(
                self.pano_hdmap_dir, ".png", height, width, nearest=True)
            sample["conditions"]["hdmap"] = hdmap * 2.0 - 1.0
        sample["frame_timestamps_us"] = torch.tensor(
            [int(frame["timestamp"]) for frame in self._active_frames],
            dtype=torch.int64)
        sample["common_protocol"] = {
            "fps": self.common_fps,
            "condition_frames": 3,
            "predicted_frames": self.common_sequence_length - 3,
        }
        return sample
