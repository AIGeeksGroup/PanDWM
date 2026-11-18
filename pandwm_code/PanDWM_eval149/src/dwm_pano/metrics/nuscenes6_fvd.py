"""Band-limited FVD in the six *real* nuScenes camera frames.

This is the metric half of the protocol bridge (see
:mod:`dwm_pano.nuscenes6_bridge`).  It exists so a number from this project
can be stated in the space every published driving video-generation number
lives in, instead of only in this project's own panorama space.

It subclasses :class:`PanoPatchFrechetVideoDistance` and overrides *only*
where the six views come from: the parent slices six Voronoi column bands out
of a V=1 stitched pano, while this one takes six views that
:func:`dwm_pano.nuscenes6_bridge.resample_pano_to_rig` already resampled into
the real cameras.  Normalization, resize, I3D weights, Frechet math and the
per-camera accounting are inherited unchanged, so the two are the same
instrument family and differ by exactly one documented step.

What the resulting number is and is not
---------------------------------------
It **is** comparable across arms that pass through the identical bridge, and
across the real-vs-real floor computed the same way.

It is **not** comparable to any published full-frame nuScenes number.  The
pano is a +-16.67 deg elevation band, so about 16% of the rows of the five
~39 deg cameras and 46% of CAM_BACK's rows have no content to render at all.
Report it as ``FVD-I3D-nusc6band-<clip_len>-n<samples>`` and never drop the
``band`` or the sample count: the estimator's absolute magnitude is
bias-dominated at these sample sizes (a real-vs-real floor with zero model
reads far above zero), so only same-n arm-vs-arm reads are meaningful.
"""
import os
from pathlib import Path

import numpy as np
import torch

from dwm_pano.metrics.fvd_pano import PanoPatchFrechetVideoDistance
from dwm_pano.nuscenes6_bridge import (
    PANO_CANVAS_OVERLAP_COLUMNS, apply_camera_sampling_grid,
    build_rig_sampling_grids, canvas_source_axes,
)


class NuScenes6BandFrechetVideoDistance(PanoPatchFrechetVideoDistance):
    """FVD pooled over six real-camera band-limited views.

    Args:
        inception_3d_checkpoint_path: same TATS I3D-400 checkpoint the pano
            metric uses; passing a different one silently makes the numbers
            incomparable.
        sequence_count: truncate clips to this many frames (-1 keeps all).
        apply_valid_mask: multiply each view by the bridge's validity mask
            before feature extraction.  The bridge already zeroes out-of-band
            pixels, so this only matters when a caller passes unmasked views;
            it is on by default so a masking mistake cannot inflate a score.
    """

    def __init__(
        self, inception_3d_checkpoint_path: str, sequence_count: int = -1,
        num_classes: int = 400, apply_valid_mask: bool = True,
        render_height: int = 360, render_width: int = 640,
        max_calibration_entries: int = 8, quantize_uint8: bool = False,
        compute_front_fid: bool = False,
        **kwargs
    ):
        super().__init__(
            inception_3d_checkpoint_path=inception_3d_checkpoint_path,
            sequence_count=sequence_count, num_classes=num_classes,
            equal_patch_width=0, **kwargs)
        self.apply_valid_mask = bool(apply_valid_mask)
        self.render_size = (int(render_height), int(render_width))
        self.max_calibration_entries = int(max_calibration_entries)
        # External-method comparisons persist generated pixels between the
        # generator and this metric.  Enable this only when every arm is
        # deliberately pinned to the same lossless uint8 interchange format.
        # Default off preserves every historical float-input score.
        self.quantize_uint8 = bool(quantize_uint8)
        if compute_front_fid:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.front_fid = FrechetInceptionDistance(
                feature=2048, normalize=True)
        else:
            self.front_fid = None
        # Grids depend only on calibration and sizes, never on pixels, and a
        # clip's frames all share one rig, so this turns 32 grid builds per
        # clip into one -- the difference between minutes and an hour per arm.
        self._grid_cache = {}
        self._source_axes = None
        self._source_shape = None
        self.coverage_reports = None

    def update(self, views, real=True, valid=None):
        """Update state from six real-camera views.

        Args:
            views: `(batch, sequence, 6, channels, height, width)` in [0, 1],
                camera-major in ``CAM_ORDER``.  All six share one height and
                width on purpose -- per-camera output sizes would reintroduce
                the per-camera horizontal squeeze that makes the parent
                metric's per-camera values incomparable across cameras.
            real: whether these are reference or generated views.
            valid: optional `(6, height, width)` or
                `(batch, sequence, 6, height, width)` bool mask from the
                bridge.
        """
        if views.ndim != 6:
            raise ValueError(
                "views must be (B, T, 6, C, H, W), got shape {}".format(
                    tuple(views.shape)))
        if views.shape[2] != len(self.CAM_NAMES):
            raise ValueError(
                "expected {} cameras on dim 2, got {}".format(
                    len(self.CAM_NAMES), views.shape[2]))

        if self.sequence_count >= 0:
            views = views[:, :self.sequence_count]
            if valid is not None and valid.ndim == 5:
                valid = valid[:, :self.sequence_count]
        assert views.shape[1] >= self.i3d_min

        if valid is not None and self.apply_valid_mask:
            batch, sequence, cameras = views.shape[:3]
            spatial = tuple(views.shape[-2:])
            if valid.ndim == 3:
                expected = (cameras,) + spatial
                mask = valid[None, None, :, None]
            elif valid.ndim == 5:
                expected = (batch, sequence, cameras) + spatial
                mask = valid[:, :, :, None]
            else:
                raise ValueError(
                    "valid must be (6, H, W) or (B, T, 6, H, W), got shape "
                    "{}".format(tuple(valid.shape)))
            # Check the shape rather than letting broadcasting decide: a mask
            # whose height/width or camera count is wrong would otherwise
            # either raise a bare RuntimeError deep in the multiply, or -- far
            # worse -- broadcast silently against a size-1 axis and mask the
            # wrong pixels.
            if tuple(valid.shape) != expected:
                raise ValueError(
                    "valid has shape {} but views require {}".format(
                        tuple(valid.shape), expected))
            views = views * mask.to(views.dtype)

        # Camera-major list, matching the parent's patch ordering so the
        # inherited per-camera accounting keeps its CAM_NAMES labels.
        patches = [views[:, :, index] for index in range(views.shape[2])]
        self._accumulate_patches(patches, real)

    # ------------------------------------------------------------------
    # Evaluation-pipeline entry point
    # ------------------------------------------------------------------

    @staticmethod
    def _calibration_key(camera_to_ego, intrinsics, image_size):
        """Rounded fingerprint of one rig.

        nuScenes has a handful of distinct rigs, and their entries differ far
        above 1e-4, so rounding cannot merge two real rigs while it does keep
        float noise from exploding the cache.
        """
        return (
            tuple(np.round(np.asarray(camera_to_ego, dtype=np.float64),
                           4).ravel().tolist()),
            tuple(np.round(np.asarray(intrinsics, dtype=np.float64),
                           3).ravel().tolist()),
            tuple(np.asarray(image_size).astype(np.int64).ravel().tolist()),
        )

    def _grids_for(self, camera_to_ego, intrinsics, image_size, source_shape):
        key = self._calibration_key(camera_to_ego, intrinsics, image_size)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        if len(self._grid_cache) >= self.max_calibration_entries:
            raise RuntimeError(
                "saw more than {} distinct sensor rigs; raise "
                "max_calibration_entries only after checking the dataset "
                "calibration, because an unbounded cache here usually means "
                "the calibration is jittering".format(
                    self.max_calibration_entries))
        grids, valid, reports = build_rig_sampling_grids(
            camera_to_ego, intrinsics, image_size, source_shape,
            self.render_size, band_limit=True,
            source_axes=self._source_axes)
        self._grid_cache[key] = (grids, valid, reports)
        if self.coverage_reports is None:
            self.coverage_reports = reports
        return self._grid_cache[key]

    def update_from_canvas(self, canvas, batch, start, real=True):
        """Bridge a batch of V=1 canvases and accumulate them.

        Args:
            canvas: ``[B, T, C, H, W]`` stitched V=1 panoramas in [0, 1] --
                exactly what the FVD block in ``evaluate_pipeline`` already
                built, so both metrics score the same pixels.
            batch: the dataloader batch, for ``pano_camera_transforms`` /
                ``pano_camera_intrinsics`` / ``pano_camera_image_size``.  Note
                those keys hold the **real six-camera** calibration; the
                ``camera_*`` keys are the virtual pano camera and are wrong
                here.
            start: index of ``canvas``'s first frame within the batch's own
                frame axis, so the calibration is read at the matching frame.
            real: reference or generated.
        """
        if canvas.ndim != 5:
            raise ValueError(
                "canvas must be [B, T, C, H, W], got shape {}".format(
                    tuple(canvas.shape)))
        for key in ("pano_camera_transforms", "pano_camera_intrinsics",
                    "pano_camera_image_size"):
            if key not in batch:
                raise KeyError(
                    "the nusc6band metric needs batch['{}']; the camera_* "
                    "keys are the virtual pano camera and must not be "
                    "substituted".format(key))

        source_shape = tuple(int(v) for v in canvas.shape[-2:])
        if self._source_shape is None:
            self._source_shape = source_shape
            # Half width is the canvas width plus the dropped overlap, split
            # in two -- the inverse of how the pipeline concatenates halves.
            half_width = (source_shape[1] + PANO_CANVAS_OVERLAP_COLUMNS) // 2
            self._source_axes = canvas_source_axes(
                half_size=(source_shape[0], half_width))
        elif source_shape != self._source_shape:
            raise ValueError(
                "canvas shape changed from {} to {} mid-run; the cached "
                "sampling grids would silently address the wrong pixels"
                .format(self._source_shape, source_shape))

        views = []
        masks = []
        for item in range(canvas.shape[0]):
            frame = start
            grids, valid, _ = self._grids_for(
                batch["pano_camera_transforms"][item, frame].cpu().numpy(),
                batch["pano_camera_intrinsics"][item, frame].cpu().numpy(),
                batch["pano_camera_image_size"][item, frame].cpu().numpy(),
                source_shape)
            per_camera = [
                apply_camera_sampling_grid(
                    canvas[item], grids[index], valid[index])
                for index in range(grids.shape[0])
            ]
            views.append(torch.stack(per_camera, dim=1))
            masks.append(torch.as_tensor(valid, device=canvas.device))
        views = torch.stack(views)
        if self.quantize_uint8:
            views = (views.clamp(0, 1) * 255).round() / 255
        if self.front_fid is not None:
            # CAM_FRONT is index 3 in the inherited authoritative CAM_NAMES.
            front = views[:, :, self.CAM_NAMES.index("CAM_FRONT")]
            self.front_fid.update(front.flatten(0, 1), real=real)
        dump_root = os.environ.get("PANDWM_COMMON_VIS_DIR")
        if dump_root:
            if "anchor_timestamp_us" not in batch:
                raise KeyError(
                    "PANDWM_COMMON_VIS_DIR requires anchor_timestamp_us")
            if not self.quantize_uint8:
                raise RuntimeError(
                    "qualitative common dumps require quantize_uint8=True")
            front = views[:, :, self.CAM_NAMES.index("CAM_FRONT")]
            role_dir = Path(dump_root) / ("real" if real else "fake")
            role_dir.mkdir(parents=True, exist_ok=True)
            for item in range(front.shape[0]):
                anchor = int(batch["anchor_timestamp_us"][item])
                target = role_dir / "{}.pt".format(anchor)
                temporary = target.with_suffix(".pt.tmp")
                torch.save(
                    (front[item] * 255).round().to(torch.uint8).cpu(),
                    temporary)
                os.replace(temporary, target)
        self.update(views, real=real, valid=masks[0])

    def compute(self):
        fvd = super().compute()
        if self.front_fid is None:
            return fvd
        return {
            # The parent metric pools all six camera feature populations.
            # CAM_FRONT FVD is reported by compute_per_camera(); labelling
            # this pooled value as front_fvd makes a single-front comparison
            # silently use the wrong number.
            "pooled_fvd": fvd,
            "front_fid": self.front_fid.compute(),
        }
