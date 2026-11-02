"""4D GS decoder trained with Depth Anything 3 pseudo-depth on every batch (2026-10-20, report section 41, C2).

Section 38 (F1) supervised depth only on the 80 training scenes with downloaded labels; the decoder's depth error on
held-out scenes stalled at ~0.21.  Section 40 showed DA3 (ground scale) reaches 0.239 against those labels and, used
at render time (C1b), removes most of the far-view ripples.  Here the depth labels of
PanoGaussianDecoder4DDepthPipeline come from DA3 run online on the step's own frames instead of label files, so all
695 training scenes get depth:

* the step's frames (frames_per_step, stride 2-4 at 10 fps) are reconstructed together by da3_depth.Da3Ranger: the six
  cameras cut back out of the stitched real pano, with the known intrinsics, rig extrinsics and ego poses; any-view
  depth scaled to the metric model and then to the ego ground plane (scale "ground", section 40.1);
* range = DA3's Euclidean range along every decoder pixel's exact content ray (the decoder's own range convention);
* sky (DA3 sky >= sky_threshold) and DA3 range >= far_cut_m get sky_fill_m and stay valid, like the 100 m fill of the
  downloaded labels: the metre terms pull them to 100 m, the log terms skip them (they are >= far_cut_m);
* ranges at or below min_range_m (or not finite) are invalid.
If the ground scale is not finite (no road pixels) the metric scale is used and the step logs da3_fallback = 1.

The DA3 networks are built lazily on the first step, frozen, and never enter the decoder, its optimizer or its
checkpoints.  Losses, weights, frame choice and the RNG stream are the parent's (depth_optional must stay false:
every batch has labels).  The step report gains da3_scale, da3_fallback, da3_ground_over_metric, da3_sky_fraction,
da3_near_fraction and da3_seconds.
"""

from __future__ import annotations

import math
import time

import torch

from dwm_pano_4d.pipeline_depth import PanoGaussianDecoder4DDepthPipeline

DEFAULT_DA3 = {"repo_src": "", "anyview_dir": "", "metric_dir": "", "scale": "ground", "sky_threshold": 0.3,
               "min_range_m": 0.5, "far_cut_m": 99.0, "sky_fill_m": 100.0, "crop_hw": [392, 784]}


class PanoGaussianDecoder4DDa3Pipeline(PanoGaussianDecoder4DDepthPipeline):
    def __init__(self, *args, gs4d_da3: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gs4d_da3 = dict(DEFAULT_DA3)
        self.gs4d_da3.update(gs4d_da3 or {})
        unknown = set(self.gs4d_da3) - set(DEFAULT_DA3)
        if unknown:
            raise ValueError(f"unknown gs4d_da3 options: {sorted(unknown)}")
        if self.gs4d_da3["scale"] not in ("ground", "pose", "metric"):
            raise ValueError(f"gs4d_da3.scale must be ground, pose or metric, got {self.gs4d_da3['scale']}")
        if self.gs4d_depth["depth_optional"]:
            raise ValueError("DA3 labels every batch: gs4d_depth.depth_optional must be false")
        self._da3 = None
        self._da3_stats = {}

    def _da3_ranger(self):
        if self._da3 is None:
            from dwm_pano_4d.da3_depth import Da3Ranger
            cfg = self.gs4d_da3
            self._da3 = Da3Ranger(cfg["repo_src"], cfg["anyview_dir"], cfg["metric_dir"], self.device,
                                  crop_hw=tuple(cfg["crop_hw"]))
            for net in (self._da3.anyview, self._da3.metric):
                net.requires_grad_(False)
        return self._da3

    @torch.no_grad()
    def _load_frame_ranges(self, batch, frame_indices, return_valid=False):
        """DA3 ranges (F, 2, H, W) on the decoder grid for the selected frames, in place of the label files."""
        if batch["vae_images"].shape[0] != 1:
            raise ValueError("DA3 labels expect batch_size 1")
        start = time.time()
        cfg = self.gs4d_da3
        idx = torch.as_tensor(frame_indices).long().cpu()
        halves = batch["vae_images"][0, idx].to(self.device, dtype=torch.float32)       # (F, 2, 3, H, W) in [0, 1]
        canvases = self._unpadded_stitch_canvas(halves).clamp(0.0, 1.0)                  # (F, 3, Hc, Wc)
        ego_to_world = batch["ego_transforms"][0, idx, 0].to(torch.float64)
        res = self._da3_ranger().frame_ranges(self, canvases, ego_to_world)
        name, fallback = cfg["scale"], 0.0
        scale = res[f"scale_{name}"]
        if not (math.isfinite(scale) and scale > 0.0):
            name, fallback = "metric", 1.0
            scale = res["scale_metric"]
        r = res[f"range_{name}"].float()
        sky = res["sky"].float() >= cfg["sky_threshold"]
        content = self.pixel_content_ray_valid.unsqueeze(0).to(r.device)
        finite = torch.isfinite(r)
        far = sky | (finite & (r >= cfg["far_cut_m"]))
        valid = content & (far | (finite & (r > cfg["min_range_m"])))
        ranges = torch.where(far, torch.full_like(r, cfg["sky_fill_m"]), r)
        ranges = torch.where(valid, ranges, torch.full_like(r, self.training_config["range_hole_fill_m"]))
        n = float(content.expand_as(valid).sum().clamp_min(1))
        self._da3_stats = {"da3_scale": float(scale), "da3_fallback": fallback,
                           "da3_ground_over_metric": float(res["ground_over_metric"]),
                           "da3_sky_fraction": float((sky & content).sum()) / n,
                           "da3_near_fraction": float((valid & ~far).sum()) / n,
                           "da3_seconds": time.time() - start}
        if not return_valid:
            return ranges
        return ranges, valid

    def train_step(self, batch: dict, global_step: int):
        self._da3_stats = {}
        super().train_step(batch, global_step)
        if self._da3_stats and self.loss_report_list:
            self.loss_report_list[-1].update(self._da3_stats)
