"""4D GS decoder trained on the geometry it is rendered with at test time (2026-10-30, GS_FT3_PLAN_20261030.md).

Why: at test time every frame's Gaussians are slid along their rays to DA3 ranges (gs_scene_v3 da3_mode "scale",
da3_depth.rescale_gaussians), but C2 / R1 learned opacity, size and colour on the decoder's own ranges.  The
three-geometry LOO diagnostic (OVERFIT_DIAG_20261029.md) showed the swap costs +7-8 % own-view error and LOO sharpness
(0.93 -> 0.86-0.90), while the clip-consistent DA3 geometry lowers LOO error 12 % (mid range 20 %).  Here the same
slide is applied inside training, to the DA3 ranges the parent already reconstructs jointly for the step's frames
(pipeline_da3), so the appearance is learned for the geometry it is rendered with.  Every loss (own, composite, LOO,
box ownership) sees the slid Gaussians; the decoder's depth terms still act on its own predicted range.

gs4d_relabel:
* bound: 0 = the DA3 range itself (exactly the test-time "scale" relabel); > 0 = da3_depth.bounded_range, the decoder
  may move a Gaussian at most a factor exp(+-bound) off DA3 (gs_scene_v3 da3_mode "bounded" at test time);
* pixels slid: content rays whose DA3 label is valid with min_range_m < DA3 < far_cut_m (sky and far pixels hold
  sky_fill_m >= far_cut_m in the labels), as the test-time valid mask (0.5 < r < 99, not sky); the others keep the
  decoder's range;
* no_offset: every centre is put back on its pixel ray, origin + direction * range, before the slide (the decoder's
  free xyz offset is dropped; gs_scene_v3 SceneOptions.no_offset at test time).  InfiniSplat v2 (2610.20341) found
  free offsets on fixed depth hurt novel views (they compensate attribute errors);
* plane_offset_px > 0 (2026-10-30, section 6; needs no_offset false): keep only the offset's component across the
  ray, softly bounded to that many decoder pixels (da3_depth.plane_offsets), so sub-pixel alignment survives while
  the depth stays on the ray; G1 / G2 (no offset) lost 0 m FID against G0 (free offset);
* loo_perceptual: weight of an LPIPS term on the leave-one-out render (the parent's LOO is L1 only, which rewards
  blur under misregistration), same mask as the LOO L1.
The step report gains relabel_fraction, relabel_abs_log (mean |log decoder / DA3| on slid pixels), offset_norm (mean
|xyz offset| in metres before it is dropped / scaled) and loo_perceptual.
"""

from __future__ import annotations

import torch

from dwm_pano_4d.da3_depth import bounded_range, plane_offsets, rescale_gaussians
from dwm_pano_4d.pipeline_da3 import PanoGaussianDecoder4DDa3Pipeline

DEFAULT_RELABEL = {"bound": 0.0, "min_range_m": 0.5, "far_cut_m": 99.0, "no_offset": False, "loo_perceptual": 0.0,
                   "plane_offset_px": 0.0}


class PanoGaussianDecoder4DRelabelPipeline(PanoGaussianDecoder4DDa3Pipeline):
    def __init__(self, *args, gs4d_relabel: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gs4d_relabel = dict(DEFAULT_RELABEL)
        self.gs4d_relabel.update(gs4d_relabel or {})
        unknown = set(self.gs4d_relabel) - set(DEFAULT_RELABEL)
        if unknown:
            raise ValueError(f"unknown gs4d_relabel options: {sorted(unknown)}")
        if self.gs4d_relabel["no_offset"] and self.gs4d_relabel["plane_offset_px"] > 0.0:
            raise ValueError("gs4d_relabel: no_offset and plane_offset_px are exclusive")
        if self.gs4d_relabel["far_cut_m"] > self.gs4d_da3["sky_fill_m"]:
            raise ValueError("gs4d_relabel.far_cut_m must not exceed gs4d_da3.sky_fill_m (sky would be slid)")

    def _train_gaussians(self, k, g, predicted_range, gt_ranges, gt_range_valid, pixel_origins,
                         camera_origin_scale):
        if gt_ranges is None:
            raise ValueError("relabel training needs the DA3 labels on every step (a depth weight > 0)")
        cfg = self.gs4d_relabel
        da3 = gt_ranges[k:k + 1].to(predicted_range.dtype)                               # (1, V, H, W)
        valid = gt_range_valid[k:k + 1] & (da3 > cfg["min_range_m"]) & (da3 < cfg["far_cut_m"])
        target = bounded_range(predicted_range, da3, float(cfg["bound"]))
        if pixel_origins is None:
            origins = torch.zeros_like(g[:, 0:3])
        else:
            origins = pixel_origins.reshape(-1, 3).to(g.device, g.dtype) * float(camera_origin_scale)
        dirs = self.pixel_directions.reshape(-1, 3).to(g.device, g.dtype)
        on_ray = origins + dirs * predicted_range.reshape(-1, 1).to(g.dtype)
        with torch.no_grad():
            offset_norm = float((g[:, 0:3] - on_ray).norm(dim=-1).mean())
        if cfg["no_offset"]:
            g = torch.cat([on_ray, g[:, 3:]], dim=-1)
        elif cfg["plane_offset_px"] > 0.0:
            g = plane_offsets(g, origins, dirs, predicted_range, float(cfg["plane_offset_px"]))
        out, _ = rescale_gaussians(g, predicted_range[0], target[0], valid[0], origins)
        with torch.no_grad():
            n = float(valid.sum().clamp_min(1))
            resid = (predicted_range.clamp_min(1e-3).log() - da3.clamp_min(1e-3).log()).abs()
            report = {"relabel_fraction": float(valid.float().mean()),
                      "relabel_abs_log": float(resid[valid].sum()) / n, "offset_norm": offset_norm}
        return out, report

    def _loo_extra_loss(self, rgb, target_rgb, mask):
        weight = float(self.gs4d_relabel["loo_perceptual"])
        if weight <= 0.0:
            return None, {}
        lpips = self._perceptual(rgb, target_rgb, mask)
        return weight * lpips, {"loo_perceptual": float(lpips.item())}
