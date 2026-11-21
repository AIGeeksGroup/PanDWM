"""4D GS decoder trained on the geometry it is rendered with at test time.

At test time every frame's Gaussians are slid along their rays to DA3 ranges (da3_depth.rescale_gaussians). Here the
same slide is applied inside training, to the DA3 ranges the parent (pipeline_da3) reconstructs jointly for the step's
frames, so appearance is learned for the geometry it is rendered with. Every loss (own, composite, LOO, box ownership)
sees the slid Gaussians; the decoder's depth terms still act on its own predicted range.

gs4d_relabel options:
* bound: 0 uses the DA3 range itself; > 0 uses da3_depth.bounded_range, so the decoder may move a Gaussian at most a
  factor exp(+-bound) off DA3;
* min_range_m, far_cut_m: only content rays whose DA3 label is valid with min_range_m < range < far_cut_m are slid
  (sky and far pixels hold sky_fill_m in the labels); the others keep the decoder's range;
* no_offset: put every centre back on its pixel ray (origin + direction * range) before the slide, dropping the
  decoder's free xyz offset;
* plane_offset_px > 0 (needs no_offset false): keep only the offset component across the ray, softly bounded to that
  many decoder pixels (da3_depth.plane_offsets);
* loo_perceptual: weight of an LPIPS term on the leave-one-out render, using the same mask as the LOO L1.
The step report gains relabel_fraction, relabel_abs_log, offset_norm and loo_perceptual.
"""

from __future__ import annotations

import torch

from dwm_pano.gs4d.da3_depth import bounded_range, plane_offsets, rescale_gaussians
from dwm_pano.gs4d.pipeline_da3 import PanoGaussianDecoder4DDa3Pipeline

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
