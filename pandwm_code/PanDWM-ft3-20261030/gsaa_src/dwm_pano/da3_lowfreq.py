"""Low-frequency DA3 scale correction of the decoder range (2026-10-26, GS_CLIPDEPTH_PLAN_20261026.md section 7).

rho = log(DA3 range / decoder range) on the trusted pixels (mask) of a (V, H, W) decoder grid is low-passed per half and
the decoder range is multiplied by exp(rho~), so DA3 corrects slowly varying range errors while the decoder keeps its
own edges and local structure:
  "gauss"   masked normalised convolution, rho~ = G_sigma * (m rho) / G_sigma * m, zero padding;
  "guided"  masked guided filter (He et al. 2013) with guide I = log decoder range: per window rho ~ a I + b fitted on
            the trusted pixels, a = cov(I, rho) / (var(I) + eps), then a, b box-averaged and rho~ = mean_a I + mean_b,
            so a correction does not leak across a depth edge of the decoder.
Pixels whose neighbourhood holds too little trusted weight (gauss: G_sigma * m < 1e-3; guided: < 10 % of the window)
fall back to the frame's global median rho (the "global" mode's ratio).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_blur(x, sigma):
    """(V, H, W) -> separable Gaussian blur with zero padding, radius ceil(3 sigma)."""
    r = int(math.ceil(3.0 * sigma))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    y = F.conv2d(x[:, None], k.view(1, 1, 1, -1), padding=(0, r))
    y = F.conv2d(y, k.view(1, 1, -1, 1), padding=(r, 0))
    return y[:, 0]


def box_sum(x, r):
    """(V, H, W) -> sum over the (2r + 1)^2 window, zero padding."""
    k = 2 * r + 1
    return F.avg_pool2d(x[:, None], k, stride=1, padding=r, count_include_pad=True)[:, 0] * (k * k)


def lowfreq_log_ratio(rho, mask, guide, method, sigma_px=32.0, radius_px=32, eps=0.01):
    """rho / guide (V, H, W) float, mask (V, H, W) bool -> (rho~ (V, H, W) float32, stats dict)."""
    m = mask.double()
    p = torch.where(mask, rho.double(), torch.zeros_like(m))
    glob = float(rho[mask].median()) if bool(mask.any()) else 0.0
    if method == "gauss":
        num, den = gaussian_blur(m * p, sigma_px), gaussian_blur(m, sigma_px)
        ok = den >= 1e-3
        out = torch.where(ok, num / den.clamp_min(1e-12), torch.full_like(num, glob))
    elif method == "guided":
        r = int(radius_px)
        centre = float(guide[mask].mean()) if bool(mask.any()) else 0.0
        I = guide.double() - centre                       # centred for float stability
        n = box_sum(m, r)
        ok = n >= 0.1 * (2 * r + 1) ** 2
        nz = n.clamp_min(1.0)
        mean_i, mean_p = box_sum(m * I, r) / nz, box_sum(m * p, r) / nz
        var = (box_sum(m * I * I, r) / nz - mean_i ** 2).clamp_min(0.0)
        cov = box_sum(m * I * p, r) / nz - mean_i * mean_p
        a = torch.where(ok, cov / (var + eps), torch.zeros_like(cov))
        b = torch.where(ok, mean_p - a * mean_i, torch.full_like(cov, glob))
        count = box_sum(torch.ones_like(m), r)
        out = box_sum(a, r) / count * I + box_sum(b, r) / count
    else:
        raise ValueError(f"lowfreq method {method!r}")
    out = out.float()
    return out, {"lf_fallback_fraction": float((~ok).float().mean()), "lf_global_log_ratio": glob}
