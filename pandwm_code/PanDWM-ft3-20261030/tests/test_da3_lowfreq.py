"""CPU tests for dwm_pano/da3_lowfreq.py (2026-10-26, GS_CLIPDEPTH_PLAN_20261026.md section 7.4).

    PYTHONPATH=src python3 -m pytest -q tests/test_da3_lowfreq.py
"""

import math

import torch

from dwm_pano.da3_lowfreq import lowfreq_log_ratio

V, H, W = 2, 64, 256


def test_constant_ratio_is_kept_by_both_filters():
    rho = torch.full((V, H, W), 0.3)
    guide = torch.rand(V, H, W) * 3.0
    mask = torch.rand(V, H, W) > 0.3
    for method in ("gauss", "guided"):
        out, stats = lowfreq_log_ratio(rho, mask, guide, method, sigma_px=8.0, radius_px=8, eps=0.01)
        assert float((out - 0.3).abs().max()) < 1e-5, method
        assert abs(stats["lf_global_log_ratio"] - 0.3) < 1e-6


def test_guided_does_not_mix_across_a_depth_edge_but_gauss_does():
    guide = torch.full((V, H, W), math.log(10.0))
    guide[..., W // 2:] = math.log(50.0)                    # near object left, far background right
    rho = torch.zeros(V, H, W)
    rho[..., W // 2:] = 0.3                                 # only the far side needs correcting
    mask = torch.ones(V, H, W, dtype=torch.bool)
    g, _ = lowfreq_log_ratio(rho, mask, guide, "guided", radius_px=16, eps=0.01)
    s, _ = lowfreq_log_ratio(rho, mask, guide, "gauss", sigma_px=16.0)
    near, far = W // 2 - 2, W // 2 + 1
    assert abs(float(g[0, H // 2, near])) < 0.02 and abs(float(g[0, H // 2, far]) - 0.3) < 0.02
    assert 0.05 < float(s[0, H // 2, near]) < 0.25 and 0.05 < float(s[0, H // 2, far]) < 0.25


def test_smooth_gradient_is_followed_and_noise_removed():
    torch.manual_seed(0)
    cols = torch.linspace(0.0, 0.4, W)
    truth = cols.expand(V, H, W).clone()
    rho = truth + 0.1 * torch.randn(V, H, W)
    guide = torch.full((V, H, W), 3.0)
    mask = torch.ones(V, H, W, dtype=torch.bool)
    for method in ("gauss", "guided"):
        out, _ = lowfreq_log_ratio(rho, mask, guide, method, sigma_px=8.0, radius_px=8, eps=0.01)
        inner = (slice(None), slice(16, H - 16), slice(32, W - 32))
        assert float((out - truth)[inner].abs().mean()) < 0.02, method


def test_empty_mask_falls_back_to_zero_and_reports_it():
    rho = torch.randn(V, H, W)
    guide = torch.rand(V, H, W)
    mask = torch.zeros(V, H, W, dtype=torch.bool)
    for method in ("gauss", "guided"):
        out, stats = lowfreq_log_ratio(rho, mask, guide, method, sigma_px=8.0, radius_px=8, eps=0.01)
        assert float(out.abs().max()) == 0.0 and stats["lf_fallback_fraction"] == 1.0
