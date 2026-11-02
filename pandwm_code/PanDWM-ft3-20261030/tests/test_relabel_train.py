"""CPU tests for the train-on-test-geometry relabel (2026-10-30, GS_FT3_PLAN_20261030.md).

    PYTHONPATH=src:<gs4d>/src python3 -m pytest -q tests/test_relabel_train.py
"""

import math
import types

import torch

from dwm_pano_4d.da3_depth import bounded_range, rescale_gaussians

V, H, W = 2, 8, 16
N = V * H * W


def _gaussians(pred, origins, dirs, offset_scale=0.0):
    off = torch.randn(N, 3) * offset_scale
    means = origins + dirs * pred.reshape(-1, 1) + off
    rgb, opa = torch.rand(N, 3), torch.rand(N, 1)
    rot = torch.nn.functional.normalize(torch.randn(N, 4), dim=-1)
    scales = torch.rand(N, 3) * 0.05
    return torch.cat([means, rgb, opa, rot, scales], dim=-1)


def test_bounded_range_limits():
    da3 = torch.rand(V, H, W) * 50 + 1
    assert torch.equal(bounded_range(da3 * 2, da3, 0.0), da3)
    far = bounded_range(da3 * 3, da3, 0.05)
    assert float((far / da3).max()) <= math.exp(0.05) + 1e-5
    near = bounded_range(da3 / 3, da3, 0.05)
    assert float((near / da3).min()) >= math.exp(-0.05) - 1e-5
    same = bounded_range(da3 * math.exp(0.002), da3, 0.05)
    assert float(((same / da3).log() - 0.002).abs().max()) < 1e-4       # tanh is linear near 0: decoder kept


def test_relabel_hook_moves_valid_pixels_and_keeps_gradients():
    from dwm_pano_4d.pipeline_relabel import DEFAULT_RELABEL, PanoGaussianDecoder4DRelabelPipeline
    torch.manual_seed(0)
    dirs = torch.nn.functional.normalize(torch.randn(N, 3), dim=-1)
    origins = torch.randn(N, 3) * 0.5
    pred = (torch.rand(1, V, H, W) * 40 + 2).requires_grad_(True)
    g = _gaussians(pred.detach(), origins, dirs)
    g = g.clone().requires_grad_(True)
    da3 = torch.rand(3, V, H, W) * 40 + 2
    da3[0, 0, 0, :4] = 100.0                                            # sky fill: must not move
    valid = torch.ones(3, V, H, W, dtype=torch.bool)
    valid[0, 1, 2, :3] = False                                          # invalid label: must not move
    fake = types.SimpleNamespace(gs4d_relabel=dict(DEFAULT_RELABEL), pixel_directions=dirs.reshape(V, H, W, 3))
    out, rep = PanoGaussianDecoder4DRelabelPipeline._train_gaussians(
        fake, 0, g, pred, da3, valid, origins.reshape(1, V, H, W, 3), 1.0)
    use = (valid[0] & (da3[0] < 99.0)).reshape(-1)
    rng = (out[:, 0:3] - origins).norm(dim=-1)
    ratio = (da3[0] / pred.detach()[0]).reshape(-1)[use].clamp(0.25, 4.0)   # rescale_gaussians' clamp
    assert float((rng[use] - pred.detach().reshape(-1)[use] * ratio).abs().max()) < 1e-3
    inside = ((da3[0] / pred.detach()[0]).reshape(-1)[use] - ratio).abs() < 1e-6
    assert int(inside.sum()) > N // 4
    assert float((rng[use][inside] - da3[0].reshape(-1)[use][inside]).abs().max()) < 1e-3
    assert torch.equal(out[~use], g[~use])
    assert torch.allclose(out[use, 11:14], g[use, 11:14] * ratio[:, None], rtol=1e-5)
    assert abs(rep["relabel_fraction"] - float(use.float().mean())) < 1e-6
    (out[:, 3:7].sum() + out[:, 11:14].sum()).backward()
    assert float(g.grad[:, 3:7].abs().min()) > 0.0                      # appearance still learns
    assert pred.grad is not None


def test_matches_test_time_scale_relabel():
    """bound 0 in training == gs_scene_v3 da3_mode "scale" (rescale_gaussians with the DA3 range)."""
    from dwm_pano_4d.pipeline_relabel import DEFAULT_RELABEL, PanoGaussianDecoder4DRelabelPipeline
    torch.manual_seed(1)
    dirs = torch.nn.functional.normalize(torch.randn(N, 3), dim=-1)
    origins = torch.randn(N, 3) * 0.5
    pred = torch.rand(1, V, H, W) * 40 + 2
    g = _gaussians(pred, origins, dirs, offset_scale=0.02)
    da3 = torch.rand(1, V, H, W) * 40 + 2
    valid = torch.ones(1, V, H, W, dtype=torch.bool)
    fake = types.SimpleNamespace(gs4d_relabel=dict(DEFAULT_RELABEL), pixel_directions=dirs.reshape(V, H, W, 3))
    out, _ = PanoGaussianDecoder4DRelabelPipeline._train_gaussians(
        fake, 0, g, pred, da3, valid, origins.reshape(1, V, H, W, 3), 1.0)
    ref, _ = rescale_gaussians(g, pred[0], da3[0], valid[0] & (da3[0] > 0.5) & (da3[0] < 99.0), origins)
    assert torch.equal(out, ref)


def test_default_hook_is_identity():
    from dwm_pano_4d.pipeline_depth import PanoGaussianDecoder4DDepthPipeline
    g = torch.randn(N, 14)
    out, rep = PanoGaussianDecoder4DDepthPipeline._train_gaussians(None, 0, g, None, None, None, None, 1.0)
    assert out is g and rep == {}
    extra, rep = PanoGaussianDecoder4DDepthPipeline._loo_extra_loss(None, None, None, None)
    assert extra is None and rep == {}


def test_no_offset_puts_centres_on_the_da3_ray_point():
    from dwm_pano_4d.pipeline_relabel import DEFAULT_RELABEL, PanoGaussianDecoder4DRelabelPipeline
    torch.manual_seed(2)
    dirs = torch.nn.functional.normalize(torch.randn(N, 3), dim=-1)
    origins = torch.randn(N, 3) * 0.5
    pred = (torch.rand(1, V, H, W) * 20 + 10).requires_grad_(True)
    g = _gaussians(pred.detach(), origins, dirs, offset_scale=0.3)
    da3 = pred.detach() * (1.0 + 0.2 * (torch.rand(1, V, H, W) - 0.5))   # within the 0.25-4 clamp
    da3[0, 1, :, :2] = 100.0                                            # sky: stays at the decoder range, on its ray
    valid = torch.ones(1, V, H, W, dtype=torch.bool)
    cfg = dict(DEFAULT_RELABEL, no_offset=True)
    fake = types.SimpleNamespace(gs4d_relabel=cfg, pixel_directions=dirs.reshape(V, H, W, 3))
    out, rep = PanoGaussianDecoder4DRelabelPipeline._train_gaussians(
        fake, 0, g, pred, da3, valid, origins.reshape(1, V, H, W, 3), 1.0)
    use = (da3[0] < 99.0).reshape(-1)
    expect = origins + dirs * torch.where(use, da3.reshape(-1), pred.detach().reshape(-1))[:, None]
    assert float((out[:, 0:3] - expect).abs().max()) < 1e-4
    assert rep["offset_norm"] > 0.1                                     # the dropped offset is reported
    out[:, 0:3].sum().backward()
    assert pred.grad is not None


def test_plane_offsets_drop_the_along_ray_part_and_bound_the_rest():
    from dwm_pano_4d.da3_depth import PIXEL_PITCH_RAD, plane_offsets
    torch.manual_seed(3)
    dirs = torch.nn.functional.normalize(torch.randn(N, 3), dim=-1)
    origins = torch.randn(N, 3) * 0.5
    r = torch.rand(N) * 40 + 2
    g = _gaussians(r, origins, dirs, offset_scale=0.5)                  # large offsets, any direction
    out = plane_offsets(g, origins, dirs, r, 1.0)
    off = out[:, 0:3] - (origins + dirs * r[:, None])
    assert float((off * dirs).sum(-1).abs().max()) < 1e-4               # depth along the ray = r
    assert bool((off.norm(dim=-1) <= PIXEL_PITCH_RAD * r + 1e-5).all())
    tiny = _gaussians(r, origins, dirs)                                 # zero offset stays zero
    assert torch.allclose(plane_offsets(tiny, origins, dirs, r, 1.0), tiny, atol=1e-5)
    assert torch.equal(out[:, 3:], g[:, 3:])


def test_plane_mode_in_training_matches_test_time_order():
    """training: plane_offsets on the decoder range, then the DA3 slide == gs_scene_v3 decode with plane_offset_px."""
    from dwm_pano_4d.da3_depth import plane_offsets
    from dwm_pano_4d.pipeline_relabel import DEFAULT_RELABEL, PanoGaussianDecoder4DRelabelPipeline
    torch.manual_seed(4)
    dirs = torch.nn.functional.normalize(torch.randn(N, 3), dim=-1)
    origins = torch.randn(N, 3) * 0.5
    pred = torch.rand(1, V, H, W) * 30 + 5
    g = _gaussians(pred, origins, dirs, offset_scale=0.05)
    da3 = pred * (1.0 + 0.3 * (torch.rand(1, V, H, W) - 0.5))
    valid = torch.ones(1, V, H, W, dtype=torch.bool)
    fake = types.SimpleNamespace(gs4d_relabel=dict(DEFAULT_RELABEL, plane_offset_px=1.0),
                                 pixel_directions=dirs.reshape(V, H, W, 3))
    out, _ = PanoGaussianDecoder4DRelabelPipeline._train_gaussians(
        fake, 0, g, pred, da3, valid, origins.reshape(1, V, H, W, 3), 1.0)
    ref, _ = rescale_gaussians(plane_offsets(g, origins, dirs, pred, 1.0), pred[0], da3[0], valid[0], origins)
    assert torch.equal(out, ref)
