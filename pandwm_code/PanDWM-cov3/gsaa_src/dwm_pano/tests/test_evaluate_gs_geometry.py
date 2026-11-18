"""Regression tests for geometry-aware GS holdout evaluation."""
from types import SimpleNamespace

import torch

from dwm_pano.evaluate_gs import (
    RGBMetricAccumulator,
    make_grid_pair,
    validate_protocol,
)


class _DummyLPIPS(torch.nn.Module):
    def reset(self):
        return None

    def forward(self, prediction, target):
        return (prediction - target).abs().mean()


def test_rgb_metrics_use_fixed_stitch_validity_and_protocol_label():
    accumulator = RGBMetricAccumulator(
        _DummyLPIPS(), protocol_label="pano4114x256-real6stitch")
    rendered = torch.full((1, 3, 11, 11), 0.5)
    target = torch.ones_like(rendered)
    rendered[:, :, 0, 0] = 0.0
    valid = torch.ones(1, 11, 11, dtype=torch.bool)
    valid[:, 0, 0] = False
    accumulator.update(rendered, target, frame_index=2, valid=valid)

    result = accumulator.result("holdout")
    key = "PSNR-pano4114x256-real6stitch-holdout"
    assert torch.isclose(
        torch.tensor(result[key]), torch.tensor(6.0206), atol=1e-4)
    # The invalid cell is replaced by target before LPIPS.
    assert torch.isclose(
        torch.tensor(
            result["LPIPS-pano4114x256-real6stitch-holdout"]),
        torch.tensor(0.5 * 120.0 / 121.0), atol=1e-7)


def test_grid_preserves_total_width_for_one_pano_or_six_views():
    pano_pair = make_grid_pair(
        torch.rand(1, 3, 16, 256), torch.rand(1, 3, 16, 256))
    six_view_pair = make_grid_pair(
        torch.rand(6, 3, 16, 32), torch.rand(6, 3, 16, 32))
    assert pano_pair.shape == (3, 150, 1200)
    assert six_view_pair.shape == (3, 150, 1200)


def test_g2_protocol_guard_accepts_real6_independent_geometry():
    training = {
        "supervision_geometry": "nuscenes6_onebev_stitch",
        "stitch_projection_model": "onebev_legacy",
        "source_camera_render_height": 360,
        "source_camera_render_width": 640,
        "range_dir": "/depth",
        "range_hole_fill_m": 100.0,
        "evaluation_use_temporal_context": False,
    }
    decoder = {
        "use_range_prior": False,
        "use_camera_origins": True,
        "ray_embedding_mode": "plucker",
    }
    pipeline = SimpleNamespace(
        training_config=training,
        uses_nuscenes6_stitch=True,
        perceptual_metric=object(),
    )
    validate_protocol(
        {"pipeline": {"decoder": decoder}}, pipeline)
