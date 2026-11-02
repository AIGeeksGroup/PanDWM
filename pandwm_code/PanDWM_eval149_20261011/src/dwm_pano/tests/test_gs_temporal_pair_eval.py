"""CPU-only regression tests for fair paired temporal GS reductions."""

import torch

from dwm_pano.evaluate_gs_temporal_pair import (
    finalize,
    new_accumulator,
    update_cross_render_reductions,
)


def _rgb_from_per_pixel(values):
    values = torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1)
    return values.expand(-1, 3, -1, -1).clone()


def test_headline_xrender_gt_uses_fixed_target_support_not_visibility():
    accumulator = new_accumulator()
    target_rgb = _rgb_from_per_pixel([0.0, 0.0, 0.0, 0.0])
    rendered = {
        "baseline": _rgb_from_per_pixel([0.5, 0.5, 0.5, 0.5]),
        # The candidate is perfect only on the one pixel it declares visible.
        "candidate": _rgb_from_per_pixel([0.0, 1.0, 1.0, 1.0]),
    }
    alphas = {
        "baseline": torch.ones(1, 1, 4),
        "candidate": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
    }
    visibility = {
        "baseline": torch.ones(1, 1, 4, dtype=torch.bool),
        "candidate": torch.tensor([[[True, False, False, False]]]),
    }
    target_support = torch.ones(1, 1, 4, dtype=torch.bool)
    own_rgb = {
        "baseline": torch.zeros_like(target_rgb),
        "candidate": torch.zeros_like(target_rgb),
    }

    update_cross_render_reductions(
        accumulator,
        rendered,
        alphas,
        visibility,
        target_support,
        target_rgb,
        own_rgb,
        alpha_threshold=0.05,
    )
    result = finalize(accumulator)

    # A candidate-selected reduction would report zero. The fixed-support
    # headline correctly retains the three hidden one-valued errors.
    assert result["candidate"]["XRenderGT-L1"] == 0.75
    assert result["candidate"]["XRenderGT-L1-render-visible"] == 0.0
    assert result["candidate"]["XRenderGT-L1-common"] == 0.0
    assert result["baseline"]["XRenderGT-L1"] == 0.5
    assert result["candidate_minus_baseline"]["XRenderGT-L1"] == 0.25

    assert result["alpha_coverage_on_target"]["baseline"] == 1.0
    assert result["alpha_coverage_on_target"]["candidate"] == 0.25
    assert result["visibility_recall_on_target"]["baseline"] == 1.0
    assert result["visibility_recall_on_target"]["candidate"] == 0.25
    assert result["visibility_recall_on_target"]["common"] == 0.25


def test_fixed_support_excludes_pixels_outside_target_static_validity():
    accumulator = new_accumulator()
    target_rgb = _rgb_from_per_pixel([0.0, 0.0, 0.0])
    rendered = {
        "baseline": _rgb_from_per_pixel([0.0, 0.0, 1.0]),
        "candidate": _rgb_from_per_pixel([0.0, 0.0, 1.0]),
    }
    alphas = {
        name: torch.ones(1, 1, 3) for name in ("baseline", "candidate")
    }
    visibility = {
        name: torch.tensor([[[True, False, False]]])
        for name in ("baseline", "candidate")
    }
    # The last pixel represents invalid/dynamic target content and must not
    # enter the fixed static-valid headline despite its large RGB error.
    target_support = torch.tensor([[[True, True, False]]])
    own_rgb = {
        name: torch.zeros_like(target_rgb)
        for name in ("baseline", "candidate")
    }

    update_cross_render_reductions(
        accumulator,
        rendered,
        alphas,
        visibility,
        target_support,
        target_rgb,
        own_rgb,
        alpha_threshold=0.05,
    )
    result = finalize(accumulator)
    assert result["counts"]["target_static_valid_pixels"] == 2
    assert result["coverage"]["target_static_valid"] == 2 / 3
    assert result["baseline"]["XRenderGT-L1"] == 0.0
    assert result["candidate"]["XRenderGT-L1"] == 0.0


def test_empty_common_visibility_does_not_suppress_fixed_support_metric():
    accumulator = new_accumulator()
    target_rgb = _rgb_from_per_pixel([0.0, 0.0])
    rendered = {
        "baseline": _rgb_from_per_pixel([0.25, 0.25]),
        "candidate": _rgb_from_per_pixel([0.75, 0.75]),
    }
    alphas = {
        name: torch.zeros(1, 1, 2) for name in ("baseline", "candidate")
    }
    visibility = {
        name: torch.zeros(1, 1, 2, dtype=torch.bool)
        for name in ("baseline", "candidate")
    }
    support = torch.ones(1, 1, 2, dtype=torch.bool)
    own_rgb = {
        name: torch.zeros_like(target_rgb)
        for name in ("baseline", "candidate")
    }

    update_cross_render_reductions(
        accumulator,
        rendered,
        alphas,
        visibility,
        support,
        target_rgb,
        own_rgb,
        alpha_threshold=0.05,
    )
    result = finalize(accumulator)
    assert result["counts"]["common_visible_pixels"] == 0
    assert result["baseline"]["XRenderGT-L1"] == 0.25
    assert result["candidate"]["XRenderGT-L1"] == 0.75
    assert result["baseline"]["XRenderGT-L1-common"] == 0.0
    assert result["candidate"]["XRenderGT-L1-common"] == 0.0
