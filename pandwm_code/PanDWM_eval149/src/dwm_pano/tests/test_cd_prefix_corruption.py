import torch

from dwm_pano.pipelines.ctsd_pano_consistency_distillation import (
    corrupt_reference_context,
    flow_corrupt_reference_context,
)


def test_zero_prefix_corruption_is_exact_noop():
    latents = torch.randn(1, 4, 2, 3, 2, 2)
    mask = torch.zeros(1, 4, 2, 1, 1, 1, dtype=torch.bool)
    noise = torch.randn_like(latents)
    assert corrupt_reference_context(latents, mask, noise, 0.0) is latents


def test_prefix_corruption_changes_only_visible_history():
    latents = torch.zeros(1, 4, 1, 1, 1, 1)
    noise = torch.ones_like(latents)
    mask = torch.zeros(1, 4, 1, 1, 1, 1, dtype=torch.bool)
    mask[:, :2] = True
    result = corrupt_reference_context(latents, mask, noise, 0.25)
    torch.testing.assert_close(result[:, :2], torch.full_like(result[:, :2], 0.25))
    torch.testing.assert_close(result[:, 2:], latents[:, 2:])


def test_flow_history_noise_preserves_future_and_labels_exact_flow_value():
    latents = torch.full((2, 4, 1, 1, 1, 1), 0.75)
    noise = torch.full_like(latents, -0.25)
    mask = torch.zeros(2, 4, 1, 1, 1, 1, dtype=torch.bool)
    mask[:, :2] = True
    sigma = torch.tensor([0.0, 0.2]).view(2, 1, 1, 1, 1, 1)
    result = flow_corrupt_reference_context(latents, mask, noise, sigma)
    torch.testing.assert_close(result[0], latents[0])
    torch.testing.assert_close(result[1, :2],
                               torch.full_like(result[1, :2], 0.55))
    torch.testing.assert_close(result[:, 2:], latents[:, 2:])
