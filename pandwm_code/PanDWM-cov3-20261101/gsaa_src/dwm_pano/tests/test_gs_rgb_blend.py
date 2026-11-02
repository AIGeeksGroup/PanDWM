import torch

from dwm_pano.evaluate_gs_decode_compare import blend_gaussian_rgb


def test_zero_blend_preserves_checkpoint_output_object():
    gaussians = torch.rand(1, 8, 14)
    halves = torch.rand(1, 2, 3, 2, 2)
    assert blend_gaussian_rgb(gaussians, halves, 0.0) is gaussians


def test_full_blend_changes_only_gaussian_rgb():
    gaussians = torch.rand(1, 8, 14)
    halves = torch.rand(1, 2, 3, 2, 2)
    result = blend_gaussian_rgb(gaussians, halves, 1.0)
    expected_rgb = halves.permute(0, 1, 3, 4, 2).reshape(1, 8, 3)
    torch.testing.assert_close(result[..., 3:6], expected_rgb)
    torch.testing.assert_close(result[..., :3], gaussians[..., :3])
    torch.testing.assert_close(result[..., 6:], gaussians[..., 6:])
