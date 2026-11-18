"""Per-camera FVD decomposition must be additive-only.

The pooled FVD is the number every previously reported run used, so the new
per-camera accumulators must not perturb it. These tests pin that, plus the
patch-order contract that makes the decomposition attributable to the right
camera.
"""
import numpy as np
import pytest
import torch

from dwm_pano.metrics.fvd_pano import PanoPatchFrechetVideoDistance as PanoFVD


class _FakeI3D(torch.nn.Module):
    """Deterministic stand-in for I3D: mean-pools to `num_classes` logits.

    Avoids needing the 400-class checkpoint while preserving the one property
    these tests care about -- distinct patches produce distinct features.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):                     # x: (N, C, T, H, W)
        pooled = x.mean(dim=(2, 3, 4))        # (N, C)
        repeats = -(-self.num_classes // pooled.shape[1])
        return pooled.repeat(1, repeats)[:, :self.num_classes]


def _metric(num_classes=8, monkeypatch=None):
    metric = PanoFVD.__new__(PanoFVD)
    torch.nn.Module.__init__(metric)
    # Re-run the real __init__ body minus the checkpoint load.
    import torchmetrics
    torchmetrics.Metric.__init__(metric)
    metric.inception = _FakeI3D(num_classes)
    metric.sequence_count = -1
    mx = (num_classes, num_classes)
    for side in ("real", "fake"):
        metric.add_state(f"{side}_features_sum",
                         torch.zeros(num_classes).double(),
                         dist_reduce_fx="sum")
        metric.add_state(f"{side}_features_cov_sum",
                         torch.zeros(mx).double(), dist_reduce_fx="sum")
        metric.add_state(f"{side}_features_num_samples",
                         torch.tensor(0).long(), dist_reduce_fx="sum")
        metric.add_state(f"{side}_cam_sum",
                         torch.zeros(len(PanoFVD.CAM_NAMES),
                                     num_classes).double(),
                         dist_reduce_fx="sum")
        metric.add_state(f"{side}_cam_cov_sum",
                         torch.zeros(len(PanoFVD.CAM_NAMES), *mx).double(),
                         dist_reduce_fx="sum")
        metric.add_state(f"{side}_cam_num_samples",
                         torch.zeros(len(PanoFVD.CAM_NAMES)).long(),
                         dist_reduce_fx="sum")
    return metric


def _frames(batch, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(
        batch, 12, 3, 8, PanoFVD.PANO_W_V1, generator=generator)


def test_per_camera_states_do_not_disturb_the_pooled_value():
    pooled = []
    for _ in range(2):
        metric = _metric()
        metric.update(_frames(4, 0), real=True)
        metric.update(_frames(4, 1), real=False)
        pooled.append(float(metric.compute()))
    assert pooled[0] == pooled[1]

    # And the pooled accumulators must equal the per-camera ones summed,
    # which is what proves both were fed the same features.
    metric = _metric()
    metric.update(_frames(4, 0), real=True)
    metric.update(_frames(4, 1), real=False)
    for side in ("real", "fake"):
        assert torch.allclose(
            getattr(metric, f"{side}_features_sum"),
            getattr(metric, f"{side}_cam_sum").sum(dim=0))
        assert torch.allclose(
            getattr(metric, f"{side}_features_cov_sum"),
            getattr(metric, f"{side}_cam_cov_sum").sum(dim=0))
        assert int(getattr(metric, f"{side}_features_num_samples")) == \
            int(getattr(metric, f"{side}_cam_num_samples").sum())


def test_per_camera_returns_all_six_named_regions():
    metric = _metric()
    metric.update(_frames(4, 0), real=True)
    metric.update(_frames(4, 1), real=False)
    result = metric.compute_per_camera()
    assert list(result) == list(PanoFVD.CAM_NAMES)
    assert all(np.isfinite(v) for v in result.values())


def test_per_camera_localises_a_single_camera_perturbation():
    """Corrupting only CAM_FRONT's columns must move CAM_FRONT the most.

    This is the property the decomposition is used for: attributing a pooled
    change to a region.
    """
    real = _frames(6, 0)
    fake = real.clone()
    lo, hi = PanoFVD.PANO_PATCH_COLS[2], PanoFVD.PANO_PATCH_COLS[3]
    fake[..., lo:hi] = (fake[..., lo:hi] * 0.2 + 0.5).clamp(0, 1)

    metric = _metric()
    metric.update(real, real=True)
    metric.update(fake, real=False)
    result = metric.compute_per_camera()
    worst = max(result, key=result.get)
    assert worst == "CAM_FRONT", f"attributed to {worst}: {result}"


def test_per_camera_raises_when_a_camera_has_too_few_samples():
    metric = _metric()
    metric.update(_frames(1, 0), real=True)
    metric.update(_frames(1, 1), real=False)
    with pytest.raises(RuntimeError, match="at least two samples"):
        metric.compute_per_camera()
