"""Contracts for the blendable pano stitch and the two new seam instruments.

The stitch is on the frozen evaluation path, so the first thing pinned here is
that the default is bit-identical to the hard butt-joint every reported number
was produced with. The instruments are then checked against synthetic panoramas
whose answer is known by construction, because a diagnostic that cannot tell a
narrow joint artefact from a broad sector-wide one would not settle the
question it was built for.
"""
import math
import os

import pytest
import torch

from dwm_pano.metrics.seam_profile import PanoSeamProfile
from dwm_pano.metrics.temporal_smoothness import PanoTemporalSmoothness
from dwm_pano.pano_stitch import PANO_OVERLAP_W, stitch_pano_halves


def _halves(batch=1, frames=3, channels=3, height=8, width=2304, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(
        batch, frames, 2, channels, height, width, generator=generator)


def _hard_cut(x):
    return torch.cat(
        [x[:, :, 0], x[:, :, 1, :, :, PANO_OVERLAP_W:]], dim=-1).unsqueeze(2)


def test_default_stitch_is_bit_identical_to_the_hard_cut():
    x = _halves()
    assert torch.equal(stitch_pano_halves(x, blend_width=0), _hard_cut(x))


def test_env_default_is_off(monkeypatch):
    monkeypatch.delenv("PANDWM_PANO_BLEND_W", raising=False)
    x = _halves()
    assert torch.equal(stitch_pano_halves(x), _hard_cut(x))


def test_blending_preserves_width_so_the_azimuth_mapping_is_unchanged():
    x = _halves()
    reference_width = _hard_cut(x).shape[-1]
    for blend_width in (1, 32, 128, PANO_OVERLAP_W):
        out = stitch_pano_halves(x, blend_width=blend_width)
        assert out.shape[-1] == reference_width
        assert out.shape[:-1] == _hard_cut(x).shape[:-1]


def test_blending_only_touches_the_overlap_columns():
    x = _halves()
    blend_width = 32
    out = stitch_pano_halves(x, blend_width=blend_width)
    hard = _hard_cut(x)
    seam = x.shape[-1]
    assert torch.equal(out[..., :seam - blend_width], hard[..., :seam - blend_width])
    assert torch.equal(out[..., seam:], hard[..., seam:])
    assert not torch.equal(
        out[..., seam - blend_width:seam], hard[..., seam - blend_width:seam])


def test_blending_consistent_halves_is_nearly_a_noop():
    """Real frames slice one source image, so the halves agree in the overlap.

    A cross-fade between two agreeing signals must return that signal -- if it
    did not, the metric change would be confounded by the real arm moving.
    """
    x = _halves()
    x[:, :, 1, ..., :PANO_OVERLAP_W] = x[:, :, 0, ..., -PANO_OVERLAP_W:]
    out = stitch_pano_halves(x, blend_width=PANO_OVERLAP_W)
    assert torch.allclose(out, _hard_cut(x), atol=1e-6)


def test_blending_removes_the_joint_discontinuity():
    """A step between the halves must survive as a ramp, not as a cliff."""
    x = torch.zeros(1, 3, 2, 1, 4, 2304)
    x[:, :, 1] = 1.0
    hard = _hard_cut(x)
    blended = stitch_pano_halves(x, blend_width=64)
    seam = 2304

    def biggest_step(pano):
        return float((pano[..., 1:] - pano[..., :-1]).abs().max())

    assert biggest_step(hard) == pytest.approx(1.0, abs=1e-6)
    assert biggest_step(blended) < 0.1
    # The transition still lands on the joint, so the column-to-azimuth
    # mapping is untouched.
    assert float(blended[0, 0, 0, 0, 0, seam - 1]) > 0.9
    assert float(blended[0, 0, 0, 0, 0, seam - 64]) < 0.1


def _pano_with_seam(jump, width=PanoSeamProfile.PANO_W_V1, seed=0):
    """Flat-noise panorama with an extra discontinuity at the LR joint."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 4, 3, 8, width, generator=generator) * 0.01
    x[..., PanoSeamProfile.SEAM_LR:] += jump
    return x


def test_seam_profile_recovers_an_injected_joint():
    metric = PanoSeamProfile()
    metric.update(_pano_with_seam(0.0, seed=1), real=True)
    metric.update(_pano_with_seam(0.5, seed=1), real=False)
    out = metric.compute()

    # The real arm has no joint, the fake arm has a large one.
    assert out["seam_lr_real"] < 3.0
    assert out["seam_lr_fake"] > 20.0
    assert out["seam_lr_excess"] > 5.0
    # One injected column must read as a narrow peak, not a broad hump.
    assert out["seam_lr_width"] <= 3.0


def test_seam_profile_calls_a_broad_hump_broad():
    """A sector-wide elevation must not be reported as a narrow joint."""
    metric = PanoSeamProfile()
    generator = torch.Generator().manual_seed(2)
    real = torch.rand(1, 4, 3, 8, PanoSeamProfile.PANO_W_V1,
                      generator=generator) * 0.01
    fake = real.clone()
    lo = PanoSeamProfile.SEAM_LR - 300
    hi = PanoSeamProfile.SEAM_LR + 300
    fake[..., lo:hi] = fake[..., lo:hi] * 4.0
    metric.update(real, real=True)
    metric.update(fake, real=False)
    out = metric.compute()

    assert out["seam_lr_width"] >= 2 * PanoSeamProfile.PEAK_HALFWIDTH - 1
    # sect06 is the front sector, which is where the hump was injected.
    assert out["sect06"] > 2.0


def test_seam_profile_measures_the_wrap_joint_circularly():
    metric = PanoSeamProfile()
    width = PanoSeamProfile.PANO_W_V1
    generator = torch.Generator().manual_seed(3)
    real = torch.rand(1, 4, 3, 8, width, generator=generator) * 0.01
    fake = real.clone()
    # Raise the whole right side, which makes column 0 and column W-1 differ.
    fake[..., width // 2:] += 0.5
    metric.update(real, real=True)
    metric.update(fake, real=False)
    out = metric.compute()
    assert out["seam_wrap_fake"] > 20.0
    assert out["seam_wrap_excess"] > 5.0


def test_seam_profile_overlap_disagreement():
    metric = PanoSeamProfile()
    consistent = _halves(seed=4)
    consistent[:, :, 1, ..., :PANO_OVERLAP_W] = \
        consistent[:, :, 0, ..., -PANO_OVERLAP_W:]
    inconsistent = _halves(seed=5)

    metric.update(stitch_pano_halves(consistent, 0)[:, :, 0], real=True)
    metric.update(stitch_pano_halves(inconsistent, 0)[:, :, 0], real=False)
    metric.update_halves(consistent, real=True)
    metric.update_halves(inconsistent, real=False)
    out = metric.compute()

    assert out["overlap_real"] == pytest.approx(0.0, abs=1e-6)
    assert out["overlap_fake"] > 0.1
    assert out["overlap_ratio"] > 100.0


def test_seam_profile_rejects_the_wrong_width():
    metric = PanoSeamProfile()
    with pytest.raises(ValueError):
        metric.update(torch.zeros(1, 3, 3, 8, 4096), real=True)


def test_smoothness_flags_over_smoothing():
    """A temporally and spatially blurred fake must read below 1 on both."""
    generator = torch.Generator().manual_seed(6)
    real = torch.rand(2, 5, 3, 16, 64, generator=generator)
    # Box filter over time and width -- the signature of over-smoothing.
    fake = real.clone()
    fake[:, 1:-1] = (real[:, :-2] + real[:, 1:-1] + real[:, 2:]) / 3.0
    fake = (fake + fake.roll(1, dims=-1) + fake.roll(-1, dims=-1)) / 3.0

    metric = PanoTemporalSmoothness()
    metric.update(real, real=True)
    metric.update(fake, real=False)
    out = metric.compute()

    assert out["tdelta_ratio"] < 1.0
    assert out["sgrad_ratio"] < 1.0


def test_smoothness_flags_flicker_the_other_way():
    """Added temporal noise must raise jitter, i.e. the opposite defect."""
    generator = torch.Generator().manual_seed(7)
    real = torch.rand(2, 5, 3, 16, 64, generator=generator) * 0.1 + 0.5
    fake = real + torch.rand(
        2, 5, 3, 16, 64, generator=generator) * 0.2 - 0.1

    metric = PanoTemporalSmoothness()
    metric.update(real, real=True)
    metric.update(fake, real=False)
    out = metric.compute()

    assert out["tjitter_ratio"] > 1.0


def test_smoothness_is_identity_on_identical_arms():
    generator = torch.Generator().manual_seed(8)
    x = torch.rand(2, 5, 3, 16, 64, generator=generator)
    metric = PanoTemporalSmoothness()
    metric.update(x, real=True)
    metric.update(x, real=False)
    out = metric.compute()
    for axis in ("tdelta", "tjitter", "sgrad"):
        assert out["{}_ratio".format(axis)] == pytest.approx(1.0, abs=1e-6)


def test_frame_profile_flat_when_error_is_uniform():
    from dwm_pano.metrics.frame_profile import PanoFrameProfile
    generator = torch.Generator().manual_seed(11)
    real = torch.rand(2, 16, 3, 8, 32, generator=generator)
    fake = real + 0.1
    metric = PanoFrameProfile()
    metric.update(fake, real)
    out = metric.compute()
    assert out["frames"] == 16.0
    assert out["pl1_growth"] == pytest.approx(1.0, abs=1e-4)
    assert abs(out["pl1_slope"]) < 1e-5


def test_frame_profile_detects_growing_drift():
    from dwm_pano.metrics.frame_profile import PanoFrameProfile
    generator = torch.Generator().manual_seed(12)
    real = torch.rand(2, 16, 3, 8, 32, generator=generator)
    ramp = torch.linspace(0.0, 0.3, 16).view(1, 16, 1, 1, 1)
    fake = real + ramp
    metric = PanoFrameProfile()
    metric.update(fake, real)
    out = metric.compute()
    assert out["pl1_growth"] > 5.0
    assert out["pl1_slope"] > 0.01


def test_frame_profile_rejects_unpaired_shapes():
    from dwm_pano.metrics.frame_profile import PanoFrameProfile
    metric = PanoFrameProfile()
    with pytest.raises(ValueError):
        metric.update(torch.zeros(1, 4, 3, 8, 32), torch.zeros(1, 5, 3, 8, 32))


def test_motion_dispersion_separates_over_and_under_spread():
    """The mean can match while the spread across clips does not."""
    # real: clips with a wide range of motion energies
    real = torch.stack([
        torch.rand(5, 3, 8, 32) * scale
        for scale in (0.2, 0.5, 0.8, 1.1)])
    # fake: every clip has the same middling motion -> under-dispersed
    fake = torch.stack([
        torch.rand(5, 3, 8, 32) * 0.65 for _ in range(4)])

    metric = PanoTemporalSmoothness()
    metric.update(real, real=True)
    metric.update(fake, real=False)
    out = metric.compute()
    assert out["tdelta_std_real"] > out["tdelta_std_fake"]
    assert out["tdelta_std_ratio"] < 1.0
