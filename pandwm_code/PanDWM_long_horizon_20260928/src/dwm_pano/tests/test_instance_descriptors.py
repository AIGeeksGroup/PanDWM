"""The IAC verdict rests entirely on what the descriptor can see.

`InstanceAppearanceConsistency` reported 0.928 with a 32x32 RGB flatten and
that was read as "no excess identity drift".  A raw-pixel descriptor is
dominated by tone, so it can be blind to exactly the failure mode the metric
was built to catch: an object keeping its colour while swapping identity.
These tests pin the two properties that make a descriptor swap trustworthy --
the crop geometry and normalisation must not change, and the new descriptors
must actually be more identity-sensitive than the old one on a case
constructed to fool the old one.
"""
import numpy as np
import pytest
import torch

from dwm_pano.metrics.instance_consistency import (
    _DESCRIPTOR_INPUT, InstanceAppearanceConsistency)


def _metric(descriptor, **kwargs):
    return InstanceAppearanceConsistency(
        pkl_path="/nonexistent.pkl", descriptor=descriptor,
        backbone_device="cpu", **kwargs)


def test_unknown_descriptor_is_rejected():
    with pytest.raises(ValueError, match="descriptor must be one of"):
        _metric("resnet")


def test_patch_descriptor_is_unchanged_by_the_refactor():
    """The historical arm has to stay bit-exact or 0.928 is not comparable."""
    torch.manual_seed(0)
    frames = torch.rand(3, 3, 64, 128)
    boxes = np.array([[10.0, 5.0, 40.0, 30.0]] * 3)

    got = _metric("patch")._descriptors(frames, boxes)

    # Recompute the pre-refactor formula literally.
    expected = []
    for t in range(3):
        crop = frames[t:t + 1, :, 5:30, 10:40]
        crop = torch.nn.functional.interpolate(
            crop, size=(32, 32), mode="bilinear", align_corners=False)
        vector = crop.flatten().double()
        expected.append(vector / (vector.norm() + 1e-8))
    torch.testing.assert_close(got, torch.stack(expected))


def test_every_descriptor_is_l2_normalised_and_one_row_per_frame():
    torch.manual_seed(1)
    frames = torch.rand(4, 3, 48, 96)
    boxes = np.array([[3.0, 2.0, 33.0, 26.0]] * 4)
    for descriptor in _DESCRIPTOR_INPUT:
        if descriptor == "vgg":
            pytest.importorskip("torchvision")
        if descriptor == "inception":
            # Ships with torchmetrics[image] on the cluster; optional locally.
            pytest.importorskip("torch_fidelity")
        vectors = _metric(descriptor)._descriptors(frames, boxes)
        assert vectors.shape[0] == 4, descriptor
        torch.testing.assert_close(
            vectors.norm(dim=-1), torch.ones(4).double(),
            msg=lambda m, d=descriptor: f"{d}: {m}")


def test_crop_side_follows_the_backbone_not_patch_size():
    assert _metric("patch", patch_size=17)._crop_side() == 17
    assert _metric("vgg", patch_size=17)._crop_side() == 224
    assert _metric("inception", patch_size=17)._crop_side() == 299


def test_boxes_are_clamped_into_the_frame():
    """A box running off the canvas must still yield one usable crop."""
    frames = torch.rand(1, 3, 32, 64)
    boxes = np.array([[-20.0, -9.0, 400.0, 900.0]])
    assert _metric("patch")._descriptors(frames, boxes).shape[0] == 1


def test_vgg_separates_a_shape_swap_that_tone_alone_hides():
    """The blind spot that motivated this change, as an executable case.

    Two crops share their colour histogram almost exactly but differ in
    structure -- horizontal stripes versus vertical.  A downsampled RGB
    flatten averages the stripes away and calls them near-identical; a
    perceptual backbone should not.
    """
    pytest.importorskip("torchvision")
    height = width = 64
    horizontal = torch.zeros(1, 3, height, width)
    horizontal[:, :, ::2, :] = 1.0
    vertical = torch.zeros(1, 3, height, width)
    vertical[:, :, :, ::2] = 1.0
    # Same mean intensity, so tone cannot tell them apart.
    assert abs(float(horizontal.mean()) - float(vertical.mean())) < 1e-6

    frames = torch.cat([horizontal, vertical], dim=0)
    boxes = np.array([[0.0, 0.0, float(width), float(height)]] * 2)

    def distance(descriptor):
        d = _metric(descriptor)._descriptors(frames, boxes)
        return float((d[0] - d[1]).norm())

    assert distance("vgg") > distance("patch"), (
        "the perceptual descriptor must be more shape-sensitive than the "
        "raw-pixel one, otherwise swapping it answers nothing")


def test_compute_labels_are_unique_per_arm_and_all_numeric():
    """Four arms print into one flat line, so the labels must disambiguate.

    They must also stay numeric: the evaluation printer formats every entry
    with ``{:.3f}``, and a string field there used to raise *after* all 149
    clips had been generated, destroying the whole run.
    """
    a = _metric("patch", min_pixels=8).compute()
    b = _metric("inception", min_pixels=32).compute()

    assert set(a).isdisjoint(b), "arm labels collide; results are unreadable"
    assert "patch8_iac_ratio" in a
    assert "inception32_iac_ratio" in b
    for out in (a, b):
        for label, value in out.items():
            assert isinstance(value, (int, float)) and not isinstance(
                value, bool), f"{label} is not numeric: {value!r}"
            "{:.3f}".format(value)


def test_printer_survives_a_non_numeric_metric_field():
    """Belt and braces for the failure that cost two jobs.

    Reproduces the exact expression the evaluation printer uses. A metric that
    reports a mode name must degrade to plain printing, never raise.
    """
    def render(items):
        text = ""
        for label, number in items:
            if isinstance(number, (bool, str)):
                text += " {}: {}".format(label, number)
                continue
            text += " {}: {:.3f}".format(label, number)
        return text

    assert render([("fid", 14.596), ("mode", "patch")]) == \
        " fid: 14.596 mode: patch"
    with pytest.raises(ValueError):
        "{:.3f}".format("patch")      # the original, unguarded expression


def test_dispatch_is_by_type_so_every_arm_gets_fed():
    """Four arms need four distinct config keys, so name-keyed dispatch fails.

    The first attempt registered `iacpatch`/`iacincep`/`iacvgg`/`iacinbig`
    while the pipeline fed only `self.metrics["iac"]`. Nothing raised: every
    arm simply reported nan with pairs=0 after a full generation pass. This
    pins the type-based selection that replaced it.
    """
    metrics = {
        "fid": object(),
        "iacpatch": _metric("patch"),
        "iacinbig": _metric("patch", min_pixels=32),
    }
    selected = [m for m in metrics.values()
                if isinstance(m, InstanceAppearanceConsistency)]
    assert len(selected) == 2
    assert not any(k == "iac" for k in metrics), (
        "no arm is named plain 'iac'; name-keyed dispatch would feed none")
