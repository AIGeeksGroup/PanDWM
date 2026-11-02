"""Tests for the band-limited six-real-camera FVD.

`pytorch_i3d` lives in the TATS submodule, which is not vendored in this
workspace, so these tests install a deterministic stub in its place.  That is
deliberate rather than a compromise: what has to be proven here is that the
bridge metric reaches the *same* accumulation path as the pano metric and
labels its six cameras the same way, not that I3D computes the right numbers
(the checkpoint is pinned by sha256 elsewhere).
"""
import sys
import types

import pytest
import torch


def _install_i3d_stub():
    """A stub InceptionI3d whose features are a linear function of input.

    Linear-in-input means two different sets of six views can never
    accidentally produce identical features, so an ordering bug shows up.
    """
    if "pytorch_i3d" in sys.modules:
        return
    module = types.ModuleType("pytorch_i3d")

    class InceptionI3d(torch.nn.Module):
        def __init__(self, num_classes=400):
            super().__init__()
            self.num_classes = num_classes
            self.register_buffer(
                "weight", torch.arange(num_classes).double() + 1.0)

        def forward(self, x):
            # x is (N, C, T, H, W); collapse everything but N.
            flat = x.flatten(1).double()
            base = flat.mean(dim=1, keepdim=True)
            spread = flat.std(dim=1, keepdim=True)
            return (base + spread * self.weight[None] / self.num_classes) \
                .to(torch.float32)

    module.InceptionI3d = InceptionI3d
    sys.modules["pytorch_i3d"] = module


_install_i3d_stub()

from dwm_pano.metrics.fvd_pano import (                        # noqa: E402
    PanoPatchFrechetVideoDistance,
)
from dwm_pano.metrics.nuscenes6_fvd import (                   # noqa: E402
    NuScenes6BandFrechetVideoDistance,
)
from scripts.horizondrive.evaluate_real_vs_real import (       # noqa: E402
    split_indices,
)


@pytest.fixture
def checkpoint(tmp_path):
    """A state dict the stub will accept."""
    path = tmp_path / "i3d_stub.pt"
    torch.save({"weight": torch.arange(400).double() + 1.0}, path)
    return str(path)


def _views(batch, frames, height, width, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(
        batch, frames, 6, 3, height, width, generator=generator)


def test_reuses_the_parent_accumulation_path(checkpoint):
    """The whole point of the subclass: one shared I3D path.

    If a future edit gives the bridge metric its own copy of the normalize /
    resize / feature code, the two metrics stop being the same instrument and
    this assertion is the tripwire.
    """
    assert NuScenes6BandFrechetVideoDistance.update \
        is not PanoPatchFrechetVideoDistance.update
    assert "_accumulate_patches" not in \
        NuScenes6BandFrechetVideoDistance.__dict__, \
        "the subclass must inherit _accumulate_patches, not redefine it"
    source = __import__(
        "inspect").getsource(NuScenes6BandFrechetVideoDistance.update)
    assert "_accumulate_patches" in source
    for forbidden in ("interpolate", "self.inception("):
        assert forbidden not in source, \
            "{} means the shared path was bypassed".format(forbidden)


def test_camera_names_and_count_are_inherited(checkpoint):
    metric = NuScenes6BandFrechetVideoDistance(checkpoint)
    assert metric.CAM_NAMES == PanoPatchFrechetVideoDistance.CAM_NAMES
    assert metric.CAM_NAMES[0] == "CAM_BACK"
    assert len(metric.CAM_NAMES) == 6
    assert metric.equal_patch_width == 0, \
        "equal_patch_width is a pano-column control and must stay off here"


def test_pooled_and_per_camera_states_advance(checkpoint):
    metric = NuScenes6BandFrechetVideoDistance(checkpoint)
    metric.update(_views(2, 10, 32, 48, 0), real=True)
    metric.update(_views(2, 10, 32, 48, 1), real=False)
    # Six cameras x two clips per side.
    assert int(metric.real_features_num_samples) == 12
    assert int(metric.fake_features_num_samples) == 12
    assert torch.equal(
        metric.real_cam_num_samples, torch.full((6,), 2, dtype=torch.long))
    value = float(metric.compute())
    assert value == value and value >= 0.0
    per_camera = metric.compute_per_camera()
    assert list(per_camera) == list(metric.CAM_NAMES)


def test_camera_ordering_is_camera_major(checkpoint):
    """Swapping two cameras must change the per-camera values, not just pool.

    This is the guard for the exact off-by-one that already bit the pano
    per-camera labelling once.
    """
    reference = _views(2, 10, 32, 48, 3)
    swapped = reference.clone()
    swapped[:, :, [1, 2]] = swapped[:, :, [2, 1]]

    def per_camera_of(views):
        metric = NuScenes6BandFrechetVideoDistance(checkpoint)
        metric.update(views, real=True)
        metric.update(_views(2, 10, 32, 48, 4), real=False)
        return metric.compute_per_camera()

    first = per_camera_of(reference)
    second = per_camera_of(swapped)
    assert first["CAM_BACK_LEFT"] != second["CAM_BACK_LEFT"]
    assert first["CAM_FRONT_LEFT"] != second["CAM_FRONT_LEFT"]
    assert first["CAM_BACK"] == pytest.approx(second["CAM_BACK"])


def test_valid_mask_is_applied(checkpoint):
    views = _views(1, 10, 32, 48, 5)
    valid = torch.ones(6, 32, 48, dtype=torch.bool)
    valid[:, :8] = False

    masked = NuScenes6BandFrechetVideoDistance(checkpoint)
    masked.update(views, real=True, valid=valid)

    pre_masked = NuScenes6BandFrechetVideoDistance(checkpoint)
    manual = views.clone()
    manual[:, :, :, :, :8] = 0.0
    pre_masked.update(manual, real=True)

    assert torch.allclose(
        masked.real_features_sum, pre_masked.real_features_sum)

    off = NuScenes6BandFrechetVideoDistance(checkpoint, apply_valid_mask=False)
    off.update(views, real=True, valid=valid)
    assert not torch.allclose(
        off.real_features_sum, masked.real_features_sum)


def test_sequence_count_truncates_views_and_mask(checkpoint):
    metric = NuScenes6BandFrechetVideoDistance(checkpoint, sequence_count=10)
    views = _views(1, 16, 32, 48, 6)
    valid = torch.ones(1, 16, 6, 32, 48, dtype=torch.bool)
    metric.update(views, real=True, valid=valid)
    assert int(metric.real_features_num_samples) == 6


def test_uint8_quantization_is_opt_in(checkpoint):
    captured = {}
    metric = NuScenes6BandFrechetVideoDistance(
        checkpoint, quantize_uint8=True,
        render_height=24, render_width=32)

    def capture(views, real=True, valid=None):
        captured["views"] = views

    metric.update = capture
    metric.update_from_canvas(
        _canvas(1, 10, seed=16), _fake_batch(1, 10), start=0, real=True)
    scaled = captured["views"] * 255
    assert torch.equal(scaled, scaled.round())


def test_common_visual_dump_is_lossless_uint8(
        checkpoint, tmp_path, monkeypatch):
    monkeypatch.setenv("PANDWM_COMMON_VIS_DIR", str(tmp_path))
    metric = NuScenes6BandFrechetVideoDistance(
        checkpoint, quantize_uint8=True,
        render_height=24, render_width=32)
    metric.update = lambda *args, **kwargs: None
    batch = _fake_batch(1, 10)
    batch["anchor_timestamp_us"] = torch.tensor([123456789])
    canvas = _canvas(1, 10, seed=17)
    metric.update_from_canvas(canvas, batch, start=0, real=True)
    metric.update_from_canvas(canvas, batch, start=0, real=False)
    real = torch.load(
        tmp_path / "real" / "123456789.pt", weights_only=True)
    fake = torch.load(
        tmp_path / "fake" / "123456789.pt", weights_only=True)
    assert real.dtype == fake.dtype == torch.uint8
    assert real.shape == fake.shape == (10, 3, 24, 32)
    assert torch.equal(real, fake)


def test_optional_fid_does_not_mislabel_pooled_fvd():
    """The six-camera pool must never masquerade as CAM_FRONT FVD."""
    import inspect

    source = inspect.getsource(NuScenes6BandFrechetVideoDistance.compute)
    assert '"pooled_fvd"' in source
    assert '"front_fvd"' not in source


def test_real_vs_real_splits_are_disjoint_and_reproducible():
    first_run = list(split_indices(149, 74, 3, 20260909))
    second_run = list(split_indices(149, 74, 3, 20260909))
    for (first, second), (first_again, second_again) in zip(
            first_run, second_run):
        assert torch.equal(first, first_again)
        assert torch.equal(second, second_again)
        assert len(set(first.tolist()) & set(second.tolist())) == 0
        assert len(first) == len(second) == 74


def test_rejects_wrong_shapes(checkpoint):
    metric = NuScenes6BandFrechetVideoDistance(checkpoint)
    with pytest.raises(ValueError):
        metric.update(torch.rand(2, 10, 3, 32, 48))
    with pytest.raises(ValueError):
        metric.update(torch.rand(2, 10, 5, 3, 32, 48))
    with pytest.raises(ValueError):
        metric.update(
            torch.rand(1, 10, 6, 3, 32, 48),
            valid=torch.ones(10, 6, 32, dtype=torch.bool))


# ---------------------------------------------------------------------------
# update_from_canvas: the evaluation-pipeline entry point
# ---------------------------------------------------------------------------

def _fake_batch(batch_size, frames, seed=11):
    """A batch carrying only what the bridge metric reads."""
    import math

    import numpy as np

    from dwm_pano.datasets.pano_projection import (
        CAM_ORDER, PANO_W, VORONOI_BOUNDARIES,
    )

    boundary = [(b - PANO_W / 2.0) / PANO_W * 2.0 * math.pi
                for b in VORONOI_BOUNDARIES]
    c2w = []
    intrinsics = []
    sizes = []
    for index in range(len(CAM_ORDER)):
        start = boundary[index - 1]
        span = (boundary[index] - start) % (2.0 * math.pi)
        yaw = -(start + span / 2.0)
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
        matrix = np.eye(4)
        matrix[:3, 0] = right
        matrix[:3, 1] = np.array([0.0, 0.0, -1.0])
        matrix[:3, 2] = forward
        matrix[:3, 3] = np.array([0.0, 0.0, 1.5])
        c2w.append(matrix)
        intrinsics.append(
            np.array([[1266.4, 0.0, 800.0], [0.0, 1266.4, 491.5],
                      [0.0, 0.0, 1.0]]))
        sizes.append((1600, 900))
    rig_c2w = torch.from_numpy(np.stack(c2w)).float()
    rig_k = torch.from_numpy(np.stack(intrinsics)).float()
    rig_size = torch.tensor(sizes, dtype=torch.int64)
    return {
        "pano_camera_transforms":
            rig_c2w[None, None].expand(batch_size, frames, -1, -1, -1).clone(),
        "pano_camera_intrinsics":
            rig_k[None, None].expand(batch_size, frames, -1, -1, -1).clone(),
        "pano_camera_image_size":
            rig_size[None, None].expand(batch_size, frames, -1, -1).clone(),
    }


def _canvas(batch_size, frames, seed=12):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch_size, frames, 3, 256, 4114, generator=generator)


def test_update_from_canvas_matches_the_offline_bridge(checkpoint):
    """The inline path and the offline runner must produce identical views.

    They are the same instrument or they are not comparable, and the whole
    calibration exercise depends on the inline number sitting on the same
    scale as the offline floor.
    """
    from dwm_pano.nuscenes6_bridge import (
        canvas_source_axes, resample_pano_to_rig,
    )

    canvas = _canvas(1, 10)
    batch = _fake_batch(1, 10)
    metric = NuScenes6BandFrechetVideoDistance(
        checkpoint, render_height=45, render_width=80)
    metric.update_from_canvas(canvas, batch, start=0, real=True)

    axes = canvas_source_axes(half_size=(256, (4114 + 494) // 2))
    offline, _, _ = resample_pano_to_rig(
        canvas[0, 3], batch["pano_camera_transforms"][0, 0].numpy(),
        batch["pano_camera_intrinsics"][0, 0].numpy(),
        batch["pano_camera_image_size"][0, 0].numpy(),
        (45, 80), source_axes=axes)

    inline = metric._grid_cache
    assert len(inline) == 1
    grids, valid, _ = next(iter(inline.values()))
    from dwm_pano.nuscenes6_bridge import apply_camera_sampling_grid
    rebuilt = torch.stack([
        apply_camera_sampling_grid(canvas[0, 3:4], grids[i], valid[i])[0]
        for i in range(6)])
    assert torch.allclose(rebuilt, offline, atol=1e-6)


def test_grid_cache_holds_one_entry_per_rig(checkpoint):
    canvas = _canvas(1, 10)
    batch = _fake_batch(1, 10)
    metric = NuScenes6BandFrechetVideoDistance(
        checkpoint, render_height=45, render_width=80)
    for _ in range(3):
        metric.update_from_canvas(canvas, batch, start=0, real=True)
    assert len(metric._grid_cache) == 1, \
        "identical calibration must not rebuild the grid"

    other = _fake_batch(1, 10)
    other["pano_camera_intrinsics"] = other["pano_camera_intrinsics"] * 1.05
    metric.update_from_canvas(canvas, other, start=0, real=True)
    assert len(metric._grid_cache) == 2


def test_grid_cache_refuses_to_grow_without_bound(checkpoint):
    canvas = _canvas(1, 10)
    metric = NuScenes6BandFrechetVideoDistance(
        checkpoint, render_height=45, render_width=80,
        max_calibration_entries=2)
    for scale in (1.0, 1.05):
        batch = _fake_batch(1, 10)
        batch["pano_camera_intrinsics"] = \
            batch["pano_camera_intrinsics"] * scale
        metric.update_from_canvas(canvas, batch, start=0, real=True)
    third = _fake_batch(1, 10)
    third["pano_camera_intrinsics"] = third["pano_camera_intrinsics"] * 1.11
    with pytest.raises(RuntimeError):
        metric.update_from_canvas(canvas, third, start=0, real=True)


def test_canvas_shape_change_is_rejected(checkpoint):
    metric = NuScenes6BandFrechetVideoDistance(
        checkpoint, render_height=45, render_width=80)
    metric.update_from_canvas(_canvas(1, 10), _fake_batch(1, 10), 0, True)
    with pytest.raises(ValueError):
        metric.update_from_canvas(
            torch.rand(1, 10, 3, 128, 4114), _fake_batch(1, 10), 0, True)


def test_requires_the_real_camera_keys_not_the_virtual_ones(checkpoint):
    """The virtual pano camera keys must never be silently accepted."""
    metric = NuScenes6BandFrechetVideoDistance(
        checkpoint, render_height=45, render_width=80)
    batch = _fake_batch(1, 10)
    virtual_only = {
        "camera_transforms": batch["pano_camera_transforms"],
        "camera_intrinsics": batch["pano_camera_intrinsics"],
        "image_size": batch["pano_camera_image_size"],
    }
    with pytest.raises(KeyError):
        metric.update_from_canvas(_canvas(1, 10), virtual_only, 0, True)
