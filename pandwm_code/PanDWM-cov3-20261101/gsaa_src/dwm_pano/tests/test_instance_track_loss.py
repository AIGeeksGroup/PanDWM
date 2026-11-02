"""Unit tests for the MV-TAP-inspired box-track relational loss."""
import torch

from dwm_pano.pipelines.instance_track_loss import (
    box_track_relational_loss,
    build_track_pairs,
    sample_track_descriptors,
)


def _metadata_fixture():
    # b, time, track, view
    return torch.tensor([
        [0, 0, 0, 0], [0, 1, 0, 0],
        [0, 1, 0, 1], [0, 1, 1, 0],
    ])


def test_pair_builder_separates_time_view_and_different_id():
    temporal, cross_view, negatives = build_track_pairs(
        _metadata_fixture(),
        torch.tensor([False, True, True, True]),
        max_temporal_gap=2,
    )
    assert temporal == [(0, 1)]
    assert cross_view == [(1, 2)]
    assert negatives == [(1, 3)]


def test_descriptor_pool_is_differentiable_and_uses_valid_rois():
    features = torch.arange(2 * 2 * 3 * 4 * 4, dtype=torch.float32) \
        .reshape(1, 2, 2, 3, 4, 4).requires_grad_(True)
    boxes = torch.zeros(1, 2, 2, 2, 4)
    boxes[..., 2:] = 1.0
    valid = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    valid[:, :, 0, :] = True
    descriptors, metadata = sample_track_descriptors(
        features, boxes, valid, pool_size=2, min_latent_size=1.0)
    assert descriptors.shape == (4, 12)
    assert metadata.shape == (4, 4)
    descriptors.sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_relational_loss_is_zero_when_prediction_matches_real_change():
    torch.manual_seed(3)
    target = torch.randn(1, 3, 2, 4, 6, 8)
    predicted = target.clone().requires_grad_(True)
    boxes = torch.zeros(1, 3, 2, 2, 4)
    boxes[..., 0] = 0.1
    boxes[..., 1] = 0.1
    boxes[..., 2] = 0.9
    boxes[..., 3] = 0.9
    valid = torch.ones(1, 3, 2, 2, dtype=torch.bool)
    reference = torch.zeros(1, 3, 2, dtype=torch.bool)
    reference[:, 0] = True
    loss, report = box_track_relational_loss(
        predicted, target, boxes, valid, reference,
        max_negative_pairs=100)
    assert float(loss.detach()) < 1e-10
    assert report["track_temporal_pairs"] > 0
    assert report["track_cross_view_pairs"] > 0
    assert report["track_negative_pairs"] > 0


def test_relational_loss_penalizes_identity_collapse_and_backpropagates():
    torch.manual_seed(5)
    target = torch.randn(1, 3, 2, 4, 6, 8)
    predicted = torch.ones_like(target).requires_grad_(True)
    boxes = torch.zeros(1, 3, 2, 2, 4)
    boxes[..., 2:] = 1.0
    valid = torch.ones(1, 3, 2, 2, dtype=torch.bool)
    reference = torch.zeros(1, 3, 2, dtype=torch.bool)
    reference[:, 0] = True
    loss, _ = box_track_relational_loss(
        predicted, target, boxes, valid, reference,
        max_negative_pairs=100)
    assert float(loss.detach()) > 0.01
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert float(predicted.grad.abs().sum()) > 0.0
