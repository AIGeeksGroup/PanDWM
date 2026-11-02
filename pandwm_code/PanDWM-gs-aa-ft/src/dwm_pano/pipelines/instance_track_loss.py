"""MV-TAP-inspired relational loss over persistent nuScenes 3D-box tracks.

The loss compares small ROI descriptors in the student's predicted clean
latent with the same relationships in the real VAE latent.  It therefore does
not demand pixel identity across viewpoints: real pose, lighting and occlusion
changes define how much variation is legitimate.  Positive relations cover
time and overlapping pano views; same-frame different-ID relations prevent a
collapsed descriptor from satisfying every positive pair.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import torch
import torch.nn.functional as F


def sample_track_descriptors(
    features,
    boxes,
    valid,
    pool_size=2,
    min_latent_size=0.75,
):
    """Bilinearly pool valid normalized ROIs from ``[B,T,V,C,H,W]``."""
    if features.ndim != 6:
        raise ValueError("features must have shape [B,T,V,C,H,W]")
    if boxes.shape != (*features.shape[:2], valid.shape[2], features.shape[2], 4):
        raise ValueError("box tensor shape does not match features/valid")
    if valid.shape != boxes.shape[:-1]:
        raise ValueError("valid tensor shape does not match boxes")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")

    boxes = boxes.to(device=features.device, dtype=torch.float32)
    valid = valid.to(device=features.device, dtype=torch.bool)
    height, width = features.shape[-2:]
    box_width = (boxes[..., 2] - boxes[..., 0]) * width
    box_height = (boxes[..., 3] - boxes[..., 1]) * height
    valid = valid & (box_width >= min_latent_size) & \
        (box_height >= min_latent_size)
    metadata = valid.nonzero(as_tuple=False)  # [K, b/t/track/view]
    if metadata.numel() == 0:
        return features.new_zeros((0, features.shape[3] * pool_size ** 2)), \
            metadata

    b, t, track, view = metadata.unbind(dim=1)
    roi = boxes[b, t, track, view].clamp(0.0, 1.0)
    image_index = (b * features.shape[1] + t) * features.shape[2] + view
    scale = torch.tensor(
        (width, height, width, height),
        device=features.device, dtype=torch.float32)
    rois = torch.cat((image_index[:, None].float(), roi * scale), dim=1)
    # roi_align indexes one shared B*T*V feature tensor.  Repeating a complete
    # C*H*W feature map once per ROI would consume ~1 GB at the production
    # track count, which is unsafe beside a 3.35B model on an 80 GB H100.
    from torchvision.ops import roi_align
    pooled = roi_align(
        features.reshape(-1, features.shape[3], height, width).float(),
        rois,
        output_size=(pool_size, pool_size),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )
    descriptors = F.normalize(pooled.flatten(1), dim=-1, eps=1e-6)
    return descriptors, metadata


def build_track_pairs(
    metadata,
    future_flags,
    max_temporal_gap=2,
    max_negative_pairs=2048,
):
    """Build deterministic temporal, cross-view and negative pair indices."""
    rows = metadata.detach().cpu().tolist()
    is_future = future_flags.detach().cpu().tolist()
    temporal_groups = defaultdict(list)
    view_groups = defaultdict(list)
    negative_groups = defaultdict(list)
    for index, (batch, time, track, view) in enumerate(rows):
        temporal_groups[(batch, track, view)].append((time, index))
        view_groups[(batch, time, track)].append((view, index))
        negative_groups[(batch, time, view)].append((track, index))

    temporal = []
    for observations in temporal_groups.values():
        observations.sort()
        for (time_a, index_a), (time_b, index_b) in zip(
            observations, observations[1:]
        ):
            if time_b - time_a <= max_temporal_gap and \
                    (is_future[index_a] or is_future[index_b]):
                temporal.append((index_a, index_b))

    cross_view = []
    for observations in view_groups.values():
        observations.sort()
        for (_, index_a), (_, index_b) in combinations(observations, 2):
            if is_future[index_a] or is_future[index_b]:
                cross_view.append((index_a, index_b))

    negatives = []
    for observations in negative_groups.values():
        observations.sort()
        for (track_a, index_a), (track_b, index_b) in combinations(
            observations, 2
        ):
            if track_a != track_b and \
                    (is_future[index_a] or is_future[index_b]):
                negatives.append((index_a, index_b))
    if max_negative_pairs >= 0 and len(negatives) > max_negative_pairs:
        # Evenly subsample rather than take the first frames/objects only.
        keep = torch.linspace(
            0, len(negatives) - 1, max_negative_pairs,
            dtype=torch.long,
        ).tolist()
        negatives = [negatives[index] for index in keep]
    return temporal, cross_view, negatives


def _relational_error(predicted, target, pairs):
    if not pairs:
        return predicted.sum() * 0.0
    pair_tensor = torch.tensor(
        pairs, device=predicted.device, dtype=torch.long)
    pred_similarity = (
        predicted[pair_tensor[:, 0]] * predicted[pair_tensor[:, 1]]
    ).sum(dim=-1)
    with torch.no_grad():
        target_similarity = (
            target[pair_tensor[:, 0]] * target[pair_tensor[:, 1]]
        ).sum(dim=-1)
    return F.smooth_l1_loss(
        pred_similarity, target_similarity, beta=0.1)


def box_track_relational_loss(
    predicted,
    target,
    boxes,
    valid,
    reference_mask,
    pool_size=2,
    min_latent_size=0.75,
    max_temporal_gap=2,
    temporal_weight=1.0,
    cross_view_weight=1.0,
    negative_weight=0.25,
    max_negative_pairs=2048,
):
    """Return a real-change-calibrated track loss and scalar diagnostics."""
    pred_desc, metadata = sample_track_descriptors(
        predicted, boxes, valid, pool_size=pool_size,
        min_latent_size=min_latent_size)
    target_desc, target_metadata = sample_track_descriptors(
        target.detach(), boxes, valid, pool_size=pool_size,
        min_latent_size=min_latent_size)
    if not torch.equal(metadata, target_metadata):
        raise RuntimeError("predicted and target ROI metadata diverged")
    if metadata.numel() == 0:
        zero = predicted.sum() * 0.0
        return zero, {
            "track_roi_count": 0.0,
            "track_temporal_pairs": 0.0,
            "track_cross_view_pairs": 0.0,
            "track_negative_pairs": 0.0,
            "track_temporal_loss": 0.0,
            "track_cross_view_loss": 0.0,
            "track_negative_loss": 0.0,
        }

    reference_mask = reference_mask.to(
        device=metadata.device, dtype=torch.bool)
    b, t, _, view = metadata.unbind(dim=1)
    future_flags = torch.logical_not(reference_mask[b, t, view])
    temporal, cross_view, negatives = build_track_pairs(
        metadata, future_flags,
        max_temporal_gap=max_temporal_gap,
        max_negative_pairs=max_negative_pairs)
    temporal_loss = _relational_error(pred_desc, target_desc, temporal)
    cross_view_loss = _relational_error(pred_desc, target_desc, cross_view)
    negative_loss = _relational_error(pred_desc, target_desc, negatives)
    total = temporal_weight * temporal_loss + \
        cross_view_weight * cross_view_loss + \
        negative_weight * negative_loss
    report = {
        "track_roi_count": float(metadata.shape[0]),
        "track_temporal_pairs": float(len(temporal)),
        "track_cross_view_pairs": float(len(cross_view)),
        "track_negative_pairs": float(len(negatives)),
        "track_temporal_loss": float(temporal_loss.detach()),
        "track_cross_view_loss": float(cross_view_loss.detach()),
        "track_negative_loss": float(negative_loss.detach()),
    }
    return total, report
