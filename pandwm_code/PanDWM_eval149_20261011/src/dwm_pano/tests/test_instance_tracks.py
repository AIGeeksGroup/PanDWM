"""Tests for the fixed-size 3D-box track supervision path."""
import numpy as np
import pytest

import dwm_pano.datasets.instance_tracks as tracks
from dwm_pano.datasets.instance_tracks import (
    _largest_intersection,
    circular_extent_segments,
)


def test_circular_extent_uses_short_non_wrapped_arc():
    segments = circular_extent_segments(np.array([100.0, 130.0, 180.0]))
    assert segments == pytest.approx([(100.0, 180.0)])


def test_circular_extent_splits_a_wrap_box():
    segments = circular_extent_segments(
        np.array([9580.0, 9590.0, 5.0, 20.0]))
    assert segments == pytest.approx([(9580.0, 9600.0), (0.0, 20.0)])
    assert _largest_intersection(segments, (0, 5376)) == \
        pytest.approx((0.0, 20.0))
    assert _largest_intersection(segments, (4224, 9600)) == \
        pytest.approx((9580.0, 9600.0))


def test_circular_extent_chooses_largest_overlap():
    assert _largest_intersection(
        [(10.0, 30.0), (80.0, 95.0)], (0.0, 100.0)
    ) == pytest.approx((10.0, 30.0))


def test_frame_projection_filters_static_classes_and_low_visibility(
    monkeypatch,
):
    monkeypatch.setattr(
        tracks, "_ego_corners_from_boxes",
        lambda boxes, rotation, translation:
            np.zeros((len(boxes), 8, 3), dtype=np.float32))
    monkeypatch.setattr(
        tracks, "project_ego_points_to_pano",
        lambda corners: (
            np.linspace(1000.0, 1200.0, 8),
            np.linspace(200.0, 300.0, 8),
            np.ones(8, dtype=np.bool_),
        ))
    info = {
        "gt_boxes": np.zeros((3, 7), dtype=np.float32),
        "gt_box_ids": ["keep", "static", "occluded"],
        "gt_names": ["car", "barrier", "truck"],
        "visibility": [2, 4, 1],
        "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
        "lidar2ego_translation": [0.0, 0.0, 0.0],
    }
    ids, boxes, valid = tracks.frame_instance_half_boxes(
        info, half_size=(2304, 256))
    assert ids == ["keep"]
    assert boxes.shape == (1, 2, 4)
    assert valid[0].tolist() == [True, False]
