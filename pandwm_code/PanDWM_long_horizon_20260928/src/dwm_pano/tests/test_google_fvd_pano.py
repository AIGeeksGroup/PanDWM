import numpy as np

from dwm_pano.metrics.google_fvd_pano import (
    CAMERA_NAMES,
    frechet_distance,
    split_camera_regions,
)


def test_panorama_owner_regions_cover_each_pixel_once():
    frame = np.zeros((8, 4114, 3), dtype=np.uint8)
    regions = split_camera_regions(frame)
    assert len(regions) == len(CAMERA_NAMES) == 6
    assert [region.shape[1] for region in regions] == [
        1032, 598, 619, 637, 635, 593,
    ]
    assert sum(region.shape[1] for region in regions) == 4114


def test_frechet_distance_is_zero_for_identical_features():
    features = np.random.default_rng(0).normal(size=(20, 4))
    assert frechet_distance(features, features) == 0.0
