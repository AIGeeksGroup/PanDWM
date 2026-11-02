"""Tests for the pano -> real-6-camera protocol bridge.

The point of these is that the bridge inverts the *authoritative* projection
(`project_ego_points_to_pano`) rather than a re-derived copy of it, and that
the band limitation is a measured number instead of an assumption.

Real nuScenes calibration is not in this workspace, so the rig used here is
the published CAM_FRONT intrinsic plus the real sensor2ego translations, with
the six cameras' yaws taken from the pano's own Voronoi cell centres.  The
coverage-fraction test instead pins the numbers measured from the actual
val pkl on the cluster, so a drift in PANO_BAND_HALF_RAD fails loudly.
"""
import math

import numpy as np
import pytest
import torch

from dwm_pano.datasets.pano_projection import (
    CAM_ORDER, PANO_H, PANO_W, VORONOI_BOUNDARIES,
    project_ego_points_to_pano,
)
from dwm_pano.nuscenes6_bridge import (
    OWNER_PROBE_RADIUS_M, PANO_BAND_HALF_RAD, ego_directions_to_pano_pixels,
    real_camera_band_table, real_camera_pixel_directions_ego,
    real_camera_valid_row_band, resample_pano_to_real_camera,
    resample_pano_to_rig, voronoi_owner_indices,
)

# nuScenes n015 CAM_FRONT, straight from the val pkl.
NUSC_K_FRONT = np.array([
    [1266.4172, 0.0, 816.267],
    [0.0, 1266.4172, 491.5071],
    [0.0, 0.0, 1.0],
])
# fy / cy of the six cameras as measured from the cluster val pkl; used to
# reproduce the coverage table without shipping the pkl.
MEASURED_RIG = {
    #                 fy          cy      height  expected covered fraction
    "CAM_BACK":      (809.2209,  479.75,   900,   0.538),
    "CAM_BACK_LEFT": (1256.7414, 492.17,   900,   0.836),
    "CAM_FRONT_LEFT": (1272.5979, 484.79,  900,   0.847),
    "CAM_FRONT":     (1266.4172, 491.5071, 900,   0.843),
    "CAM_FRONT_RIGHT": (1260.8474, 495.33, 900,   0.839),
    "CAM_BACK_RIGHT": (1259.5122, 496.28,  900,   0.838),
}


def _yaw_c2w(yaw_rad, translation):
    """OpenCV-convention camera-to-ego for a camera looking along `yaw_rad`."""
    forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
    right = np.array([math.sin(yaw_rad), -math.cos(yaw_rad), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = translation
    return c2w


def _voronoi_cell_center_yaws():
    """Ego yaw of each Voronoi cell centre, in CAM_ORDER.

    pano_projection's own rule is that camera ``i`` owns
    ``[boundary[i - 1], boundary[i])`` with CAM_BACK wrapping, so cell ``i``
    starts at the *previous* boundary.  Taking ``boundary[i]`` as the start
    instead points every camera one wedge clockwise, which shows up as a
    ~90% foreign-region fraction -- the same off-by-one class as the
    per-camera label slip recorded in the change log.
    """
    boundary = [(b - PANO_W / 2.0) / PANO_W * 2.0 * math.pi
                for b in VORONOI_BOUNDARIES]
    yaws = []
    for index in range(len(boundary)):
        start = boundary[index - 1]
        stop = boundary[index]
        span = (stop - start) % (2.0 * math.pi)
        azimuth = start + span / 2.0
        # pano azimuth is atan2(-y_ego, x_ego), so ego yaw = -azimuth
        yaws.append(-azimuth)
    return yaws


def _synthetic_rig(output_translation=True):
    yaws = _voronoi_cell_center_yaws()
    c2w = []
    intrinsics = []
    image_size = []
    for index, name in enumerate(CAM_ORDER):
        fy, cy, height, _ = MEASURED_RIG[name]
        k = np.array([[fy, 0.0, 800.0], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        translation = np.array([0.0, 0.0, 1.5]) if output_translation \
            else np.zeros(3)
        c2w.append(_yaw_c2w(yaws[index], translation))
        intrinsics.append(k)
        image_size.append((1600, height))
    return np.stack(c2w), np.stack(intrinsics), np.asarray(image_size)


def test_band_half_extent_is_the_cropped_pano_not_the_pre_crop_fov():
    """16.67 deg, not the 25 deg pre-crop half-FOV."""
    assert math.degrees(PANO_BAND_HALF_RAD) == pytest.approx(16.6667, abs=1e-3)


def test_measured_coverage_fractions_are_reproduced():
    """Pins the numbers measured from the cluster val pkl.

    CAM_BACK is the one that matters: it is a wide-FOV camera, so almost half
    its frame has no pano content and a full-frame 6-camera protocol is
    impossible, not merely inconvenient.
    """
    for name, (fy, cy, height, expected) in MEASURED_RIG.items():
        k = np.array([[fy, 0.0, 800.0], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        low, high = real_camera_valid_row_band(k, height)
        assert (high - low) / height == pytest.approx(expected, abs=2e-3), \
            "{} coverage drifted".format(name)
    back = MEASURED_RIG["CAM_BACK"]
    low, high = real_camera_valid_row_band(
        np.array([[back[0], 0.0, 800.0], [0.0, back[0], back[1]],
                  [0.0, 0.0, 1.0]]), back[2])
    assert (high - low) / back[2] < 0.6, \
        "CAM_BACK must stay the binding camera"


def test_band_table_shape_and_ordering():
    c2w, intrinsics, image_size = _synthetic_rig()
    table = real_camera_band_table(c2w, intrinsics, image_size)
    assert [row["camera"] for row in table] == list(CAM_ORDER)
    assert all(0.0 < row["covered_fraction"] <= 1.0 for row in table)
    with pytest.raises(ValueError):
        real_camera_band_table(c2w[:5], intrinsics, image_size)


def test_directions_are_unit_and_center_pixel_hits_the_optical_axis():
    c2w = _yaw_c2w(0.0, np.array([1.7, 0.0, 1.5]))
    directions = real_camera_pixel_directions_ego(
        c2w, NUSC_K_FRONT, 64, 64,
        row_offset=0.0, row_stride=900.0 / 64,
        column_offset=0.0, column_stride=1600.0 / 64)
    norms = np.linalg.norm(directions, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-12)
    # The pixel nearest the principal point must look nearly straight ahead.
    u = (np.arange(64) + 0.5) * (1600.0 / 64)
    v = (np.arange(64) + 0.5) * (900.0 / 64)
    j = int(np.argmin(np.abs(u - NUSC_K_FRONT[0, 2])))
    i = int(np.argmin(np.abs(v - NUSC_K_FRONT[1, 2])))
    assert directions[i, j] @ np.array([1.0, 0.0, 0.0]) > 0.9995


def test_direction_projection_matches_the_authoritative_point_projection():
    """The bridge must invert project_ego_points_to_pano, not a copy of it."""
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(500, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    tx, ty, in_band = ego_directions_to_pano_pixels(directions)
    for radius in (1.0, 7.5, 250.0):
        tx_point, ty_point, valid_point = project_ego_points_to_pano(
            directions * radius)
        both = in_band & valid_point
        assert both.sum() > 100
        assert np.allclose(tx[both], tx_point[both], atol=1e-9)
        assert np.allclose(ty[both], ty_point[both], atol=1e-9)
        # And the in-band flag must agree with the authority's own row test.
        assert np.array_equal(in_band, valid_point) or radius < 1.0


def test_pano_pixel_roundtrip_is_exact_for_a_column_ramp():
    """Resampling a pano whose value encodes azimuth recovers that azimuth."""
    column = np.arange(PANO_W, dtype=np.float64)
    pano = torch.from_numpy(
        np.broadcast_to(column[None, None], (1, PANO_H, PANO_W)).copy())
    c2w = _yaw_c2w(0.0, np.zeros(3))
    image, valid, _ = resample_pano_to_real_camera(
        pano, c2w, NUSC_K_FRONT, (1600, 900), (90, 160), band_limit=True)
    directions = real_camera_pixel_directions_ego(
        c2w, NUSC_K_FRONT, 90, 160,
        row_offset=real_camera_valid_row_band(NUSC_K_FRONT, 900)[0],
        row_stride=(real_camera_valid_row_band(NUSC_K_FRONT, 900)[1]
                    - real_camera_valid_row_band(NUSC_K_FRONT, 900)[0]) / 90,
        column_offset=0.0, column_stride=1600.0 / 160)
    tx, _, in_band = ego_directions_to_pano_pixels(directions)
    expected = tx - 0.5  # align_corners=False pixel centre of `tx`
    got = image[0].numpy()
    inner = in_band.copy()
    # Bilinear interpolation of a ramp is exact away from the wrap seam only;
    # exclude the two columns that straddle it.
    inner &= (tx > 1.5) & (tx < PANO_W - 1.5)
    assert inner.sum() > 0.8 * in_band.sum()
    assert np.abs(got[inner] - expected[inner]).max() < 1e-6


def test_band_limit_removes_dead_rows_and_full_frame_keeps_them():
    pano = torch.ones(3, PANO_H, PANO_W)
    c2w = _yaw_c2w(0.0, np.zeros(3))
    _, valid_band, report_band = resample_pano_to_real_camera(
        pano, c2w, NUSC_K_FRONT, (1600, 900), (90, 160), band_limit=True)
    _, valid_full, report_full = resample_pano_to_real_camera(
        pano, c2w, NUSC_K_FRONT, (1600, 900), (90, 160), band_limit=False)
    assert report_band["in_band_pixel_fraction"] > 0.99
    assert report_full["in_band_pixel_fraction"] < 0.9
    assert report_band["native_row_fraction"] == pytest.approx(0.843, abs=2e-3)
    assert report_full["native_row_fraction"] == pytest.approx(1.0)
    assert bool(valid_band.all())
    assert not bool(valid_full.all())


def test_foreign_region_fraction_matches_the_fov_minus_wedge_geometry():
    """Parallax risk is a measured number, not an assumption.

    Each camera's real horizontal FOV is wider than the Voronoi wedge it owns
    in the pano, so the overhanging pixels land in a neighbour's region -- and
    that region's content was rendered from a *different* optical centre, so
    the depth-free resample carries a residual parallax error exactly there.
    The five narrow cameras overhang by 12-24%; CAM_BACK is the wide-FOV one
    whose wedge (90.3 deg) is slightly wider than its FOV (89.3 deg), so it is
    the single camera that is horizontally exact.  Measured with the rig built
    from the cluster val pkl's fy/cy.

    Note the real CAM_FRONT translation *lowers* its foreign fraction
    (0.163 -> 0.125): moving forward pulls the far probe points' ego-origin
    azimuth toward the optical axis.  The effect is not monotone in
    translation, so this test pins values instead of a direction.
    """
    expected = {
        "CAM_BACK": 0.000,
        "CAM_BACK_LEFT": 0.225,
        "CAM_FRONT_LEFT": 0.188,
        "CAM_FRONT": 0.163,
        "CAM_FRONT_RIGHT": 0.175,
        "CAM_BACK_RIGHT": 0.237,
    }
    yaws = _voronoi_cell_center_yaws()
    pano = torch.zeros(1, PANO_H, PANO_W)
    for index, name in enumerate(CAM_ORDER):
        fy, cy, height, _ = MEASURED_RIG[name]
        k = np.array([[fy, 0.0, 800.0], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        _, _, report = resample_pano_to_real_camera(
            pano, _yaw_c2w(yaws[index], np.zeros(3)), k, (1600, height),
            (90, 160), band_limit=True, camera_index=index)
        assert report["owner_probe_radius_m"] == OWNER_PROBE_RADIUS_M
        assert report["foreign_region_fraction"] == pytest.approx(
            expected[name], abs=0.02), name
    # CAM_BACK is the only horizontally exact camera, and it is also the one
    # that loses the most rows vertically.  Both halves of that trade must
    # hold, or the bridge's instrument description is wrong.
    back = CAM_ORDER.index("CAM_BACK")
    fy, cy, height, _ = MEASURED_RIG["CAM_BACK"]
    k = np.array([[fy, 0.0, 800.0], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    _, _, report = resample_pano_to_real_camera(
        pano, _yaw_c2w(yaws[back], np.zeros(3)), k, (1600, height),
        (90, 160), band_limit=True, camera_index=back)
    assert report["foreign_region_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert report["native_row_fraction"] < 0.6


def test_real_front_translation_is_reported_separately():
    """The real 1.7 m forward offset changes the number, so it is not ignored."""
    yaws = _voronoi_cell_center_yaws()
    front = CAM_ORDER.index("CAM_FRONT")
    pano = torch.zeros(1, PANO_H, PANO_W)
    _, _, report = resample_pano_to_real_camera(
        pano, _yaw_c2w(yaws[front], np.array([1.7008, 0.0159, 1.511])),
        NUSC_K_FRONT, (1600, 900), (90, 160), band_limit=True,
        camera_index=front)
    assert report["foreign_region_fraction"] == pytest.approx(0.125, abs=0.02)


def test_owner_indices_cover_all_six_regions_over_a_full_circle():
    azimuths = np.linspace(-math.pi, math.pi, 2000, endpoint=False)
    directions = np.stack(
        [np.cos(-azimuths), np.sin(-azimuths), np.zeros_like(azimuths)],
        axis=-1)
    owners = voronoi_owner_indices(directions, np.zeros(3), 100.0)
    assert set(np.unique(owners).tolist()) == set(range(6))


def test_rig_resample_shapes_and_common_output_size():
    c2w, intrinsics, image_size = _synthetic_rig()
    pano = torch.rand(3, PANO_H, PANO_W)
    images, valid, reports = resample_pano_to_rig(
        pano, c2w, intrinsics, image_size, (90, 160))
    assert images.shape == (6, 3, 90, 160)
    assert valid.shape == (6, 90, 160)
    assert [r["camera"] for r in reports] == list(CAM_ORDER)
    # The whole reason for a common output size: per-camera sizes would
    # reintroduce the per-camera squeeze documented for PanoPatch-FVD.
    assert len({tuple(image.shape) for image in images}) == 1


def test_resample_is_deterministic():
    c2w, intrinsics, image_size = _synthetic_rig()
    pano = torch.rand(3, PANO_H, PANO_W, generator=torch.Generator()
                      .manual_seed(7))
    first, _, _ = resample_pano_to_rig(
        pano, c2w, intrinsics, image_size, (64, 96))
    second, _, _ = resample_pano_to_rig(
        pano, c2w, intrinsics, image_size, (64, 96))
    assert torch.equal(first, second)


def test_rejects_wrong_pano_shape():
    c2w = _yaw_c2w(0.0, np.zeros(3))
    with pytest.raises(ValueError):
        resample_pano_to_real_camera(
            torch.zeros(3, 256, 2304), c2w, NUSC_K_FRONT, (1600, 900),
            (90, 160))
    with pytest.raises(ValueError):
        resample_pano_to_real_camera(
            torch.zeros(PANO_H, PANO_W), c2w, NUSC_K_FRONT, (1600, 900),
            (90, 160))


# ---------------------------------------------------------------------------
# V=1 canvas source: the evaluation pipeline never materialises a 600x9600
# pano, so the bridge has to sample the 4114-wide canvas it actually produces.
# ---------------------------------------------------------------------------

def test_canvas_overlap_constant_matches_the_pipeline():
    """The mirrored constant must not drift from the pipeline's own value.

    Read as source text rather than imported: the pipeline module pulls in
    diffusers, which is not installed in this workspace.
    """
    import pathlib
    import re

    from dwm_pano.nuscenes6_bridge import PANO_CANVAS_OVERLAP_COLUMNS

    pipeline = pathlib.Path(__file__).resolve().parents[1] \
        / "pipelines" / "ctsd_pano_gs_decoder.py"
    match = re.search(
        r"^PANO_CANVAS_OVERLAP_COLUMNS\s*=\s*(\d+)\s*$",
        pipeline.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None, "the pipeline no longer declares the constant"
    assert int(match.group(1)) == PANO_CANVAS_OVERLAP_COLUMNS


def test_canvas_axes_span_the_circle_and_are_monotone():
    from dwm_pano.nuscenes6_bridge import canvas_source_axes

    rows, columns = canvas_source_axes(half_size=(256, 2304))
    assert columns.shape[0] == 2 * 2304 - 494 == 4114
    assert rows.shape[0] == 256
    assert np.all(np.diff(columns) > 0)
    assert np.all(np.diff(rows) > 0)
    # Post-crop rows, so the 900-row pre-crop offset must already be removed.
    assert -1.0 < rows[0] < 2.0
    assert PANO_H - 3.0 < rows[-1] < PANO_H
    assert 0.0 <= columns[0] < 3.0
    assert PANO_W - 3.0 < columns[-1] < PANO_W


def test_canvas_resample_agrees_with_full_pano_resample():
    """Sampling the canvas must land on the same content as the full pano.

    Built as a column ramp so the comparison is analytic: a canvas made by the
    pipeline's own two-half resize of that ramp, sampled through
    `source_axes`, has to reproduce the azimuth the full-pano path gives, to
    within the canvas's own 2.33x horizontal downscale.
    """
    from dwm_pano.nuscenes6_bridge import canvas_source_axes

    rows, columns = canvas_source_axes(half_size=(256, 2304))
    ramp_full = np.broadcast_to(
        np.arange(PANO_W, dtype=np.float64)[None, None],
        (1, PANO_H, PANO_W)).copy()
    # The canvas holds the ramp evaluated at its own column coordinates, which
    # is exactly what an ideal resize of the ramp produces.
    canvas = np.broadcast_to(
        columns[None, None], (1, rows.shape[0], columns.shape[0])).copy()

    c2w = _yaw_c2w(0.0, np.zeros(3))
    from_full, valid_full, _ = resample_pano_to_real_camera(
        torch.from_numpy(ramp_full), c2w, NUSC_K_FRONT, (1600, 900),
        (90, 160), band_limit=True)
    from_canvas, valid_canvas, _ = resample_pano_to_real_camera(
        torch.from_numpy(canvas), c2w, NUSC_K_FRONT, (1600, 900),
        (90, 160), band_limit=True,
        source_axes=(rows, columns))

    assert torch.equal(valid_full, valid_canvas)
    mask = valid_full.numpy()
    difference = np.abs(
        from_full[0].numpy()[mask] - from_canvas[0].numpy()[mask])
    # One canvas column spans 9600/4114 = 2.33 pano columns; allow a column.
    assert difference.max() < 2.5


def test_canvas_source_axes_shape_is_validated():
    from dwm_pano.nuscenes6_bridge import canvas_source_axes

    rows, columns = canvas_source_axes(half_size=(256, 2304))
    c2w = _yaw_c2w(0.0, np.zeros(3))
    with pytest.raises(ValueError):
        resample_pano_to_real_camera(
            torch.zeros(1, 128, 4114), c2w, NUSC_K_FRONT, (1600, 900),
            (90, 160), source_axes=(rows, columns))


def test_rig_resample_accepts_a_canvas():
    from dwm_pano.nuscenes6_bridge import canvas_source_axes

    rows, columns = canvas_source_axes(half_size=(256, 2304))
    c2w, intrinsics, image_size = _synthetic_rig()
    canvas = torch.rand(3, rows.shape[0], columns.shape[0])
    images, valid, reports = resample_pano_to_rig(
        canvas, c2w, intrinsics, image_size, (90, 160),
        source_axes=(rows, columns))
    assert images.shape == (6, 3, 90, 160)
    assert valid.shape == (6, 90, 160)
    assert reports[0]["camera"] == "CAM_BACK"
