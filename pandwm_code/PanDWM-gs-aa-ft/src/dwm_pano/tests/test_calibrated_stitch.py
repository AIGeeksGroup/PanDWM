"""CPU regression tests for calibrated/legacy differentiable pano stitching."""
import math

import numpy as np
import pytest
import torch

from dwm_pano.calibrated_stitch import (
    build_owned_pano_sampling_grid,
    camera_records_to_arrays,
    differentiable_hard_seam_stitch,
    half_canvas_pano_axes,
    hard_seam_owner_map,
    hard_seam_owners_from_columns,
    opencv_intrinsics_to_gsplat,
    prepare_gsplat_intrinsics,
    resize_intrinsics,
    sampling_grid_to_ego_rays,
    scale_camera_intrinsics,
    stitch_camera_views_from_owned_grid,
)
from dwm_pano.datasets.pano_projection import (
    CAM_ORDER,
    PANO_H_PRE_CROP,
    PANO_W,
    VORONOI_BOUNDARIES,
)


def _cell_center_rig(*, source_size=(16, 32)):
    """OpenCV c2w cameras looking at the six fixed owner-cell centers."""
    transforms = torch.eye(4, dtype=torch.float32).repeat(6, 1, 1)
    cell_centers = []
    for camera_index in range(6):
        if camera_index == 0:
            start = VORONOI_BOUNDARIES[-1]
            stop = VORONOI_BOUNDARIES[0] + PANO_W
        else:
            start = VORONOI_BOUNDARIES[camera_index - 1]
            stop = VORONOI_BOUNDARIES[camera_index]
        column = ((start + stop) / 2.0) % PANO_W
        yaw = (
            (column - PANO_W / 2.0 - 0.5) / PANO_W
        ) * (2.0 * math.pi)
        cell_centers.append(column)
        # OpenCV columns are right, down, forward in ego coordinates.
        transforms[camera_index, :3, 0] = torch.tensor([
            -math.sin(yaw), -math.cos(yaw), 0.0])
        transforms[camera_index, :3, 1] = torch.tensor([0.0, 0.0, -1.0])
        transforms[camera_index, :3, 2] = torch.tensor([
            math.cos(yaw), -math.sin(yaw), 0.0])
        transforms[camera_index, :3, 3] = torch.tensor([
            camera_index / 10.0, -camera_index / 20.0, 1.5])

    height, width = source_size
    # A generous FOV keeps every fixed owner cell in bounds in the small test.
    intrinsics = torch.tensor([
        [12.0, 0.0, width / 2.0],
        [0.0, 12.0, height / 2.0],
        [0.0, 0.0, 1.0],
    ]).repeat(6, 1, 1)
    return transforms, intrinsics, cell_centers


def test_camera_records_are_packed_in_hard_seam_order():
    cameras = {}
    for index, name in enumerate(reversed(CAM_ORDER)):
        source_index = CAM_ORDER.index(name)
        cameras[name] = {
            "sensor2ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "sensor2ego_translation": [source_index, 2.0, 3.0],
            "camera_intrinsics": [
                [100.0 + source_index, 0.0, 20.0],
                [0.0, 101.0 + source_index, 10.0],
                [0.0, 0.0, 1.0],
            ],
        }
    transforms, intrinsics, image_sizes = camera_records_to_arrays(cameras)
    assert transforms.shape == (6, 4, 4)
    assert intrinsics.shape == (6, 3, 3)
    assert image_sizes.shape == (6, 2)
    assert transforms[:, 0, 3].tolist() == list(range(6))
    assert intrinsics[:, 0, 0].tolist() == list(np.arange(100.0, 106.0))
    assert np.all(image_sizes == np.asarray([1600, 900]))


def test_intrinsics_scaling_preserves_homogeneous_row_and_skew():
    intrinsics = torch.tensor([
        [100.0, 4.0, 80.0],
        [0.0, 120.0, 45.0],
        [0.0, 0.0, 1.0],
    ])
    scaled = scale_camera_intrinsics(
        intrinsics, source_size=(900, 1600), target_size=(360, 640))
    assert torch.allclose(
        scaled[0], torch.tensor([40.0, 1.6, 32.2]))
    assert torch.allclose(
        scaled[1], torch.tensor([0.0, 48.0, 18.2]))
    assert torch.equal(scaled[2], intrinsics[2])


def test_half_canvas_axes_preserve_two_independent_resizes():
    rows, columns = half_canvas_pano_axes(
        half_size=(256, 2304), right_trim_px=494, dtype=torch.float64)
    assert rows.shape == (256,)
    assert columns.shape == (4114,)
    assert rows[0].item() == pytest.approx(150.671875)
    assert rows[-1].item() == pytest.approx(748.328125)
    assert columns[2303].item() == pytest.approx(5374.333333333333)
    assert columns[2304].item() == pytest.approx(5377.333333333333)
    # A global 9600->4114 resize would have no 3-canonical-pixel jump here.
    assert (columns[2304] - columns[2303]).item() == pytest.approx(3.0)


def test_hard_seam_erosion_removes_exact_output_pixels_per_side():
    columns = torch.tensor([1226.0, 1227.0, 1228.0, 1229.0,
                            1230.0, 1231.0, 1232.0])
    owners, valid = hard_seam_owners_from_columns(
        columns, seam_erosion_px=2)
    assert owners.tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert valid.tolist() == [True, False, False, False, False, True, True]


def test_calibrated_grid_uses_align_corners_false_pixel_centers():
    transforms, intrinsics, centers = _cell_center_rig()
    # Exact horizontal center of CAM_FRONT's owner cell and exact zero-elevation
    # center in the pre-crop 900-row panorama.
    rows = torch.tensor([PANO_H_PRE_CROP / 2.0 + 0.5])
    columns = torch.tensor([centers[3]])
    grid, owner, valid = build_owned_pano_sampling_grid(
        transforms, intrinsics, source_size=(16, 32),
        pano_rows=rows, pano_columns=columns,
        projection_model="calibrated")
    assert owner.item() == 3
    assert valid.item()
    # gsplat/K use image-edge coordinates and pixel centers x+0.5/y+0.5;
    # K principal point (W/2,H/2) is normalized zero for align_corners=False.
    assert torch.allclose(grid[0, 0], torch.zeros(2), atol=2e-5)

    origins, directions, z_cosine = sampling_grid_to_ego_rays(
        grid, owner, transforms, intrinsics, source_size=(16, 32))
    assert torch.allclose(origins[0, 0], transforms[3, :3, 3])
    assert torch.allclose(
        directions[0, 0], transforms[3, :3, 2], atol=2e-5)
    assert z_cosine.item() == pytest.approx(1.0, abs=2e-5)


def test_arbitrary_resolution_stitch_selects_owner_and_backpropagates():
    transforms, intrinsics, _ = _cell_center_rig()
    values = torch.arange(1.0, 7.0, requires_grad=True)
    images = values[:, None, None, None].expand(6, 1, 16, 32)
    panorama, valid = differentiable_hard_seam_stitch(
        images, transforms, intrinsics, output_size=(8, 96),
        projection_model="calibrated")
    owner, _ = hard_seam_owner_map((8, 96))
    expected = values.detach()[owner]
    assert panorama.shape == (1, 8, 96)
    assert valid.shape == (1, 8, 96)
    assert valid.all()
    assert torch.allclose(panorama[0], expected, atol=2e-5)

    panorama.sum().backward()
    assert values.grad is not None
    assert torch.all(values.grad > 0.0)


@pytest.mark.parametrize("projection_model", ["calibrated", "onebev_legacy"])
def test_precomputed_grid_matches_convenience_stitch(projection_model):
    transforms, intrinsics, _ = _cell_center_rig()
    if projection_model == "onebev_legacy":
        # Match the legacy 70/110-degree scale at this 32-wide resolution.
        intrinsics[:, 0, 0] = 25.2
        intrinsics[:, 1, 1] = 25.2
        intrinsics[0, 0, 0] = 8.09
        intrinsics[0, 1, 1] = 8.09
    torch.manual_seed(19)
    images = torch.rand(2, 6, 3, 16, 32, requires_grad=True)
    rows, columns = half_canvas_pano_axes(
        half_size=(8, 24), right_trim_px=5)
    grid, owner, valid = build_owned_pano_sampling_grid(
        transforms, intrinsics, source_size=(16, 32),
        pano_rows=rows, pano_columns=columns,
        projection_model=projection_model, seam_erosion_px=1)
    precomputed, precomputed_valid = stitch_camera_views_from_owned_grid(
        images, grid, owner, valid)
    direct, direct_valid = differentiable_hard_seam_stitch(
        images, transforms, intrinsics,
        pano_rows=rows, pano_columns=columns,
        projection_model=projection_model, seam_erosion_px=1)
    assert torch.equal(precomputed_valid, direct_valid)
    # Direct construction repeats shared calibration over the batch before
    # einsum, while the cached path builds once and broadcasts; CPU kernels
    # differ by at most a couple of float32 ulps.
    assert torch.allclose(precomputed, direct, atol=3e-6, rtol=1e-6)


def test_legacy_grid_exposes_real_content_rays_and_origins():
    transforms, intrinsics, _ = _cell_center_rig()
    intrinsics[:, 0, 0] = 25.2
    intrinsics[:, 1, 1] = 25.2
    intrinsics[0, 0, 0] = 8.09
    intrinsics[0, 1, 1] = 8.09
    rows, columns = half_canvas_pano_axes(
        half_size=(8, 24), right_trim_px=5)
    grid, owner, valid = build_owned_pano_sampling_grid(
        transforms, intrinsics, source_size=(16, 32),
        pano_rows=rows, pano_columns=columns,
        projection_model="onebev_legacy", legacy_mapping_size=None)
    origins, directions, z_cosine = sampling_grid_to_ego_rays(
        grid, owner, transforms, intrinsics, source_size=(16, 32))
    assert torch.allclose(
        torch.linalg.vector_norm(directions[valid], dim=-1),
        torch.ones_like(z_cosine[valid]), atol=2e-5)
    assert torch.all(z_cosine[valid] > 0.0)
    for camera_index in range(6):
        camera_mask = (owner == camera_index) & valid
        assert torch.allclose(
            origins[camera_mask],
            transforms[camera_index, :3, 3].expand(
                int(camera_mask.sum()), 3))


def test_legacy_grid_matches_public_onebev_equations():
    """Numbers independently transcribed from im2sphere/warp_image_fast."""
    transforms, intrinsics, _ = _cell_center_rig()
    rows = torch.tensor([440.0, 460.0])
    columns = torch.tensor([3200.0, 3400.0])
    grid, owner, valid = build_owned_pano_sampling_grid(
        transforms, intrinsics, source_size=(16, 32),
        pano_rows=rows, pano_columns=columns,
        projection_model="onebev_legacy", legacy_mapping_size=None)
    assert torch.all(owner == 2)
    assert valid.all()
    expected = torch.tensor([
        [[-0.00171834, 0.22064996], [0.18658888, 0.22073674]],
        [[-0.00171834, 0.27655470], [0.18658888, 0.27647615]],
    ])
    # Each output row/column forms a Cartesian mesh.  The hard-coded values
    # include OneBEV's +1 crop/remap shift and radians-passed-as-degrees roll.
    assert torch.allclose(grid, expected, atol=8e-5)


def test_native_legacy_mapping_is_invariant_to_render_resolution():
    transforms, intrinsics, _ = _cell_center_rig()
    rows = torch.tensor([440.0, 460.0])
    columns = torch.tensor([3200.0, 3400.0])
    grid_small, owner_small, valid_small = build_owned_pano_sampling_grid(
        transforms, intrinsics, source_size=(16, 32),
        pano_rows=rows, pano_columns=columns,
        projection_model="onebev_legacy")
    large_intrinsics = resize_intrinsics(
        intrinsics, source_size=(16, 32), target_size=(360, 640))
    grid_large, owner_large, valid_large = build_owned_pano_sampling_grid(
        transforms, large_intrinsics, source_size=(360, 640),
        pano_rows=rows, pano_columns=columns,
        projection_model="onebev_legacy")
    assert torch.equal(owner_small, owner_large)
    assert torch.equal(valid_small, valid_large)
    assert torch.allclose(grid_small, grid_large, atol=2e-7, rtol=2e-7)


def test_nuscenes_3d_to_gsplat_grid_ray_golden():
    """Golden chain: nuScenes K projection -> gsplat -> grid -> K^-1 ray."""
    # Exact CAM_FRONT K from the first train_p2 rig. nuScenes view_points uses
    # this raw matrix and later rounds projected coordinates to image indices.
    k_cv = torch.tensor([
        [1266.417203046554, 0.0, 816.2670197447984],
        [0.0, 1266.417203046554, 491.50706579294757],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float64)
    points_camera = torch.tensor([
        [0.0, 0.0, 8.0],
        [1.0, -0.5, 12.0],
        [-2.0, 1.0, 18.0],
        [3.0, 2.0, 25.0],
        [-1.5, -1.0, 10.0],
    ], dtype=torch.float64)
    native_size = (900, 1600)
    render_size = (360, 640)
    scale_xy = torch.tensor([0.4, 0.4], dtype=torch.float64)

    # Official nuScenes view_points: K@X followed by division by camera z.
    projected_cv = torch.einsum(
        "ij,nj->ni", k_cv, points_camera)
    uv_cv = projected_cv[:, :2] / projected_cv[:, 2:3]

    k_gs = prepare_gsplat_intrinsics(k_cv, native_size, render_size)
    try:
        # Exercise gsplat itself whenever the test runs in the production
        # environment.  The lightweight local test environment does not ship
        # its compiled extension, so retain the exact official torch fallback.
        from gsplat.cuda._torch_impl import _persp_proj
    except ImportError:
        projected_gs = torch.einsum(
            "ij,nj->ni", k_gs, points_camera)
        uv_gs = projected_gs[:, :2] / projected_gs[:, 2:3]
    else:
        means2d, _ = _persp_proj(
            points_camera[None, None],
            torch.eye(3, dtype=torch.float64).expand(
                1, 1, len(points_camera), 3, 3).clone(),
            k_gs[None, None],
            width=render_size[1], height=render_size[0],
        )
        uv_gs = means2d[0, 0]
    expected_pixel_centers = (uv_cv + 0.5) * scale_xy
    assert torch.max(torch.abs(uv_gs - expected_pixel_centers)).item() < 1e-12

    # align_corners=False normalized coordinates encode image-edge positions.
    render_height, render_width = render_size
    grid = torch.stack([
        uv_gs[:, 0] * (2.0 / render_width) - 1.0,
        uv_gs[:, 1] * (2.0 / render_height) - 1.0,
    ], dim=-1)[None]
    owner = torch.full((1, len(points_camera)), 3, dtype=torch.long)
    transforms = torch.eye(4, dtype=torch.float64).repeat(6, 1, 1)
    intrinsics = k_gs.repeat(6, 1, 1)
    _, recovered_rays, _ = sampling_grid_to_ego_rays(
        grid, owner, transforms, intrinsics, source_size=render_size)
    expected_rays = torch.nn.functional.normalize(points_camera, dim=-1)
    assert torch.max(torch.abs(
        recovered_rays[0] - expected_rays)).item() < 2e-13

    reprojected = torch.einsum(
        "ij,nj->ni", k_gs, recovered_rays[0])
    uv_roundtrip = reprojected[:, :2] / reprojected[:, 2:3]
    assert torch.max(torch.abs(uv_roundtrip - uv_gs)).item() < 2e-12

    # The disputed direct row scaling is observably wrong: at 640x360 it
    # misses 0.5*0.4 = 0.2 output pixel on both axes.
    naive_k = resize_intrinsics(k_cv, native_size, render_size)
    naive_projected = torch.einsum(
        "ij,nj->ni", naive_k, points_camera)
    naive_uv = naive_projected[:, :2] / naive_projected[:, 2:3]
    assert torch.max(torch.abs(
        naive_uv - expected_pixel_centers)).item() == pytest.approx(0.2)


def test_opencv_remap_integer_center_matches_grid_sample_plus_half():
    cv2 = pytest.importorskip("cv2")
    image = np.zeros((12, 24), dtype=np.float32)
    image[6, 10] = 1.0
    remapped = cv2.remap(
        image,
        np.asarray([[10.0]], dtype=np.float32),
        np.asarray([[6.0]], dtype=np.float32),
        interpolation=cv2.INTER_LINEAR,
    )
    assert remapped.item() == pytest.approx(1.0)

    source = torch.from_numpy(image)[None, None]
    correct_grid = torch.tensor([[[[
        (10.0 + 0.5) * 2.0 / 24.0 - 1.0,
        (6.0 + 0.5) * 2.0 / 12.0 - 1.0,
    ]]]])
    wrong_grid = torch.tensor([[[[
        10.0 * 2.0 / 24.0 - 1.0,
        6.0 * 2.0 / 12.0 - 1.0,
    ]]]])
    correct = torch.nn.functional.grid_sample(
        source, correct_grid, align_corners=False)
    wrong = torch.nn.functional.grid_sample(
        source, wrong_grid, align_corners=False)
    assert correct.item() == pytest.approx(1.0, abs=2e-6)
    assert wrong.item() == pytest.approx(0.25, abs=2e-6)
