"""Regression tests for causal and calibrated temporal Gaussian geometry."""

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from dwm_pano.datasets.pano_projection import (
    VORONOI_BOUNDARIES,
    pano_column_camera_indices,
)
from dwm_pano.models.gs_decoder_pano import (
    GAUSSIAN_PARAMETER_SPLIT,
    PanoGaussianDecoder,
    TemporalDeltaAdapter,
)
from dwm_pano.pipelines.ctsd_pano_gs_decoder import (
    PanoGaussianDecoderPipeline,
    alpha_coverage_loss,
    cross_render_visibility_mask,
    cross_target_validity_mask,
    decoder_option,
    masked_mean,
    sample_frame_indices,
    select_cross_photometric_mask,
)
from dwm_pano.generated_latent_cache import (
    atomic_torch_save,
    latent_cache_path,
    make_cache_payload,
    validate_cache_payload,
)


def test_temporal_delta_adapter_is_exactly_zero_initialized():
    adapter = TemporalDeltaAdapter(latent_channels=4, hidden_channels=8)
    current = torch.randn(2, 2, 4, 5, 7)
    previous = torch.randn_like(current)
    output = adapter(current, previous)
    assert torch.equal(output, torch.zeros_like(output))


def test_supervision_output_label_is_derived_from_geometry():
    pipeline = object.__new__(PanoGaussianDecoderPipeline)
    pipeline.uses_nuscenes6_stitch = True
    pipeline.training_config = {"half_width": 2304, "half_height": 256}
    assert pipeline.supervision_output_label() == \
        "pano4114x256-real6stitch"

    pipeline.uses_nuscenes6_stitch = False
    pipeline.training_config = {
        "target_yaws_deg": [0, 60, 120, 180],
        "target_width": 640,
        "target_height": 360,
    }
    assert pipeline.supervision_output_label() == "4view-640x360"


def test_decoder_option_accepts_config_and_instantiated_module():
    config = {
        "use_camera_origins": True,
        "ray_embedding_mode": "plucker",
    }
    module = SimpleNamespace(**config)
    for decoder in (config, module):
        assert decoder_option(decoder, "use_camera_origins") is True
        assert decoder_option(decoder, "ray_embedding_mode") == "plucker"
        assert decoder_option(decoder, "missing", "fallback") == "fallback"


def test_display_mask_is_identical_for_gt_and_prediction():
    pipeline = object.__new__(PanoGaussianDecoderPipeline)
    pipeline.uses_nuscenes6_stitch = True
    pipeline._calibrated_supervision_ready = True
    pipeline.stitch_valid = torch.tensor([
        [True, False],
        [True, True],
    ])
    ground_truth = torch.ones(1, 3, 2, 2)
    prediction = torch.full_like(ground_truth, 0.5)
    masked_gt = pipeline.mask_supervision_for_display(ground_truth)
    masked_prediction = pipeline.mask_supervision_for_display(prediction)
    assert torch.equal(masked_gt[:, :, 0, 1], torch.zeros(1, 3))
    assert torch.equal(masked_prediction[:, :, 0, 1], torch.zeros(1, 3))
    assert torch.equal(masked_gt[:, :, 1, 1], torch.ones(1, 3))
    assert torch.equal(
        masked_prediction[:, :, 1, 1], torch.full((1, 3), 0.5))


def test_temporal_delta_adapter_has_no_future_dependency():
    torch.manual_seed(7)
    adapter = TemporalDeltaAdapter(latent_channels=4, hidden_channels=8)
    torch.nn.init.normal_(adapter.conv_out.weight)
    sequence = torch.randn(1, 4, 2, 4, 5, 7)

    def run(values):
        return torch.stack([
            adapter(values[:, index], values[:, max(0, index - 1)])
            for index in range(values.shape[1])
        ], dim=1)

    baseline = run(sequence)
    changed = sequence.clone()
    changed[:, 3] += 10.0
    perturbed = run(changed)
    assert torch.equal(baseline[:, :3], perturbed[:, :3])
    assert not torch.equal(baseline[:, 3], perturbed[:, 3])


def test_adjacent_pair_sampler_returns_disjoint_ordered_pairs():
    pipeline = object.__new__(PanoGaussianDecoderPipeline)
    pipeline.training_config = {
        "frames_per_step": 4,
        "frame_sampling": "adjacent_pairs",
    }
    pipeline.generator = torch.Generator().manual_seed(11)
    for _ in range(32):
        indices = pipeline._select_frame_indices(19)
        pairs = indices.reshape(-1, 2)
        assert torch.equal(pairs[:, 1], pairs[:, 0] + 1)
        assert len(torch.unique(indices)) == len(indices)
        assert int(indices.min()) >= 0
        assert int(indices.max()) < 19


def test_shared_frame_sampler_matches_pipeline_method():
    direct_generator = torch.Generator().manual_seed(23)
    pipeline = object.__new__(PanoGaussianDecoderPipeline)
    pipeline.training_config = {
        "frames_per_step": 4,
        "frame_sampling": "random",
    }
    pipeline.generator = torch.Generator().manual_seed(23)
    for _ in range(8):
        direct = sample_frame_indices(
            19, 4, "random", direct_generator)
        assert torch.equal(direct, pipeline._select_frame_indices(19))


def test_generated_latent_cache_rejects_frame_mismatch(tmp_path):
    frame_indices = torch.tensor([1, 4, 8, 12])
    timestamps = torch.tensor([101, 104, 108, 112])
    latents = torch.randn(4, 2, 16, 3, 5)
    payload = make_cache_payload(
        dataset_index=7,
        anchor_timestamp_us=100,
        frame_indices=frame_indices,
        frame_timestamps_us=timestamps,
        latents=latents,
    )
    path = latent_cache_path(tmp_path, 7, 100)
    atomic_torch_save(payload, path)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    validated = validate_cache_payload(
        restored,
        dataset_index=7,
        anchor_timestamp_us=100,
        frame_indices=frame_indices,
        frame_timestamps_us=timestamps,
    )
    assert validated.dtype == torch.float16
    assert torch.allclose(validated.float(), latents, atol=1e-3)

    try:
        validate_cache_payload(
            restored,
            dataset_index=7,
            anchor_timestamp_us=100,
            frame_indices=torch.tensor([1, 4, 8, 13]),
            frame_timestamps_us=timestamps,
        )
    except RuntimeError as error:
        assert "frame indices" in str(error)
    else:
        raise AssertionError("mismatched cache frames were silently accepted")


def test_pipeline_loads_exact_cached_stage2_latents(tmp_path):
    pipeline = object.__new__(PanoGaussianDecoderPipeline)
    pipeline.latent_domain_adaptation_config = {
        "enabled": True,
        "cache_dir": str(tmp_path),
        "probability": 1.0,
        "strict": True,
    }
    pipeline.latent_domain_generator = torch.Generator().manual_seed(9)
    pipeline.device = torch.device("cpu")
    pipeline.latent_directions = torch.zeros(2, 3, 2, 2)
    frame_indices = torch.tensor([0, 3, 7, 18])
    all_timestamps = torch.arange(19) + 1000
    latents = torch.randn(4, 2, 16, 2, 2)
    payload = make_cache_payload(
        dataset_index=5,
        anchor_timestamp_us=1000,
        frame_indices=frame_indices,
        frame_timestamps_us=all_timestamps[frame_indices],
        latents=latents,
    )
    atomic_torch_save(
        payload, latent_cache_path(tmp_path, 5, 1000))
    batch = {
        "vae_images": torch.zeros(1, 19, 2, 3, 4, 4),
        "dataset_index": torch.tensor([5]),
        "anchor_timestamp_us": torch.tensor([1000]),
        "frame_timestamps_us": all_timestamps.unsqueeze(0),
    }
    restored = pipeline._maybe_load_generated_latents(batch, frame_indices)
    assert restored.dtype == pipeline.latent_directions.dtype
    assert torch.allclose(restored, latents, atol=1e-3)


def test_frame_sampler_rejects_silent_truncation():
    pipeline = object.__new__(PanoGaussianDecoderPipeline)
    pipeline.training_config = {
        "frames_per_step": 20,
        "frame_sampling": "random",
    }
    pipeline.generator = torch.Generator().manual_seed(11)
    try:
        pipeline._select_frame_indices(19)
    except ValueError as error:
        assert "exceeds frame_count" in str(error)
    else:
        raise AssertionError("oversized frame sample was silently truncated")


def test_target_camera_is_expressed_in_source_ego_coordinates():
    pipeline = object.__new__(PanoGaussianDecoderPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.target_c2w = torch.eye(4).repeat(2, 1, 1)
    ego_transforms = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(
        1, 2, 1, 1, 1)
    ego_transforms[0, 1, 0, 0, 3] = 2.0
    result = pipeline._target_cameras_in_source_ego(
        {"ego_transforms": ego_transforms}, 0, 0, 1)
    assert torch.equal(result[:, :3, 3], torch.tensor([
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]))


def test_hard_seam_camera_owners_match_voronoi_boundaries():
    columns = np.array([
        -1, 0, VORONOI_BOUNDARIES[0] - 1,
        VORONOI_BOUNDARIES[0], VORONOI_BOUNDARIES[1],
        VORONOI_BOUNDARIES[-1] - 1, VORONOI_BOUNDARIES[-1],
    ])
    owners = pano_column_camera_indices(columns)
    assert owners.tolist() == [0, 0, 0, 1, 2, 5, 0]


def test_frame_ray_origins_expand_six_camera_calibration():
    pipeline = object.__new__(PanoGaussianDecoderPipeline)
    pipeline.decoder = SimpleNamespace(use_camera_origins=True)
    pipeline.device = torch.device("cpu")
    pipeline.pixel_directions = torch.zeros(2, 2, 3, 3)
    pipeline.latent_directions = torch.zeros(2, 3, 1, 2)
    pipeline.pixel_camera_indices = torch.tensor([[0, 1, 2], [3, 4, 5]])
    pipeline.latent_camera_indices = torch.tensor([[5, 4], [1, 0]])
    translations = torch.arange(18, dtype=torch.float32).reshape(1, 1, 6, 3)

    latent, pixel = pipeline._frame_ray_origins(
        {"pano_camera_translations": translations}, torch.tensor([0]))
    assert pixel.shape == (1, 2, 2, 3, 3)
    assert latent.shape == (1, 2, 3, 1, 2)
    assert torch.equal(pixel[0, 0, 0, 1], translations[0, 0, 1])
    assert torch.equal(pixel[0, 1, 1, 2], translations[0, 0, 5])
    assert torch.equal(latent[0, 0, :, 0, 0], translations[0, 0, 5])


def test_legacy_input_checkpoint_is_zero_padded_for_plucker_rays():
    legacy = PanoGaussianDecoder(
        latent_channels=4,
        block_out_channels=(8,),
        layers_per_block=1,
        norm_num_groups=4,
        attention_block_count=0,
        ray_embedding_mode="direction",
    )
    plucker = PanoGaussianDecoder(
        latent_channels=4,
        block_out_channels=(8,),
        layers_per_block=1,
        norm_num_groups=4,
        attention_block_count=0,
        ray_embedding_mode="plucker",
    )
    state, migrated = plucker.migrate_input_checkpoint(legacy.state_dict())
    key = "decoder.conv_in.weight"
    old_channels = legacy.decoder.conv_in.weight.shape[1]
    assert migrated == [key]
    assert torch.equal(
        state[key][:, :old_channels], legacy.state_dict()[key])
    assert torch.count_nonzero(state[key][:, old_channels:]) == 0


class _ConstantRawDecoder(nn.Module):
    def forward(self, values):
        batch_size, view_count, _, height, width = values.shape
        return torch.zeros(
            batch_size * view_count, sum(GAUSSIAN_PARAMETER_SPLIT),
            height, width,
            device=values.device, dtype=values.dtype)


def test_gaussian_means_and_plucker_channels_use_real_origins():
    model = PanoGaussianDecoder(
        latent_channels=1,
        block_out_channels=(8,),
        layers_per_block=1,
        norm_num_groups=4,
        attention_block_count=0,
        use_range_prior=True,
        use_camera_origins=True,
        ray_embedding_mode="plucker",
    )
    model.decoder = _ConstantRawDecoder()
    latents = torch.zeros(1, 1, 1, 1, 1)
    directions_chw = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 1, 3, 1, 1)
    origins_chw = torch.tensor([0.5, 1.0, 1.5]).reshape(1, 1, 3, 1, 1)
    directions_hwc = directions_chw.permute(0, 1, 3, 4, 2)
    origins_hwc = origins_chw.permute(0, 1, 3, 4, 2)
    gaussians, _, _ = model(
        latents,
        directions_chw,
        directions_hwc,
        prior_range=torch.full((1, 1, 1, 1), 2.0),
        latent_origins=origins_chw,
        pixel_origins=origins_hwc,
    )
    assert torch.allclose(
        gaussians[0, 0, :3], torch.tensor([2.5, 1.0, 1.5]))
    legacy_geometry, _, _ = model(
        latents,
        directions_chw,
        directions_hwc,
        prior_range=torch.full((1, 1, 1, 1), 2.0),
        latent_origins=origins_chw,
        pixel_origins=origins_hwc,
        camera_origin_scale=0.0,
    )
    assert torch.allclose(
        legacy_geometry[0, 0, :3], torch.tensor([2.0, 0.0, 0.0]))


def test_visibility_rejects_holes_occlusion_and_low_alpha():
    rendered_depth = torch.tensor([5.0, 9.0, 5.0, 5.0])
    target_depth = torch.tensor([5.1, 5.0, 5.0, 5.0])
    alpha = torch.tensor([1.0, 1.0, 0.01, 1.0])
    target_static = torch.tensor([True, True, True, True])
    target_valid = torch.tensor([True, True, True, False])
    visible = cross_render_visibility_mask(
        rendered_depth, target_depth, alpha, target_static, target_valid,
        min_alpha=0.05, absolute_tolerance_m=0.5,
        relative_tolerance=0.0, near_plane=0.1)
    assert visible.tolist() == [True, False, False, False]


def test_experimental_target_mask_ignores_candidate_holes():
    target_depth = torch.tensor([5.0, 5.0, float("nan"), 5.0])
    target_static = torch.tensor([True, True, True, False])
    target_valid = torch.tensor([True, True, True, True])
    target_support = cross_target_validity_mask(
        target_depth, target_static, target_valid, near_plane=0.1)
    # Simulate a candidate that makes every valid target pixel invisible.
    rendered_visibility = torch.zeros_like(target_support)
    target_mask = select_cross_photometric_mask(
        target_support, rendered_visibility, "target_static_valid")
    legacy = select_cross_photometric_mask(
        target_support, rendered_visibility, "render_visibility")
    assert target_mask.tolist() == [True, True, False, False]
    assert not legacy.any()


def test_alpha_coverage_loss_penalizes_holes_on_independent_support():
    alpha = torch.tensor([0.0, 0.25, 0.75], requires_grad=True)
    support = torch.tensor([True, True, True])
    loss = alpha_coverage_loss(alpha, support, target_alpha=0.5)
    assert torch.allclose(loss, torch.tensor(0.5))
    loss.backward()
    assert alpha.grad[0] < 0.0
    assert alpha.grad[1] < 0.0
    assert alpha.grad[2] == 0.0


def test_experimental_target_support_cannot_hide_error_with_alpha_hole():
    pipeline = object.__new__(PanoGaussianDecoderPipeline)
    pipeline.training_config = {
        "near_plane": 0.1,
        "cross_min_alpha": 0.05,
        "cross_depth_abs_tolerance_m": 0.5,
        "cross_depth_rel_tolerance": 0.0,
        "cross_charbonnier_epsilon": 1e-3,
        "cross_ssim_weight": 0.0,
        "cross_photometric_mask": "target_static_valid",
        "cross_target_alpha": 0.5,
    }
    cross_rgb = torch.zeros(1, 3, 1, 2)
    cross_depth = torch.tensor([[[5.0, 50.0]]])
    cross_alpha = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    pipeline.render_targets = lambda *args, **kwargs: (
        cross_rgb, cross_depth, cross_alpha)
    pipeline._stitch_canvas = lambda values: values
    pipeline._resample_targets = lambda values, mode="bilinear": values
    seen_scale = []

    def target_depths(ranges, valid, origins, camera_origin_scale=1.0):
        seen_scale.append(camera_origin_scale)
        return torch.full((1, 1, 1, 2), 5.0), torch.ones(
            1, 1, 1, 2, dtype=torch.bool)

    pipeline._target_view_depths = target_depths
    gaussians = torch.zeros(1, 2, 14)
    source_segments = torch.zeros(1, 1, 1, 2, dtype=torch.long)
    target_images = torch.zeros(1, 3, 1, 2)
    target_images[:, :, :, 1] = 1.0
    result = pipeline._cross_time_rgb_loss(
        gaussians,
        source_segments,
        target_images,
        torch.ones(1, 1, 1, 2),
        torch.ones(1, 1, 1, 2, dtype=torch.bool),
        torch.zeros(1, 1, 1, 2, dtype=torch.long),
        torch.eye(4).unsqueeze(0),
        torch.zeros(1, 1, 1, 2, 3),
        camera_origin_scale=0.25,
    )
    assert seen_scale == [0.25]
    assert torch.allclose(result["rgb_l1"], torch.tensor(0.5))
    assert torch.equal(result["rgb_l1_visible"], torch.tensor(0.0))
    assert torch.allclose(result["alpha_coverage"], torch.tensor(0.5))
    assert torch.allclose(result["alpha_missing"], torch.tensor(0.5))


def test_target_depth_uses_the_same_scaled_camera_origins_as_means():
    pipeline = object.__new__(PanoGaussianDecoderPipeline)
    pipeline.pixel_directions = torch.tensor(
        [[[[0.0, 0.0, 1.0]]]], dtype=torch.float32)
    pipeline.target_c2w = torch.eye(4).unsqueeze(0)
    pipeline._resample_masked_targets = lambda values, valid: (values, valid)
    half_ranges = torch.full((1, 1, 1, 1), 2.0)
    half_valid = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    pixel_origins = torch.tensor([[[[[0.0, 0.0, 4.0]]]]])

    depth_zero, _ = pipeline._target_view_depths(
        half_ranges, half_valid, pixel_origins, camera_origin_scale=0.0)
    depth_half, _ = pipeline._target_view_depths(
        half_ranges, half_valid, pixel_origins, camera_origin_scale=0.5)
    depth_full, _ = pipeline._target_view_depths(
        half_ranges, half_valid, pixel_origins, camera_origin_scale=1.0)
    assert torch.equal(depth_zero, torch.full_like(depth_zero, 2.0))
    assert torch.equal(depth_half, torch.full_like(depth_half, 4.0))
    assert torch.equal(depth_full, torch.full_like(depth_full, 6.0))


def test_masked_mean_ignores_depth_hole_fill():
    values = torch.tensor([1.0, 99.0])
    valid = torch.tensor([True, False])
    assert torch.equal(masked_mean(values, valid), torch.tensor(1.0))


def _calibration_batch(rig_offset=0.0):
    transforms = torch.eye(4).repeat(6, 1, 1)
    transforms[:, 0, 3] = torch.linspace(-0.5, 0.5, 6) + rig_offset
    intrinsics = torch.tensor([
        [25.0, 0.0, 16.0],
        [0.0, 25.0, 8.0],
        [0.0, 0.0, 1.0],
    ]).repeat(6, 1, 1)
    image_sizes = torch.tensor([[32, 16]]).repeat(6, 1)
    return {
        "pano_camera_transforms": transforms[None, None].repeat(
            1, 2, 1, 1, 1),
        "pano_camera_intrinsics": intrinsics[None, None].repeat(
            1, 2, 1, 1, 1),
        "pano_camera_image_size": image_sizes[None, None].repeat(
            1, 2, 1, 1),
    }


def test_nuscenes6_supervision_caches_and_reactivates_multiple_rigs():
    pipeline = object.__new__(PanoGaussianDecoderPipeline)
    pipeline.uses_nuscenes6_stitch = True
    pipeline._calibrated_supervision_ready = False
    pipeline._active_calibration_signature = None
    pipeline._calibrated_supervision_cache = {}
    pipeline.device = torch.device("cpu")
    pipeline.should_save = False
    pipeline.vae_downsample = 8
    pipeline.training_config = {
        "half_height": 16,
        "half_width": 512,
        "source_camera_render_height": 16,
        "source_camera_render_width": 32,
        "stitch_projection_model": "onebev_legacy",
        "stitch_seam_erosion_px": 0,
        "max_calibration_cache_entries": 3,
    }

    rig_a = _calibration_batch(0.0)
    rig_b = _calibration_batch(0.1)
    pipeline._prepare_nuscenes6_supervision(rig_a)
    signature_a = pipeline._active_calibration_signature
    directions_a = pipeline.pixel_directions
    pipeline._prepare_nuscenes6_supervision(rig_b)
    signature_b = pipeline._active_calibration_signature
    assert signature_a != signature_b
    assert len(pipeline._calibrated_supervision_cache) == 2

    pipeline._prepare_nuscenes6_supervision(rig_a)
    assert pipeline._active_calibration_signature == signature_a
    assert pipeline.pixel_directions is directions_a


def test_nuscenes6_supervision_rejects_mixed_rig_inside_one_batch():
    pipeline = object.__new__(PanoGaussianDecoderPipeline)
    pipeline.uses_nuscenes6_stitch = True
    pipeline._calibrated_supervision_ready = False
    pipeline._active_calibration_signature = None
    pipeline._calibrated_supervision_cache = {}
    pipeline.device = torch.device("cpu")
    pipeline.should_save = False
    pipeline.vae_downsample = 8
    pipeline.training_config = {
        "half_height": 16,
        "half_width": 512,
        "source_camera_render_height": 16,
        "source_camera_render_width": 32,
        "stitch_projection_model": "onebev_legacy",
    }
    mixed = _calibration_batch(0.0)
    mixed["pano_camera_intrinsics"][0, 1, 0, 0, 0] += 1.0
    try:
        pipeline._prepare_nuscenes6_supervision(mixed)
    except ValueError as error:
        assert "changes across B/T" in str(error)
    else:
        raise AssertionError("mixed calibration was silently cached")


def test_nuscenes6_target_depth_and_validity_use_content_geometry():
    pipeline = object.__new__(PanoGaussianDecoderPipeline)
    pipeline.uses_nuscenes6_stitch = True
    pipeline.half_range_correction = torch.full((2, 1, 500), 2.0)
    pipeline.pixel_content_ray_valid = torch.ones(
        2, 1, 500, dtype=torch.bool)
    pipeline.pixel_content_ray_valid[0, 0, 4] = False
    pipeline.stitch_valid = torch.ones(1, 506, dtype=torch.bool)
    pipeline.stitch_valid[0, 2] = False
    frame_images = torch.rand(1, 2, 3, 1, 500)
    frame_ranges = torch.full((1, 2, 1, 500), 10.0)
    frame_valid = torch.ones_like(frame_ranges, dtype=torch.bool)

    target_rgb, target_depth, target_valid = \
        pipeline._build_own_supervision_targets(
            frame_images, frame_ranges, frame_valid,
            frame_pixel_origins=None, camera_origin_scale=1.0)
    assert target_rgb.shape == (1, 3, 1, 506)
    assert target_depth.shape == (1, 1, 506)
    assert torch.allclose(target_depth, torch.full_like(target_depth, 5.0))
    assert not target_valid[0, 0, 2]
    assert not target_valid[0, 0, 4]
    assert target_valid[0, 0, 5]
