"""GS decoder training pipeline downstream of the stage-2 CD world model.

Consumes the SAME video batches as the CD stage (`PanoNuScenesDataset` via
`vae_images` of shape (B, T, V=2, 3, H, W)); each step trains the decoder on
`frames_per_step` sampled frames. The frozen SD3.5 VAE encodes halves into
the latent space the stage-2 model generates, so generated latents plug in
at inference. Range priors come from the precomputed pano depth h5 files
looked up by frame timestamp.  Two supervision protocols are supported:
the legacy ego-centred virtual rig, and a geometry-aligned path that renders
the six real nuScenes cameras before applying the fixed OneBEV hard seam.
"""
import contextlib
import hashlib
import os
import time

import diffusers
import h5py
import hdf5plugin
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image

import dwm.common
import dwm.utils.preview
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from dwm_pano.calibrated_stitch import (
    build_owned_pano_sampling_grid,
    half_canvas_pano_axes,
    prepare_gsplat_intrinsics,
    sampling_grid_to_ego_rays,
    stitch_camera_views_from_owned_grid,
)
from dwm_pano.datasets.pano_projection import (
    PANO_CROP_TOP, PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE, PANO_H,
    PANO_W,
)
from dwm_pano.gs_geometry import (
    PANO_LAT_HALF_RAD, build_virtual_camera, pano_half_camera_indices,
    pano_half_directions, pano_source_camera_cosine_map,
    virtual_pixel_directions_ego,
)
from dwm_pano.generated_latent_cache import (
    latent_cache_path,
    scalar_int,
    validate_cache_payload,
)

# Resized-half overlap in columns: round(1152 / 5376 * 2304) = 494.
PANO_CANVAS_OVERLAP_COLUMNS = 494
SUPERVISION_VIRTUAL_CENTER = "virtual_center"
SUPERVISION_NUSCENES6_STITCH = "nuscenes6_onebev_stitch"
SUPPORTED_SUPERVISION_GEOMETRIES = (
    SUPERVISION_VIRTUAL_CENTER,
    SUPERVISION_NUSCENES6_STITCH,
)


def masked_mean(values, valid_mask):
    """Mean over a boolean/float mask, returning zero for an empty mask."""
    valid = valid_mask.to(device=values.device, dtype=values.dtype)
    while valid.ndim < values.ndim:
        valid = valid.unsqueeze(1)
    valid = torch.broadcast_to(valid, values.shape)
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def cross_render_visibility_mask(
    rendered_depth,
    target_depth,
    rendered_alpha,
    target_static,
    target_valid,
    min_alpha,
    absolute_tolerance_m,
    relative_tolerance,
    near_plane,
):
    """Occlusion-aware mask shared by training and paired evaluation."""
    tolerance = absolute_tolerance_m \
        + relative_tolerance * target_depth.abs()
    return target_static.bool() \
        & target_valid.bool() \
        & torch.isfinite(rendered_depth) \
        & torch.isfinite(target_depth) \
        & (target_depth > near_plane) \
        & (rendered_alpha >= min_alpha) \
        & ((rendered_depth - target_depth).abs() <= tolerance)


def cross_target_validity_mask(
    target_depth, target_static, target_valid, near_plane
):
    """Candidate-independent pixels that cross-render training may supervise.

    Unlike :func:`cross_render_visibility_mask`, this mask deliberately has no
    dependency on rendered alpha or rendered depth. It therefore cannot be
    reduced by making a candidate transparent or moving its geometry away.
    """
    return target_static.bool() \
        & target_valid.bool() \
        & torch.isfinite(target_depth) \
        & (target_depth > near_plane)


def select_cross_photometric_mask(
    target_validity, rendered_visibility, mode
):
    """Select the cross-render optimization mask with an explicit policy."""
    if mode == "render_visibility":
        return rendered_visibility
    if mode == "target_static_valid":
        return target_validity
    raise ValueError(
        "cross_photometric_mask must be 'render_visibility' or "
        "'target_static_valid', got {!r}".format(mode))


def alpha_coverage_loss(rendered_alpha, target_validity, target_alpha):
    """Penalize missing alpha on a candidate-independent target support.

    The normalization makes the loss live in [0, 1] for rasterizer alpha in
    [0, 1], independently of the configured target alpha.
    """
    if not 0.0 < target_alpha <= 1.0:
        raise ValueError("cross_target_alpha must be in (0, 1]")
    missing = (target_alpha - rendered_alpha).clamp_min(0.0) / target_alpha
    return masked_mean(missing, target_validity)


def ssim_loss_map(prediction, target, window_size=3):
    """Small differentiable SSIM dissimilarity map in [0, 1]."""
    padding = window_size // 2
    mean_prediction = F.avg_pool2d(
        prediction, window_size, stride=1, padding=padding)
    mean_target = F.avg_pool2d(
        target, window_size, stride=1, padding=padding)
    variance_prediction = F.avg_pool2d(
        prediction * prediction, window_size, stride=1, padding=padding
    ) - mean_prediction.square()
    variance_target = F.avg_pool2d(
        target * target, window_size, stride=1, padding=padding
    ) - mean_target.square()
    covariance = F.avg_pool2d(
        prediction * target, window_size, stride=1, padding=padding
    ) - mean_prediction * mean_target
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    score = ((2.0 * mean_prediction * mean_target + c1)
             * (2.0 * covariance + c2)) / (
                 (mean_prediction.square() + mean_target.square() + c1)
                 * (variance_prediction + variance_target + c2)
             ).clamp_min(1e-8)
    return ((1.0 - score) * 0.5).clamp(0.0, 1.0).mean(dim=1)


def decoder_option(decoder, name, default=None):
    """Read an option before/after recursive config instantiation.

    ``dwm.common.create_instance_from_config`` recursively constructs nested
    ``_class_name`` objects before calling the pipeline constructor.  Direct
    unit callers may still pass the original dictionary, so geometry guards
    deliberately accept both representations.
    """
    if isinstance(decoder, dict):
        return decoder.get(name, default)
    return getattr(decoder, name, default)


def sample_frame_indices(
    frame_count, frames_per_step, sampling, generator
):
    """Sample frames with one implementation shared by training and caching."""
    if frames_per_step <= 0:
        raise ValueError("frames_per_step must be positive")
    if frames_per_step > frame_count:
        raise ValueError(
            "frames_per_step={} exceeds frame_count={}".format(
                frames_per_step, frame_count))
    if sampling == "adjacent_pairs":
        if frames_per_step % 2 != 0:
            raise ValueError(
                "adjacent_pairs requires an even frames_per_step")
        pair_count = frames_per_step // 2
        parity = int(torch.randint(
            0, 2, (1,), generator=generator).item())
        starts = torch.arange(parity, frame_count - 1, 2)
        if len(starts) < pair_count:
            raise ValueError(
                "sequence has {} non-overlapping adjacent pairs, but {} "
                "are required".format(len(starts), pair_count))
        permutation = torch.randperm(
            len(starts), generator=generator)[:pair_count]
        starts = starts[permutation]
        return torch.stack([starts, starts + 1], dim=1).flatten()
    if sampling != "random":
        raise ValueError("Unsupported frame_sampling: {}".format(sampling))
    permutation = torch.randperm(frame_count, generator=generator)
    return permutation[:frames_per_step]


class PanoGaussianDecoderPipeline:

    def __init__(
        self,
        output_path,
        config: dict,
        device,
        training_config: dict,
        pretrained_model_name_or_path: str,
        decoder,
        latent_refinement: dict = None,
        latent_domain_adaptation: dict = None,
        initial_decoder_checkpoint_path=None,
        resume_from=None,
    ):
        self.should_save = not torch.distributed.is_initialized() or \
            torch.distributed.get_rank() == 0
        self.config = config
        self.device = device
        self.training_config = training_config
        # This experiment copy owns the rasterizer choice.  It intentionally
        # does not read PANDWM_GS_RASTERIZE_MODE: an environment override
        # would make the AA fine-tuning recipe irreproducible.
        self.rasterize_mode = training_config.get(
            "rasterize_mode", "classic")
        if self.rasterize_mode not in ("classic", "antialiased"):
            raise ValueError(
                "rasterize_mode must be 'classic' or 'antialiased', got "
                f"{self.rasterize_mode!r}")
        self.aa_eps2d_start = float(training_config.get(
            "aa_eps2d_start", 0.3))
        self.aa_eps2d_end = float(training_config.get(
            "aa_eps2d_end", self.aa_eps2d_start))
        self.aa_eps2d_ramp_steps = int(training_config.get(
            "aa_eps2d_ramp_steps", 0))
        if self.aa_eps2d_start < 0.0 or self.aa_eps2d_end < 0.0:
            raise ValueError("AA eps2d values must be non-negative")
        if self.aa_eps2d_ramp_steps < 0:
            raise ValueError("aa_eps2d_ramp_steps must be non-negative")
        # A freshly constructed pipeline is used for evaluation, where the
        # completed checkpoint must be rendered at the terminal AA strength.
        self.active_rasterize_eps2d = self.aa_eps2d_end
        self.supervision_geometry = training_config.get(
            "supervision_geometry", SUPERVISION_VIRTUAL_CENTER)
        if self.supervision_geometry not in SUPPORTED_SUPERVISION_GEOMETRIES:
            raise ValueError(
                "supervision_geometry must be one of {}, got {!r}".format(
                    SUPPORTED_SUPERVISION_GEOMETRIES,
                    self.supervision_geometry))
        self.uses_nuscenes6_stitch = self.supervision_geometry == \
            SUPERVISION_NUSCENES6_STITCH
        if self.uses_nuscenes6_stitch:
            if not training_config.get("range_is_source_camera_z", False):
                raise ValueError(
                    "nuscenes6 stitch supervision requires source-camera-z "
                    "depth so it can construct metric content-ray ranges")
            if not decoder_option(decoder, "use_camera_origins", False):
                raise ValueError(
                    "nuscenes6 stitch supervision requires calibrated camera "
                    "origins in the Gaussian decoder")
            if decoder_option(decoder, "ray_embedding_mode") != "plucker":
                raise ValueError(
                    "nuscenes6 stitch supervision requires Plucker rays")
            if training_config.get("camera_origin_warmup_steps", 0) != 0:
                raise ValueError(
                    "nuscenes6 stitch supervision uses the real camera "
                    "origins on both sides of the loss from step 0; "
                    "camera_origin_warmup_steps must be 0")
            unsupported_cross = {
                "weight_cross_rgb_l1": training_config.get(
                    "weight_cross_rgb_l1", 0.0),
                "weight_cross_alpha_coverage": training_config.get(
                    "weight_cross_alpha_coverage", 0.0),
                "cross_ssim_weight": training_config.get(
                    "cross_ssim_weight", 0.0),
            }
            enabled_cross = [
                key for key, value in unsupported_cross.items()
                if float(value) != 0.0
            ]
            if enabled_cross:
                raise ValueError(
                    "nuscenes6 stitch supervision does not yet have a "
                    "candidate-independent source-depth z-buffer "
                    "co-visibility mask; disable {} for the G2 own-view "
                    "geometry experiment".format(", ".join(enabled_cross)))
            if training_config.get("weight_perceptual", 0.0) > 0.0:
                perceptual_height, perceptual_width = training_config[
                    "perceptual_resolution"]
                canvas_aspect = (
                    2 * training_config["half_width"]
                    - PANO_CANVAS_OVERLAP_COLUMNS
                ) / training_config["half_height"]
                perceptual_aspect = perceptual_width / perceptual_height
                aspect_error = abs(
                    perceptual_aspect / canvas_aspect - 1.0)
                if aspect_error > 0.05:
                    raise ValueError(
                        "nuscenes6 stitched LPIPS must preserve the wide "
                        "canvas aspect ratio (within 5%); got resolution {} "
                        "for aspect {:.4f}".format(
                            training_config["perceptual_resolution"],
                            canvas_aspect))
        self._calibrated_supervision_ready = False
        self._active_calibration_signature = None
        self._calibrated_supervision_cache = {}
        self.latent_refinement_config = latent_refinement
        self.latent_domain_adaptation_config = latent_domain_adaptation
        self.generator = torch.Generator()
        self.generator.manual_seed(config["generator_seed"])
        self.latent_domain_generator = torch.Generator()
        latent_domain_seed = int(config["generator_seed"]) + 104729
        if latent_domain_adaptation is not None and \
                "generator_seed" in latent_domain_adaptation:
            latent_domain_seed = int(
                latent_domain_adaptation["generator_seed"])
        self.latent_domain_generator.manual_seed(latent_domain_seed)

        refiner_enabled = latent_refinement is not None \
            and latent_refinement.get("enabled", False)
        domain_enabled = latent_domain_adaptation is not None \
            and latent_domain_adaptation.get("enabled", False)
        if refiner_enabled and domain_enabled:
            raise ValueError(
                "latent_refinement and latent_domain_adaptation cannot both "
                "be enabled")
        if domain_enabled:
            probability = float(latent_domain_adaptation["probability"])
            if not 0.0 < probability <= 1.0:
                raise ValueError(
                    "latent_domain_adaptation.probability must be in (0, 1]")
            if not latent_domain_adaptation.get("cache_dir"):
                raise ValueError(
                    "latent_domain_adaptation.cache_dir is required")
            if training_config.get("weight_cross_rgb_l1", 0.0) != 0.0 \
                    or training_config.get(
                        "weight_cross_alpha_coverage", 0.0) != 0.0:
                raise ValueError(
                    "generated-latent adaptation currently requires cross-"
                    "frame losses to be disabled")

        self.vae = diffusers.AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae").to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.decoder = dwm.common.create_instance_from_config(decoder)
        self.decoder = self.decoder.to(device)
        if resume_from is not None:
            checkpoint_path = os.path.join(
                output_path, "checkpoints", f"{resume_from}.pth")
            self.decoder.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu"))
        elif initial_decoder_checkpoint_path is not None:
            initial_state = torch.load(
                initial_decoder_checkpoint_path, map_location="cpu")
            initial_state, migrated_keys = \
                self.decoder.migrate_input_checkpoint(initial_state)
            incompatible = self.decoder.load_state_dict(
                initial_state, strict=False)
            allowed_prefix = "temporal_delta_adapter."
            invalid_missing = [
                key for key in incompatible.missing_keys
                if not key.startswith(allowed_prefix)
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Initial decoder checkpoint is incompatible: missing={} "
                    "unexpected={}".format(
                        invalid_missing, incompatible.unexpected_keys))
            if self.should_save:
                print(
                    "Initialized decoder from {} (new parameters: {}).".format(
                        initial_decoder_checkpoint_path,
                        incompatible.missing_keys),
                    flush=True)
                if migrated_keys:
                    print(
                        "Zero-padded new ray channels in: {}".format(
                            migrated_keys),
                        flush=True)

        if torch.distributed.is_initialized():
            self.decoder_wrapper = torch.nn.parallel.DistributedDataParallel(
                self.decoder, device_ids=[device.index])
        else:
            self.decoder_wrapper = self.decoder

        half_height = training_config["half_height"]
        half_width = training_config["half_width"]
        downsample = 2 ** (len(self.vae.config.down_block_types) - 1)
        self.vae_downsample = downsample
        pixel_direction_list = []
        latent_direction_list = []
        pixel_camera_index_list = []
        latent_camera_index_list = []
        for column_range in (PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE):
            pixel_direction_list.append(pano_half_directions(
                column_range, half_height, half_width))
            latent_direction_list.append(pano_half_directions(
                column_range, half_height // downsample,
                half_width // downsample))
            # Ownership is constant down each pano column.  Keep only one row
            # and expand on demand to avoid a persistent ~10 MB index map.
            pixel_camera_index_list.append(pano_half_camera_indices(
                column_range, 1, half_width)[0])
            latent_camera_index_list.append(pano_half_camera_indices(
                column_range, 1, half_width // downsample)[0])
        self.pixel_directions = torch.from_numpy(
            np.stack(pixel_direction_list)).float().to(device)
        latent_directions = torch.from_numpy(
            np.stack(latent_direction_list)).float().to(device)
        self.latent_directions = latent_directions.permute(0, 3, 1, 2)
        self.pixel_camera_indices = torch.from_numpy(
            np.stack(pixel_camera_index_list)).long().to(device)
        self.latent_camera_indices = torch.from_numpy(
            np.stack(latent_camera_index_list)).long().to(device)
        self.pixel_content_ray_valid = torch.ones(
            self.pixel_directions.shape[:-1], dtype=torch.bool,
            device=device)
        self.latent_content_ray_valid = torch.ones(
            self.latent_directions.shape[0],
            self.latent_directions.shape[-2],
            self.latent_directions.shape[-1],
            dtype=torch.bool, device=device)

        self._build_target_cameras()
        # Stitched pano depth stores per-cell source-camera z; divide by
        # cos(ray, camera axis) to get Euclidean range along the pano ray.
        if training_config["range_is_source_camera_z"]:
            self.range_correction = 1.0 / pano_source_camera_cosine_map()
        else:
            self.range_correction = np.ones((PANO_H, PANO_W))
        # The geometry-aligned protocol replaces this midpoint-axis
        # approximation on the first real batch with exact content-ray
        # cosines recovered from the legacy stitch sampling grid.
        self.half_range_correction = None

        if training_config["weight_perceptual"] > 0.0:
            self.perceptual_metric = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=True).to(device)
            self.perceptual_metric.requires_grad_(False)
        else:
            self.perceptual_metric = None

        if latent_refinement is not None and latent_refinement["enabled"]:
            self._build_latent_refiner()
        else:
            self.refiner_model = None

        self.optimizer = dwm.common.create_instance_from_config(
            config["optimizer"], params=self.decoder_wrapper.parameters())
        if resume_from is not None:
            optimizer_path = os.path.join(
                output_path, "optimizer", f"{resume_from}.pth")
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location="cpu"))

        self.loss_report_list = []
        self.step_duration = 0.0

    def supervision_output_label(self):
        """Stable metric/benchmark label derived from the active geometry."""
        if self.uses_nuscenes6_stitch:
            canvas_width = 2 * self.training_config["half_width"] \
                - PANO_CANVAS_OVERLAP_COLUMNS
            canvas_height = self.training_config["half_height"]
            return "pano{}x{}-real6stitch".format(
                canvas_width, canvas_height)
        return "{}view-{}x{}".format(
            len(self.training_config["target_yaws_deg"]),
            self.training_config["target_width"],
            self.training_config["target_height"],
        )

    def mask_supervision_for_display(self, images):
        """Apply the fixed G2 stitch support equally to GT and prediction."""
        if not self.uses_nuscenes6_stitch:
            return images
        if not self._calibrated_supervision_ready:
            raise RuntimeError(
                "nuscenes6 supervision cache was not prepared from a batch")
        valid = self.stitch_valid.to(device=images.device)
        while valid.ndim < images.ndim:
            valid = valid.unsqueeze(0)
        return torch.where(valid, images, torch.zeros_like(images))

    def _build_latent_refiner(self):
        """Load the frozen CD DiT and its constant uncond conditions."""
        from dwm_pano.models.flex_block_causal import \
            make_blockwise_causal_block_mask
        from dwm_pano.pipelines.ctsd_pano import CrossviewTemporalSD

        refiner_config = self.latent_refinement_config
        self.refiner_model = dwm.common.create_instance_from_config(
            refiner_config["model"]).to(dtype=torch.float16)
        self.refiner_model.load_state_dict(
            CrossviewTemporalSD.load_state(refiner_config["checkpoint_path"]))
        self.refiner_model.requires_grad_(False)
        self.refiner_model = self.refiner_model.to(self.device)
        self.refiner_model.eval()

        pack = torch.load(
            refiner_config["uncond_conditions_path"], map_location="cpu")
        # No FSDP auto-cast here: float conditions must match fp16 weights.
        self.refiner_conditions = {}
        for key, value in pack.items():
            if value.is_floating_point():
                value = value.to(self.device, dtype=torch.float16)
            else:
                value = value.to(self.device)
            self.refiner_conditions[key] = value
        self._make_causal_mask = make_blockwise_causal_block_mask
        self._causal_mask_cache = {}

    def _refiner_causal_mask(self, frame_count, latent_height, latent_width):
        if frame_count not in self._causal_mask_cache:
            patch_size = self.refiner_model.config.patch_size
            frame_seqlen = (latent_height // patch_size) \
                * (latent_width // patch_size)
            self._causal_mask_cache[frame_count] = self._make_causal_mask(
                self.device, num_frames=frame_count,
                frame_seqlen=frame_seqlen)
        return self._causal_mask_cache[frame_count]

    @torch.no_grad()
    def refine_clip_latents(self, clip_latents):
        """Noise the clip and k-step denoise it with the frozen CD DiT."""
        refiner_config = self.latent_refinement_config
        batch_size, frame_count = clip_latents.shape[:2]
        latent_height, latent_width = clip_latents.shape[-2:]
        num_train_timestep = refiner_config["num_train_timestep"]
        denoise_steps = refiner_config["denoise_steps"]

        timestep_value = int(torch.randint(
            refiner_config["noise_timestep_min"],
            refiner_config["noise_timestep_max"] + 1, (1,),
            generator=self.generator))
        conditions = {}
        for key, value in self.refiner_conditions.items():
            if value.ndim >= 3:
                expanded = value.expand(
                    batch_size, frame_count, *value.shape[2:])
            else:
                expanded = value.expand(batch_size)
            conditions[key] = expanded
        conditions["temporal_attention_mask"] = self._refiner_causal_mask(
            frame_count, latent_height, latent_width)

        latents = clip_latents
        for step_index in range(denoise_steps):
            step_timestep = timestep_value \
                * (denoise_steps - step_index) // denoise_steps
            sigma = step_timestep / num_train_timestep
            noise = torch.randn(
                latents.shape, generator=self.generator).to(self.device)
            noisy = (1.0 - sigma) * latents + sigma * noise
            timesteps = torch.full(
                latents.shape[:3], float(step_timestep),
                dtype=torch.float32, device=self.device)
            model_output, _, _ = self.refiner_model(
                noisy.to(torch.float16), timesteps, **conditions)
            velocity = model_output[0].float()
            latents = noisy - sigma * velocity
        return latents

    def _build_target_cameras(self):
        """Precompute c2w, K and canvas sampling grids for the target views."""
        target_height = self.training_config["target_height"]
        target_width = self.training_config["target_width"]
        target_fov_deg = self.training_config["target_fov_deg"]
        canvas_width = 2 * self.training_config["half_width"] \
            - PANO_CANVAS_OVERLAP_COLUMNS
        canvas_height = self.training_config["half_height"]
        self.canvas_pad_columns = self.training_config["canvas_pad_columns"]
        lat_half_rad = PANO_LAT_HALF_RAD

        # All views share one K; cos(pixel ray, optical axis) = 1/|(x, y, 1)|.
        focal = (target_width / 2.0) \
            / np.tan(np.deg2rad(target_fov_deg) / 2.0)
        x_cam = (np.arange(target_width) + 0.5 - target_width / 2.0) / focal
        y_cam = (np.arange(target_height) + 0.5 - target_height / 2.0) / focal
        ray_norm = np.sqrt(
            1.0 + x_cam[None, :] ** 2 + y_cam[:, None] ** 2)
        self.target_view_cosine = torch.from_numpy(
            (1.0 / ray_norm)[None]).float().to(self.device)

        c2w_list = []
        intrinsics_list = []
        grid_list = []
        for yaw_deg in self.training_config["target_yaws_deg"]:
            c2w, intrinsics = build_virtual_camera(
                yaw_deg, target_height, target_width, target_fov_deg, 0.0)
            directions = virtual_pixel_directions_ego(
                c2w, intrinsics, target_height, target_width)
            pano_x = -directions[..., 1]
            pano_y = directions[..., 0]
            pano_z = directions[..., 2]
            norm = np.sqrt(pano_x ** 2 + pano_y ** 2 + pano_z ** 2)
            azimuth = np.arctan2(pano_x, pano_y)
            elevation = np.arcsin(pano_z / norm)
            column = (azimuth + np.pi) / (2.0 * np.pi) * canvas_width
            # Unwrap views straddling the canvas seam into the padded region.
            if column.max() - column.min() > canvas_width / 2.0:
                column = np.where(
                    column < canvas_width / 2.0, column + canvas_width, column)
            row = (lat_half_rad - elevation) / (2.0 * lat_half_rad) \
                * canvas_height
            padded_width = canvas_width + self.canvas_pad_columns
            grid_x = column / padded_width * 2.0 - 1.0
            grid_y = row / canvas_height * 2.0 - 1.0
            grid = np.stack([grid_x, grid_y], axis=-1)
            c2w_list.append(c2w.astype(np.float32))
            intrinsics_list.append(intrinsics.astype(np.float32))
            grid_list.append(grid.astype(np.float32))
        self.target_c2w = torch.from_numpy(np.stack(c2w_list)).to(self.device)
        self.target_intrinsics = torch.from_numpy(
            np.stack(intrinsics_list)).to(self.device)
        self.target_grids = torch.from_numpy(
            np.stack(grid_list)).to(self.device)

    @staticmethod
    def _half_pano_axes(column_range, height, width, device, dtype):
        """Canonical OneBEV row/column coordinates for one resized half."""
        column_start, column_stop = column_range
        rows = (
            (torch.arange(height, device=device, dtype=dtype) + 0.5)
            * (PANO_H / height) - 0.5 + PANO_CROP_TOP
        )
        columns = column_start + (
            torch.arange(width, device=device, dtype=dtype) + 0.5
        ) * ((column_stop - column_start) / width) - 0.5
        return rows, columns

    @staticmethod
    def _batch_calibration_reference(batch, key):
        if key not in batch:
            raise KeyError(
                "{} is required by nuscenes6 stitch supervision; rebuild "
                "the batch with PanoNuScenesDataset".format(key))
        values = batch[key]
        if not torch.is_tensor(values) or values.ndim < 2:
            raise ValueError(
                "{} must be a batched [B,T,...] tensor".format(key))
        reference = values[0, 0]
        expanded = reference.reshape((1, 1) + reference.shape).expand_as(
            values)
        if values.is_floating_point():
            matches = torch.allclose(values, expanded, atol=1e-6, rtol=1e-6)
        else:
            matches = torch.equal(values, expanded)
        if not matches:
            raise ValueError(
                "nuscenes6 stitch currently requires static sensor "
                "calibration inside a batch; {} changes across B/T".format(
                    key))
        return reference

    @staticmethod
    def _calibration_signature(references):
        """Stable exact key for one of the finite nuScenes sensor rigs."""
        digest = hashlib.sha256()
        for key in sorted(references):
            value = references[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def _activate_calibrated_supervision(self, signature, state):
        for key, value in state.items():
            setattr(self, key, value)
        self._active_calibration_signature = signature
        self._calibrated_supervision_ready = True

    def _prepare_nuscenes6_supervision(self, batch):
        """Activate a cached rig or build exact stitch/ray maps for it."""
        if not self.uses_nuscenes6_stitch:
            return

        calibration_keys = (
            "pano_camera_transforms",
            "pano_camera_intrinsics",
            "pano_camera_image_size",
        )
        references = {
            key: self._batch_calibration_reference(batch, key).detach().cpu()
            for key in calibration_keys
        }
        signature = self._calibration_signature(references)
        if signature == self._active_calibration_signature:
            return
        cached = self._calibrated_supervision_cache.get(signature)
        if cached is not None:
            self._activate_calibrated_supervision(signature, cached)
            return
        max_cached_rigs = int(self.training_config.get(
            "max_calibration_cache_entries", 8))
        if max_cached_rigs <= 0:
            raise ValueError(
                "max_calibration_cache_entries must be positive")
        if len(self._calibrated_supervision_cache) >= max_cached_rigs:
            raise RuntimeError(
                "encountered more than {} distinct sensor rigs; increase "
                "max_calibration_cache_entries only after checking the "
                "dataset calibration".format(max_cached_rigs))
        camera_to_ego = references["pano_camera_transforms"].to(
            self.device, dtype=torch.float32)
        native_intrinsics = references["pano_camera_intrinsics"].to(
            self.device, dtype=torch.float32)
        image_sizes = references["pano_camera_image_size"]
        if not torch.equal(image_sizes, image_sizes[0:1].expand_as(image_sizes)):
            raise ValueError(
                "nuscenes6 stitch requires a common source resolution for "
                "all six cameras")
        native_width, native_height = [int(value) for value in image_sizes[0]]
        render_height = int(self.training_config.get(
            "source_camera_render_height", 360))
        render_width = int(self.training_config.get(
            "source_camera_render_width", 640))
        if min(render_height, render_width) <= 0:
            raise ValueError(
                "source camera render dimensions must be positive")
        render_intrinsics = prepare_gsplat_intrinsics(
            native_intrinsics,
            (native_height, native_width),
            (render_height, render_width),
        )
        projection_model = self.training_config.get(
            "stitch_projection_model", "onebev_legacy")
        seam_erosion = int(self.training_config.get(
            "stitch_seam_erosion_px", 0))

        half_height = self.training_config["half_height"]
        half_width = self.training_config["half_width"]
        canvas_rows, canvas_columns = half_canvas_pano_axes(
            half_size=(half_height, half_width),
            right_trim_px=PANO_CANVAS_OVERLAP_COLUMNS,
            device=self.device,
            dtype=torch.float32,
        )
        stitch_grid, stitch_owner, stitch_valid = \
            build_owned_pano_sampling_grid(
                camera_to_ego,
                render_intrinsics,
                source_size=(render_height, render_width),
                pano_rows=canvas_rows,
                pano_columns=canvas_columns,
                projection_model=projection_model,
                seam_erosion_px=seam_erosion,
            )

        def build_half_rays(height, width):
            origin_list = []
            direction_list = []
            owner_list = []
            valid_list = []
            z_cosine_list = []
            for column_range in (
                PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE
            ):
                rows, columns = self._half_pano_axes(
                    column_range, height, width,
                    self.device, torch.float32)
                grid, owner, valid = build_owned_pano_sampling_grid(
                    camera_to_ego,
                    render_intrinsics,
                    source_size=(render_height, render_width),
                    pano_rows=rows,
                    pano_columns=columns,
                    projection_model=projection_model,
                    seam_erosion_px=0,
                )
                origins, directions, z_cosine = sampling_grid_to_ego_rays(
                    grid,
                    owner,
                    camera_to_ego,
                    render_intrinsics,
                    source_size=(render_height, render_width),
                )
                # Invalid legacy-FOV cells should be exceptional.  Retain a
                # finite fallback ray so the decoder never receives NaNs; all
                # render supervision still carries the explicit valid mask.
                ideal = torch.from_numpy(pano_half_directions(
                    column_range, height, width)).to(
                        device=self.device, dtype=torch.float32)
                directions = torch.where(
                    valid[..., None], directions, ideal)
                origins = torch.where(
                    valid[..., None], origins, torch.zeros_like(origins))
                z_cosine = torch.where(
                    valid, z_cosine,
                    torch.ones_like(z_cosine))
                origin_list.append(origins)
                direction_list.append(directions)
                owner_list.append(owner[0])
                valid_list.append(valid)
                z_cosine_list.append(z_cosine)
            return (
                torch.stack(origin_list),
                torch.stack(direction_list),
                torch.stack(owner_list),
                torch.stack(valid_list),
                torch.stack(z_cosine_list),
            )

        pixel_geometry = build_half_rays(half_height, half_width)
        latent_geometry = build_half_rays(
            half_height // self.vae_downsample,
            half_width // self.vae_downsample)
        _, pixel_directions, pixel_owners, pixel_valid, pixel_z_cosine = \
            pixel_geometry
        _, latent_directions, latent_owners, latent_valid, _ = \
            latent_geometry
        half_range_correction = 1.0 / pixel_z_cosine.clamp_min(1e-4)
        state = {
            "pixel_directions": pixel_directions,
            "latent_directions": latent_directions.permute(0, 3, 1, 2),
            "pixel_camera_indices": pixel_owners,
            "latent_camera_indices": latent_owners,
            "pixel_content_ray_valid": pixel_valid,
            "latent_content_ray_valid": latent_valid,
            "half_range_correction": half_range_correction,
            "half_range_correction_cpu":
                half_range_correction.detach().cpu(),
            "nuscenes_camera_to_ego": camera_to_ego,
            "nuscenes_render_intrinsics": render_intrinsics,
            "nuscenes_render_height": render_height,
            "nuscenes_render_width": render_width,
            "stitch_sampling_grid": stitch_grid,
            "stitch_owner": stitch_owner,
            "stitch_valid": stitch_valid,
        }
        self._calibrated_supervision_cache[signature] = state
        self._activate_calibrated_supervision(signature, state)
        if self.should_save:
            print(
                "Prepared nuScenes6 rig {} with {} stitch at camera {}x{} "
                "-> canvas {}x{} (valid {:.4f}); exact content rays "
                "enabled ({} cached rig(s)).".format(
                    signature[:12],
                    projection_model,
                    render_width,
                    render_height,
                    len(canvas_columns),
                    len(canvas_rows),
                    float(stitch_valid.float().mean()),
                    len(self._calibrated_supervision_cache)),
                flush=True)

    def _unpadded_stitch_canvas(self, halves):
        """De-overlap two resized halves without the virtual-view wrap pad."""
        return torch.cat([
            halves[:, 0],
            halves[:, 1, :, :, PANO_CANVAS_OVERLAP_COLUMNS:],
        ], dim=-1)

    def _render_nuscenes6_stitched(self, gaussians, target_c2w=None):
        """Render six calibrated cameras, then apply the cached hard seam."""
        if not self._calibrated_supervision_ready:
            raise RuntimeError(
                "nuscenes6 supervision cache was not prepared from a batch")
        if target_c2w is None:
            target_c2w = self.nuscenes_camera_to_ego
        rgb, depth, alpha = self.render_targets(
            gaussians,
            target_c2w=target_c2w,
            target_intrinsics=self.nuscenes_render_intrinsics,
            width=self.nuscenes_render_width,
            height=self.nuscenes_render_height,
        )
        camera_channels = torch.cat([
            rgb,
            depth.unsqueeze(1),
            alpha.unsqueeze(1),
        ], dim=1)
        stitched, valid = stitch_camera_views_from_owned_grid(
            camera_channels,
            self.stitch_sampling_grid,
            self.stitch_owner,
            self.stitch_valid,
        )
        return (
            stitched[None, :3],
            stitched[None, 3],
            stitched[None, 4],
            valid,
        )

    def _stitch_canvas(self, halves):
        """(N, 2, C, H, W) halves -> (N, C, H, 2W-494+pad) wrapped canvas."""
        canvas = self._unpadded_stitch_canvas(halves)
        return torch.cat(
            [canvas, canvas[:, :, :, :self.canvas_pad_columns]], dim=-1)

    def _resample_targets(self, canvas, mode="bilinear"):
        """Sample every target view from every canvas: (N, K, C, Hv, Wv)."""
        sample_count = canvas.shape[0]
        view_count = self.target_grids.shape[0]
        canvas_repeated = canvas.repeat_interleave(view_count, dim=0)
        grids = self.target_grids.repeat(sample_count, 1, 1, 1)
        sampled = F.grid_sample(
            canvas_repeated, grids, mode=mode, padding_mode="border",
            align_corners=False)
        return sampled.unflatten(0, (sample_count, view_count))

    def _resample_masked_targets(self, half_values, half_valid):
        """Resample pano values without leaking the configured hole fill."""
        valid = half_valid.to(half_values.dtype)
        scalar_values = half_values.ndim == 4
        if scalar_values:
            half_values = half_values.unsqueeze(2)
        elif half_values.ndim != 5:
            raise ValueError(
                "half_values must be (N,V,H,W) or (N,V,C,H,W)")
        weighted_canvas = self._stitch_canvas(
            half_values * valid.unsqueeze(2))
        valid_canvas = self._stitch_canvas(valid.unsqueeze(2))
        weighted_targets = self._resample_targets(weighted_canvas)
        valid_weights = self._resample_targets(valid_canvas)[:, :, 0]
        targets = weighted_targets / valid_weights.unsqueeze(2).clamp_min(1e-6)
        min_fraction = self.training_config.get(
            "range_min_valid_fraction", 0.5)
        target_valid = valid_weights >= min_fraction
        hole_fill = self.training_config["range_hole_fill_m"]
        targets = torch.where(
            target_valid.unsqueeze(2), targets,
            torch.full_like(targets, hole_fill))
        if scalar_values:
            targets = targets[:, :, 0]
        return targets, target_valid

    def _target_view_depths(
        self, half_ranges, half_valid, pixel_origins=None,
        camera_origin_scale=1.0,
    ):
        """Convert stitched ray ranges to camera-z in the target rig."""
        if pixel_origins is None:
            target_ranges, target_valid = self._resample_masked_targets(
                half_ranges, half_valid)
            return (
                target_ranges * self.target_view_cosine.unsqueeze(0),
                target_valid,
            )
        directions = self.pixel_directions.unsqueeze(0).to(
            device=half_ranges.device, dtype=half_ranges.dtype)
        # Keep the target geometry on the same continuation path as decoder
        # means during camera-origin warmup. Previously the decoder received
        # scaled origins while own/cross depth targets always used full ones.
        points = pixel_origins.to(half_ranges.dtype) * camera_origin_scale \
            + directions * half_ranges.unsqueeze(-1)
        point_channels = points.permute(0, 1, 4, 2, 3)
        target_points, target_valid = self._resample_masked_targets(
            point_channels, half_valid)
        world_to_camera = torch.linalg.inv(self.target_c2w).to(
            device=target_points.device, dtype=target_points.dtype)
        target_depth = torch.einsum(
            "kc,nkchw->nkhw",
            world_to_camera[:, 2, :3], target_points,
        ) + world_to_camera[None, :, 2, 3, None, None]
        return target_depth, target_valid

    def _frame_ray_origins(self, batch, frame_indices):
        """Expand compact six-camera translations to latent/pixel ray maps."""
        if not self.decoder.use_camera_origins:
            return None, None
        if "pano_camera_translations" not in batch:
            raise KeyError(
                "The decoder uses camera origins, but the batch has no "
                "pano_camera_translations. Rebuild it with "
                "PanoNuScenesDataset.")
        camera_origins = batch["pano_camera_translations"]
        frame_indices = frame_indices.to(camera_origins.device)
        camera_origins = camera_origins[:, frame_indices].flatten(0, 1)
        camera_origins = camera_origins.to(
            self.device, dtype=self.pixel_directions.dtype)
        pixel_owners = self.pixel_camera_indices[:, None, :].expand(
            -1, self.pixel_directions.shape[1], -1)
        latent_owners = self.latent_camera_indices[:, None, :].expand(
            -1, self.latent_directions.shape[-2], -1)
        # Advanced indexing yields (N, V, H, W, 3).
        pixel_origins = camera_origins[:, pixel_owners]
        latent_origins = camera_origins[:, latent_owners]
        latent_origins = latent_origins.permute(0, 1, 4, 2, 3)
        return latent_origins, pixel_origins

    def _camera_origin_scale(self, global_step):
        """Ramp a legacy checkpoint into calibrated multi-origin geometry."""
        if not self.decoder.use_camera_origins:
            return 0.0
        warmup_steps = self.training_config.get(
            "camera_origin_warmup_steps", 0)
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, global_step / warmup_steps))

    def _load_frame_ranges(self, batch, frame_indices, return_valid=False):
        """Range priors and optional validity for the selected frames."""
        range_dir = self.training_config["range_dir"]
        hole_fill = self.training_config["range_hole_fill_m"]
        half_height = self.training_config["half_height"]
        half_width = self.training_config["half_width"]
        batch_size = batch["vae_images"].shape[0]
        half_list = []
        valid_half_list = []
        for sample_index in range(batch_size):
            for frame_index in frame_indices.tolist():
                timestamp = int(
                    batch["frame_timestamps_us"][sample_index, frame_index])
                with h5py.File(
                    os.path.join(range_dir, f"{timestamp}.h5"), "r"
                ) as handle:
                    depth_z = np.asarray(handle["depth"], dtype=np.float32)
                valid_full = torch.from_numpy(depth_z > 0.0).float()
                if self.half_range_correction is None:
                    full = torch.from_numpy(
                        (depth_z * self.range_correction).astype(np.float32))
                else:
                    # Resize metric source-camera z first, then convert with
                    # the exact content ray represented at the decoder pixel.
                    # Multiplying z by an approximate full-pano cosine before
                    # resize was one source of the old seam-depth mismatch.
                    full = torch.from_numpy(depth_z)
                for half_index, column_range in enumerate((
                    PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE
                )):
                    source_range = full[
                        None, None, :, column_range[0]:column_range[1]]
                    source_valid = valid_full[
                        None, None, :, column_range[0]:column_range[1]]
                    weighted = F.interpolate(
                        source_range * source_valid,
                        size=(half_height, half_width),
                        mode="bilinear", antialias=True)
                    valid_weight = F.interpolate(
                        source_valid, size=(half_height, half_width),
                        mode="bilinear", antialias=True)
                    half_valid = valid_weight >= self.training_config.get(
                        "range_min_valid_fraction", 0.5)
                    half = weighted / valid_weight.clamp_min(1e-6)
                    if self.half_range_correction is not None:
                        half = half * self.half_range_correction_cpu[
                            half_index][None, None]
                    half = torch.where(
                        half_valid, half, torch.full_like(half, hole_fill))
                    half_list.append(half[0, 0])
                    valid_half_list.append(half_valid[0, 0])
        stacked = torch.stack(half_list).to(self.device)
        ranges = stacked.unflatten(
            0, (batch_size * len(frame_indices), 2))
        if not return_valid:
            return ranges
        valid = torch.stack(valid_half_list).to(self.device).unflatten(
            0, (batch_size * len(frame_indices), 2))
        return ranges, valid

    def _ranges_to_source_depth_z(self, half_ranges):
        if self.half_range_correction is None:
            raise RuntimeError(
                "exact source-camera depth conversion is only available "
                "after nuscenes6 supervision preparation")
        correction = self.half_range_correction.to(
            device=half_ranges.device, dtype=half_ranges.dtype)
        return half_ranges / correction.unsqueeze(0)

    def _load_frame_segments(self, batch, frame_indices):
        """Binary dynamic masks (N, 2, half_H, half_W) for selected frames."""
        segment_dir = self.training_config["segment_dir"]
        dynamic_ids = torch.tensor(
            self.training_config["dynamic_class_ids"], dtype=torch.uint8)
        half_height = self.training_config["half_height"]
        half_width = self.training_config["half_width"]
        batch_size = batch["vae_images"].shape[0]
        half_list = []
        for sample_index in range(batch_size):
            for frame_index in frame_indices.tolist():
                timestamp = int(
                    batch["frame_timestamps_us"][sample_index, frame_index])
                segment_image = Image.open(
                    os.path.join(segment_dir, f"{timestamp}.png"))
                class_ids = torch.from_numpy(
                    np.array(segment_image, copy=True)[:, :, 0])
                dynamic = torch.isin(class_ids, dynamic_ids).float()
                dynamic = dynamic[None, None]
                for column_range in (
                    PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE
                ):
                    half = dynamic[:, :, :, column_range[0]:column_range[1]]
                    half = F.interpolate(
                        half, size=(half_height, half_width), mode="nearest")
                    half_list.append(half[0, 0])
        stacked = torch.stack(half_list).to(self.device)
        return stacked.unflatten(0, (batch_size * len(frame_indices), 2)).long()

    @torch.no_grad()
    def encode_half_latents(self, half_images):
        """VAE-encode (N, 2, 3, H, W) halves into (N, 2, C, h, w) latents."""
        sample_count = half_images.shape[0]
        flat_images = half_images.flatten(0, 1) * 2.0 - 1.0
        posterior = self.vae.encode(flat_images).latent_dist
        shift_factor = self.vae.config.shift_factor
        latents = (posterior.sample() - shift_factor) \
            * self.vae.config.scaling_factor
        return latents.unflatten(0, (sample_count, 2))

    @torch.no_grad()
    def decode_half_latents(self, latents):
        """VAE-decode (N,2,C,h,w) latents into [0,1] panorama halves."""
        sample_count = latents.shape[0]
        flat_latents = latents.flatten(0, 1).to(dtype=self.vae.dtype)
        shift_factor = self.vae.config.shift_factor
        if shift_factor is None:
            shift_factor = 0
        decode_batch = int(
            self.latent_domain_adaptation_config.get("vae_decode_batch", 2)
        )
        if decode_batch < 1:
            raise ValueError("vae_decode_batch must be positive")
        decoded = []
        for start in range(0, len(flat_latents), decode_batch):
            part = flat_latents[start:start + decode_batch]
            decoded.append(self.vae.decode(
                part / self.vae.config.scaling_factor + shift_factor,
                return_dict=False)[0])
        images = (torch.cat(decoded) / 2.0 + 0.5).clamp(0.0, 1.0)
        return images.unflatten(0, (sample_count, 2))

    def _maybe_load_generated_latents(self, batch, frame_indices):
        """Load one exact Stage-2 cache item according to the mix probability."""
        config = self.latent_domain_adaptation_config
        if config is None or not config.get("enabled", False):
            return None
        if batch["vae_images"].shape[0] != 1:
            raise RuntimeError(
                "generated-latent adaptation currently requires batch_size=1")
        dataset_index = scalar_int(batch["dataset_index"])
        anchor_timestamp_us = scalar_int(batch["anchor_timestamp_us"])
        path = latent_cache_path(
            config["cache_dir"], dataset_index, anchor_timestamp_us)
        if not path.is_file():
            if config.get("strict", True):
                raise FileNotFoundError(
                    f"Required generated-latent cache item is missing: {path}")
            return None
        draw = float(torch.rand(
            1, generator=self.latent_domain_generator).item())
        if draw >= float(config["probability"]):
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        selected_timestamps = batch["frame_timestamps_us"][0, frame_indices]
        latents = validate_cache_payload(
            payload,
            dataset_index=dataset_index,
            anchor_timestamp_us=anchor_timestamp_us,
            frame_indices=frame_indices,
            frame_timestamps_us=selected_timestamps,
        )
        return latents.to(
            self.device, dtype=self.latent_directions.dtype, non_blocking=True)

    def render_targets(
        self, gaussians, target_c2w=None, target_intrinsics=None,
        width=None, height=None
    ):
        """Rasterize (N_gaussians, 14) at the target (or given) cameras.

        `target_intrinsics` / `width` / `height` default to the supervised
        virtual rig, so every existing caller keeps its exact behavior. They
        are overridable so the same Gaussians can be rasterized at the real
        nuScenes six-camera calibration (each camera has its own K and the
        frames are 1600x900), which is what an evaluation against the
        original camera images requires.
        """
        import gsplat
        if target_c2w is None:
            target_c2w = self.target_c2w
        if target_intrinsics is None:
            target_intrinsics = self.target_intrinsics
        if width is None:
            width = self.training_config["target_width"]
        if height is None:
            height = self.training_config["target_height"]
        view_matrices = torch.linalg.inv(target_c2w)
        rendered, alphas, _ = gsplat.rasterization(
            means=gaussians[:, 0:3],
            quats=gaussians[:, 7:11],
            scales=gaussians[:, 11:14],
            opacities=gaussians[:, 6],
            colors=gaussians[:, 3:6],
            viewmats=view_matrices,
            Ks=target_intrinsics,
            width=width,
            height=height,
            near_plane=self.training_config["near_plane"],
            far_plane=self.training_config["far_plane"],
            render_mode="RGB+ED",
            eps2d=self.active_rasterize_eps2d,
            rasterize_mode=self.rasterize_mode)
        rendered_rgb = rendered[..., 0:3].permute(0, 3, 1, 2)
        rendered_depth = rendered[..., 3]
        return rendered_rgb, rendered_depth, alphas[..., 0]

    def _build_own_supervision_targets(
        self, frame_images, frame_ranges, frame_range_valid,
        frame_pixel_origins, camera_origin_scale,
    ):
        """Return pixel-aligned RGB/depth targets for the active protocol."""
        if self.uses_nuscenes6_stitch:
            target_images = self._unpadded_stitch_canvas(frame_images)
            source_depth_z = self._ranges_to_source_depth_z(frame_ranges)
            target_depth = self._unpadded_stitch_canvas(
                source_depth_z.unsqueeze(2))[:, 0]
            content_valid = self.pixel_content_ray_valid.unsqueeze(0)
            target_valid = self._unpadded_stitch_canvas(
                (frame_range_valid & content_valid).unsqueeze(2).float()
            )[:, 0] >= 0.5
            target_valid &= self.stitch_valid.unsqueeze(0)
            return target_images, target_depth, target_valid

        rgb_canvas = self._stitch_canvas(frame_images)
        target_images = self._resample_targets(rgb_canvas)[0]
        target_view_depth_all, target_range_valid_all = \
            self._target_view_depths(
                frame_ranges, frame_range_valid,
                frame_pixel_origins,
                camera_origin_scale=camera_origin_scale)
        return (
            target_images,
            target_view_depth_all[0],
            target_range_valid_all[0],
        )

    def _render_own_supervision(self, gaussians):
        if self.uses_nuscenes6_stitch:
            return self._render_nuscenes6_stitched(gaussians)
        rgb, depth, alpha = self.render_targets(gaussians)
        return rgb, depth, alpha, torch.ones_like(
            alpha, dtype=torch.bool)

    def _target_rgb_for_supervision(self, frame_images):
        if self.uses_nuscenes6_stitch:
            return self._unpadded_stitch_canvas(frame_images)
        return self._resample_targets(
            self._stitch_canvas(frame_images))[0]

    def _select_frame_indices(self, frame_count):
        return sample_frame_indices(
            frame_count,
            self.training_config["frames_per_step"],
            self.training_config.get("frame_sampling", "random"),
            self.generator,
        )

    def _target_cameras_in_source_ego(
        self, batch, batch_index, source_frame_index, target_frame_index
    ):
        """Express the target-time virtual cameras in source ego coordinates."""
        source_e2g = batch["ego_transforms"][
            batch_index, source_frame_index, 0].to(
                self.device, dtype=self.target_c2w.dtype)
        target_e2g = batch["ego_transforms"][
            batch_index, target_frame_index, 0].to(
                self.device, dtype=self.target_c2w.dtype)
        target_to_source = torch.linalg.inv(source_e2g) @ target_e2g
        return target_to_source.unsqueeze(0) @ self.target_c2w

    def _cross_time_rgb_loss(
        self, gaussians, source_segments, target_images,
        target_ranges, target_range_valid, target_segments, cross_c2w,
        target_pixel_origins=None, camera_origin_scale=1.0,
    ):
        """Occlusion-aware static cross-render photometric consistency."""
        source_static = 1.0 - source_segments.reshape(
            source_segments.shape[0], -1, 1).to(gaussians.dtype)
        static_gaussians = torch.cat([
            gaussians[..., :6],
            gaussians[..., 6:7] * source_static,
            gaussians[..., 7:],
        ], dim=-1)
        cross_rgb, cross_depth, cross_alpha = self.render_targets(
            static_gaussians[0], target_c2w=cross_c2w)

        target_segment_canvas = self._stitch_canvas(
            target_segments.unsqueeze(2).float())
        target_dynamic = self._resample_targets(
            target_segment_canvas, mode="nearest")[0, :, 0]
        target_view_depths, target_valid_views = self._target_view_depths(
            target_ranges, target_range_valid, target_pixel_origins,
            camera_origin_scale=camera_origin_scale)
        target_view_depth = target_view_depths[0]
        target_validity = cross_target_validity_mask(
            target_view_depth,
            target_dynamic < 0.5,
            target_valid_views[0],
            near_plane=self.training_config["near_plane"],
        )
        rendered_visibility = cross_render_visibility_mask(
            cross_depth.detach(),
            target_view_depth,
            cross_alpha.detach(),
            target_dynamic < 0.5,
            target_valid_views[0],
            min_alpha=self.training_config.get("cross_min_alpha", 0.05),
            absolute_tolerance_m=self.training_config.get(
                "cross_depth_abs_tolerance_m", 2.0),
            relative_tolerance=self.training_config.get(
                "cross_depth_rel_tolerance", 0.1),
            near_plane=self.training_config["near_plane"],
        )
        photometric_mask = select_cross_photometric_mask(
            target_validity,
            rendered_visibility,
            self.training_config.get(
                "cross_photometric_mask", "render_visibility"),
        )

        difference = cross_rgb - target_images
        epsilon = self.training_config.get(
            "cross_charbonnier_epsilon", 1e-3)
        charbonnier = torch.sqrt(
            difference.square() + epsilon ** 2).mean(dim=1)
        l1_map = difference.abs().mean(dim=1)
        raw_l1 = masked_mean(l1_map, photometric_mask)
        ssim_weight = self.training_config.get("cross_ssim_weight", 0.0)
        if ssim_weight > 0.0:
            structural = ssim_loss_map(cross_rgb, target_images)
            photometric = (1.0 - ssim_weight) * charbonnier \
                + ssim_weight * structural
        else:
            photometric = charbonnier
        loss = masked_mean(photometric, photometric_mask)

        min_alpha = self.training_config.get("cross_min_alpha", 0.05)
        target_alpha = self.training_config.get("cross_target_alpha", 0.5)
        coverage_loss = alpha_coverage_loss(
            cross_alpha, target_validity, target_alpha)
        target_support = target_validity.float().sum().clamp_min(1.0)
        visibility_recall = rendered_visibility.float().sum() / target_support
        alpha_coverage = (
            (cross_alpha.detach() >= min_alpha) & target_validity
        ).float().sum() / target_support
        return {
            "photometric": loss,
            "rgb_l1": raw_l1,
            "rgb_l1_target": masked_mean(l1_map, target_validity),
            "rgb_l1_visible": masked_mean(l1_map, rendered_visibility),
            "valid_ratio": rendered_visibility.float().mean(),
            "target_valid_ratio": target_validity.float().mean(),
            "visibility_recall": visibility_recall,
            "alpha_coverage": alpha_coverage,
            "alpha_missing": coverage_loss,
        }

    def train_step(self, batch: dict, global_step: int):
        step_start = time.time()
        self.decoder_wrapper.train()

        if self.rasterize_mode == "antialiased" \
                and self.aa_eps2d_ramp_steps > 0:
            progress = min(
                max(float(global_step), 0.0) / self.aa_eps2d_ramp_steps,
                1.0)
            self.active_rasterize_eps2d = self.aa_eps2d_start + progress * (
                self.aa_eps2d_end - self.aa_eps2d_start)
        elif self.rasterize_mode == "antialiased":
            self.active_rasterize_eps2d = self.aa_eps2d_end

        # The active rig controls decoder rays, depth conversion and target
        # rendering, so it must be selected before any frame tensor is built.
        self._prepare_nuscenes6_supervision(batch)
        vae_images = batch["vae_images"]
        frame_indices = self._select_frame_indices(vae_images.shape[1])
        half_images = vae_images[:, frame_indices].flatten(0, 1)
        half_images = half_images.to(self.device)
        range_loss_enabled = self.training_config.get(
            "weight_render_depth", 0.0) > 0.0 or \
            self.training_config.get("weight_range_l1", 0.0) > 0.0
        need_frame_ranges = self.decoder.use_range_prior or \
            range_loss_enabled
        if need_frame_ranges:
            gt_ranges, gt_range_valid = self._load_frame_ranges(
                batch, frame_indices, return_valid=True)
            gt_range_valid = gt_range_valid \
                & self.pixel_content_ray_valid.unsqueeze(0)
        else:
            # The RGB-only generated-latent AA pilot has no geometry loss and
            # no range prior.  Do not fabricate depth targets or require a
            # depth mirror it cannot consume.
            gt_ranges = None
            gt_range_valid = None
        segment_loss_enabled = self.training_config.get("weight_seg", 0.0) \
            > 0.0
        # The original G2 configuration names semantic masks, but its data
        # mirror has no such asset.  The AA experiment explicitly sets the
        # corresponding loss to zero rather than silently substituting an
        # unrelated condition map.  Cross-time losses stay disabled.
        gt_segments = self._load_frame_segments(
            batch, frame_indices) if segment_loss_enabled else None
        latent_origins, pixel_origins = self._frame_ray_origins(
            batch, frame_indices)
        sample_count = half_images.shape[0]
        selected_frame_count = len(frame_indices)
        adjacent_pair_sampling = self.training_config.get(
            "frame_sampling", "random") == "adjacent_pairs"
        cross_weight = self.training_config.get("weight_cross_rgb_l1", 0.0)
        coverage_weight = self.training_config.get(
            "weight_cross_alpha_coverage", 0.0)
        if (cross_weight > 0.0 or coverage_weight > 0.0) \
                and not adjacent_pair_sampling:
            raise ValueError(
                "cross-render losses require frame_sampling=adjacent_pairs")
        pair_count = sample_count // 2 if adjacent_pair_sampling else 0
        camera_origin_scale = self._camera_origin_scale(global_step)

        generated_latents = self._maybe_load_generated_latents(
            batch, frame_indices)
        uses_generated_latents = generated_latents is not None
        geometry_on_generated = False if not uses_generated_latents else \
            self.latent_domain_adaptation_config.get(
                "supervise_geometry", False)
        supervise_geometry = not uses_generated_latents \
            or geometry_on_generated

        refined_latents = None
        if self.refiner_model is not None:
            draw = float(torch.rand(1, generator=self.generator))
            if draw < self.latent_refinement_config["probability"]:
                frame_latent_list = []
                for clip_frame in range(vae_images.shape[1]):
                    frame_halves = vae_images[:, clip_frame].to(self.device)
                    frame_latent_list.append(
                        self.encode_half_latents(frame_halves))
                clip_latents = torch.stack(frame_latent_list, dim=1)
                refined_clip = self.refine_clip_latents(clip_latents)
                refined_latents = refined_clip[:, frame_indices].flatten(0, 1)

        report = {
            "loss": 0.0, "rgb_l1": 0.0, "perceptual": 0.0,
            "render_depth_l1": 0.0, "range_l1": 0.0, "segment_ce": 0.0,
            "depth_valid_ratio": 0.0 if gt_range_valid is None else \
                gt_range_valid.float().mean().item(),
            "camera_origin_scale": camera_origin_scale,
            "rasterize_eps2d": self.active_rasterize_eps2d,
            "cross_photometric": 0.0, "cross_rgb_l1": 0.0,
            "cross_rgb_l1_target": 0.0, "cross_rgb_l1_visible": 0.0,
            "cross_valid_ratio": 0.0, "cross_target_valid_ratio": 0.0,
            "cross_visibility_recall": 0.0, "cross_alpha_coverage": 0.0,
            "cross_alpha_missing": 0.0, "temporal_context_ratio": 0.0,
            "generated_latent_ratio": float(uses_generated_latents),
            "geometry_supervision_ratio": float(supervise_geometry),
        }
        previous_latents = {}
        # One frame per forward/backward keeps activation memory flat in
        # frames_per_step; gradients accumulate into a single optimizer step.
        for sample_index in range(sample_count):
            frame_images = half_images[sample_index:sample_index + 1]
            frame_gt_ranges = None if gt_ranges is None else \
                gt_ranges[sample_index:sample_index + 1]
            frame_gt_range_valid = None if gt_range_valid is None else \
                gt_range_valid[sample_index:sample_index + 1]
            frame_gt_segments = None if gt_segments is None else \
                gt_segments[sample_index:sample_index + 1]
            frame_latent_origins = None if latent_origins is None else \
                latent_origins[sample_index:sample_index + 1]
            frame_pixel_origins = None if pixel_origins is None else \
                pixel_origins[sample_index:sample_index + 1]
            if generated_latents is not None:
                latents = generated_latents[
                    sample_index:sample_index + 1]
            elif refined_latents is None:
                latents = self.encode_half_latents(frame_images)
            else:
                latents = refined_latents[sample_index:sample_index + 1]

            batch_index = sample_index // selected_frame_count
            local_frame_index = sample_index % selected_frame_count
            is_pair_source = adjacent_pair_sampling \
                and local_frame_index % 2 == 0
            prior_latents = previous_latents.get(batch_index)
            if is_pair_source:
                prior_latents = None
                # Context is an input, never a route for gradients across
                # frames. Detach explicitly so this remains true if latent
                # production becomes trainable in a future experiment.
                previous_latents[batch_index] = latents.detach()
            report["temporal_context_ratio"] += \
                float(prior_latents is not None) / sample_count

            is_last_frame = sample_index == sample_count - 1
            if isinstance(
                self.decoder_wrapper,
                torch.nn.parallel.DistributedDataParallel
            ) and not is_last_frame:
                sync_context = self.decoder_wrapper.no_sync()
            else:
                sync_context = contextlib.nullcontext()

            if self.decoder.use_range_prior:
                if frame_gt_ranges is None:
                    raise RuntimeError(
                        "range prior requires loaded range targets")
                decoder_prior = frame_gt_ranges
            else:
                decoder_prior = None

            with sync_context:
                gaussians, predicted_range, predicted_segments = \
                    self.decoder_wrapper(
                        latents,
                        self.latent_directions.unsqueeze(0),
                        self.pixel_directions.unsqueeze(0),
                        decoder_prior,
                        previous_latents=prior_latents,
                        temporal_context_mask=0.0 if prior_latents is None
                        else 1.0,
                        latent_origins=frame_latent_origins,
                        pixel_origins=frame_pixel_origins,
                        camera_origin_scale=camera_origin_scale)

                with torch.no_grad():
                    appearance_images = frame_images
                    if uses_generated_latents:
                        # A generated future is one valid scene hypothesis, not
                        # necessarily the paired GT pixels.  The VAE decoding of
                        # this exact latent is therefore the aligned appearance
                        # target for the generated branch.
                        appearance_images = self.decode_half_latents(latents)
                    if supervise_geometry:
                        target_images, target_view_depth, target_range_valid = \
                            self._build_own_supervision_targets(
                                appearance_images,
                                frame_gt_ranges,
                                frame_gt_range_valid,
                                frame_pixel_origins,
                                camera_origin_scale)
                    else:
                        target_images = self._target_rgb_for_supervision(
                            appearance_images)
                        target_view_depth = None
                        target_range_valid = None

                rendered_rgb, rendered_depth, _, render_valid = \
                    self._render_own_supervision(gaussians[0])
                rgb_loss = masked_mean(
                    (rendered_rgb - target_images).abs(), render_valid)
                if self.perceptual_metric is None:
                    perceptual_loss = torch.zeros((), device=self.device)
                else:
                    resolution = self.training_config[
                        "perceptual_resolution"]
                    # LPIPS has no external spatial mask.  Make invalid
                    # stitch cells exactly equal to GT before resizing so
                    # they cannot reward black/transparent predictions.
                    perceptual_rendered = torch.where(
                        render_valid.unsqueeze(1),
                        rendered_rgb,
                        target_images)
                    # Clamp AFTER resizing: antialias lobes overshoot [0, 1].
                    rendered_resized = F.interpolate(
                        perceptual_rendered, size=resolution,
                        mode="bilinear", antialias=True).clamp(0.0, 1.0)
                    target_resized = F.interpolate(
                        target_images, size=resolution,
                        mode="bilinear", antialias=True).clamp(0.0, 1.0)
                    perceptual_loss = self.perceptual_metric(
                        rendered_resized, target_resized)
                    self.perceptual_metric.reset()
                if supervise_geometry:
                    if frame_gt_ranges is None or \
                            frame_gt_range_valid is None:
                        raise RuntimeError(
                            "geometry supervision requires range targets")
                    render_depth_loss = masked_mean(
                        (rendered_depth - target_view_depth).abs(),
                        target_range_valid & render_valid)
                    range_loss = masked_mean(
                        (predicted_range - frame_gt_ranges).abs(),
                        frame_gt_range_valid)
                    if segment_loss_enabled:
                        segment_loss_map = F.cross_entropy(
                            predicted_segments.reshape(-1, 2),
                            frame_gt_segments.reshape(-1), reduction="none")
                        segment_loss_map = segment_loss_map.reshape_as(
                            frame_gt_segments)
                        segment_loss = masked_mean(
                            segment_loss_map,
                            self.pixel_content_ray_valid.unsqueeze(0))
                    else:
                        segment_loss = predicted_segments.sum() * 0.0
                else:
                    render_depth_loss = rendered_depth.sum() * 0.0
                    range_loss = predicted_range.sum() * 0.0
                    segment_loss = predicted_segments.sum() * 0.0
                loss = self.training_config["weight_rgb_l1"] * rgb_loss \
                    + self.training_config["weight_perceptual"] \
                    * perceptual_loss \
                    + self.training_config["weight_render_depth"] \
                    * render_depth_loss \
                    + self.training_config["weight_range_l1"] \
                    * range_loss \
                    + self.training_config["weight_seg"] * segment_loss
                scaled_loss = loss / sample_count

                cross_result = None
                if adjacent_pair_sampling and (
                    cross_weight > 0.0 or coverage_weight > 0.0
                ):
                    # Train both t->t+1 and t+1->t.  Each direction receives
                    # gradients only through its source frame, so activation
                    # memory stays flat while neither temporal direction is
                    # privileged.
                    target_local_index = local_frame_index + (
                        1 if is_pair_source else -1)
                    target_sample_index = sample_index + (
                        1 if is_pair_source else -1)
                    target_images_for_cross = half_images[
                        target_sample_index:target_sample_index + 1]
                    with torch.no_grad():
                        target_images_for_cross = self._resample_targets(
                            self._stitch_canvas(
                                target_images_for_cross))[0]
                    source_frame_index = int(frame_indices[local_frame_index])
                    target_frame_index = int(frame_indices[target_local_index])
                    cross_c2w = self._target_cameras_in_source_ego(
                        batch, batch_index, source_frame_index,
                        target_frame_index)
                    cross_result = self._cross_time_rgb_loss(
                            gaussians, frame_gt_segments,
                            target_images_for_cross,
                            gt_ranges[
                                target_sample_index:target_sample_index + 1],
                            gt_range_valid[
                                target_sample_index:target_sample_index + 1],
                            gt_segments[
                                target_sample_index:target_sample_index + 1],
                            cross_c2w,
                            None if pixel_origins is None else pixel_origins[
                                target_sample_index:target_sample_index + 1],
                            camera_origin_scale=camera_origin_scale)
                    scaled_loss = scaled_loss \
                        + (cross_weight * cross_result["photometric"]
                           + coverage_weight
                           * cross_result["alpha_missing"]) \
                        / (2 * pair_count)
                scaled_loss.backward()

            report["loss"] += loss.item() / sample_count
            report["rgb_l1"] += rgb_loss.item() / sample_count
            report["perceptual"] += perceptual_loss.item() / sample_count
            report["render_depth_l1"] += \
                render_depth_loss.item() / sample_count
            report["range_l1"] += range_loss.item() / sample_count
            report["segment_ce"] += segment_loss.item() / sample_count
            if cross_result is not None:
                report_keys = (
                    ("cross_photometric", "photometric"),
                    ("cross_rgb_l1", "rgb_l1"),
                    ("cross_rgb_l1_target", "rgb_l1_target"),
                    ("cross_rgb_l1_visible", "rgb_l1_visible"),
                    ("cross_valid_ratio", "valid_ratio"),
                    ("cross_target_valid_ratio", "target_valid_ratio"),
                    ("cross_visibility_recall", "visibility_recall"),
                    ("cross_alpha_coverage", "alpha_coverage"),
                    ("cross_alpha_missing", "alpha_missing"),
                )
                for report_key, result_key in report_keys:
                    report[report_key] += cross_result[result_key].item() \
                        / (2 * pair_count)
                report["loss"] += (
                    cross_weight * cross_result["photometric"].item()
                    + coverage_weight * cross_result["alpha_missing"].item()
                ) / (2 * pair_count)

        if "max_norm_for_grad_clip" in self.training_config:
            torch.nn.utils.clip_grad_norm_(
                self.decoder_wrapper.parameters(),
                self.training_config["max_norm_for_grad_clip"])
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.loss_report_list.append(report)
        self.step_duration += time.time() - step_start

    def log(self, global_step: int, log_steps: int):
        means = {
            key: float(np.mean([row[key] for row in self.loss_report_list]))
            for key in self.loss_report_list[0]
        }
        seconds_per_step = self.step_duration / len(self.loss_report_list)
        self.loss_report_list = []
        self.step_duration = 0.0
        if self.should_save:
            summary = " ".join(
                f"{key}={value:.4f}" for key, value in means.items())
            print(f"step {global_step} {summary} "
                  f"{seconds_per_step:.2f}s/step", flush=True)
            if wandb.run is not None:
                wandb.log(means, step=global_step)

    @torch.no_grad()
    def preview_pipeline(self, batch: dict, output_path: str,
                         global_step: int):
        self.decoder_wrapper.eval()
        self._prepare_nuscenes6_supervision(batch)
        frame_count = min(
            self.training_config["preview_frame_count"],
            batch["vae_images"].shape[1])
        frame_grids = []
        previous_latents = None
        preview_use_temporal = self.training_config.get(
            "preview_use_temporal_context", True)
        for frame_index in range(frame_count):
            frame_selector = torch.tensor([frame_index], dtype=torch.long)
            half_images = batch["vae_images"][:1, frame_selector]
            half_images = half_images.flatten(0, 1).to(self.device)
            if self.decoder.use_range_prior:
                decoder_prior = self._load_frame_ranges(
                    {key: batch[key][:1] for key in
                     ("vae_images", "frame_timestamps_us")},
                    frame_selector)
            else:
                decoder_prior = None
            origin_batch = batch
            if "pano_camera_translations" in batch:
                origin_batch = {
                    "pano_camera_translations":
                        batch["pano_camera_translations"][:1]
                }
            frame_latent_origins, frame_pixel_origins = \
                self._frame_ray_origins(origin_batch, frame_selector)

            latents = self.encode_half_latents(half_images)
            gaussians, _, _ = self.decoder_wrapper(
                latents, self.latent_directions.unsqueeze(0),
                self.pixel_directions.unsqueeze(0), decoder_prior,
                previous_latents=(
                    previous_latents if preview_use_temporal else None),
                temporal_context_mask=(
                    1.0 if preview_use_temporal
                    and previous_latents is not None else 0.0),
                latent_origins=frame_latent_origins,
                pixel_origins=frame_pixel_origins,
                camera_origin_scale=self._camera_origin_scale(global_step))
            previous_latents = (
                latents.detach() if preview_use_temporal else None)
            rendered_rgb, _, _, _ = \
                self._render_own_supervision(gaussians[0])
            target_images = self._target_rgb_for_supervision(half_images)
            display_target = self.mask_supervision_for_display(target_images)
            display_render = self.mask_supervision_for_display(rendered_rgb)
            truth_row = torch.cat(list(display_target), dim=-1)
            rendered_row = torch.cat(list(display_render), dim=-1)
            grid = torch.cat([truth_row, rendered_row], dim=-2)
            frame_grids.append(grid.clamp(0.0, 1.0).cpu())

        if self.should_save:
            preview_dir = os.path.join(output_path, "preview")
            os.makedirs(preview_dir, exist_ok=True)
            dwm.utils.preview.save_tensor_to_video(
                os.path.join(preview_dir, f"{global_step}.mp4"),
                "libx264", float(batch["fps"][0]), frame_grids)

    def save_checkpoint(self, output_path: str, steps: int,
                        keep_last_n: int = 0):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.should_save:
            checkpoint_dir = os.path.join(output_path, "checkpoints")
            optimizer_dir = os.path.join(output_path, "optimizer")
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(optimizer_dir, exist_ok=True)
            torch.save(
                self.decoder.state_dict(),
                os.path.join(checkpoint_dir, f"{steps}.pth"))
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(optimizer_dir, f"{steps}.pth"))
            if keep_last_n > 0:
                self.prune_checkpoints(checkpoint_dir, keep_last_n)
                self.prune_checkpoints(optimizer_dir, keep_last_n)

    @staticmethod
    def prune_checkpoints(directory: str, keep_last_n: int):
        step_numbers = []
        for file_name in os.listdir(directory):
            if file_name.endswith(".pth"):
                step_numbers.append(int(file_name[:-len(".pth")]))
        for step_number in sorted(step_numbers)[:-keep_last_n]:
            os.remove(os.path.join(directory, f"{step_number}.pth"))
