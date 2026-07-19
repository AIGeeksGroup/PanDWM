"""GS decoder training pipeline downstream of the stage-2 CD world model.

Consumes the SAME video batches as the CD stage (`PanoNuScenesDataset` via
`vae_images` of shape (B, T, V=2, 3, H, W)); each step trains the decoder on
`frames_per_step` sampled frames. The frozen SD3.5 VAE encodes halves into
the latent space the stage-2 model generates, so generated latents plug in
at inference. Range priors come from the precomputed pano depth h5 files
looked up by frame timestamp. Supervision renders the predicted Gaussians
with gsplat at virtual perspective cameras whose ground truth is resampled
on GPU from the 4114-wide stitched canvas (cat(left, right[:, 494:])).
"""
import contextlib
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
from dwm_pano.datasets.pano_projection import (
    PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE, PANO_H, PANO_W,
)
from dwm_pano.gs_geometry import (
    PANO_LAT_HALF_RAD, build_virtual_camera, pano_half_directions,
    pano_source_camera_cosine_map, virtual_pixel_directions_ego,
)

# Resized-half overlap in columns: round(1152 / 5376 * 2304) = 494.
PANO_CANVAS_OVERLAP_COLUMNS = 494


class PanoGaussianDecoderPipeline:

    def __init__(
        self,
        output_path,
        config: dict,
        device,
        training_config: dict,
        pretrained_model_name_or_path: str,
        decoder: dict,
        latent_refinement: dict = None,
        resume_from=None,
    ):
        self.should_save = not torch.distributed.is_initialized() or \
            torch.distributed.get_rank() == 0
        self.config = config
        self.device = device
        self.training_config = training_config
        self.latent_refinement_config = latent_refinement
        self.generator = torch.Generator()
        self.generator.manual_seed(config["generator_seed"])

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

        if torch.distributed.is_initialized():
            self.decoder_wrapper = torch.nn.parallel.DistributedDataParallel(
                self.decoder, device_ids=[device.index])
        else:
            self.decoder_wrapper = self.decoder

        half_height = training_config["half_height"]
        half_width = training_config["half_width"]
        downsample = 2 ** (len(self.vae.config.down_block_types) - 1)
        pixel_direction_list = []
        latent_direction_list = []
        for column_range in (PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE):
            pixel_direction_list.append(pano_half_directions(
                column_range, half_height, half_width))
            latent_direction_list.append(pano_half_directions(
                column_range, half_height // downsample,
                half_width // downsample))
        self.pixel_directions = torch.from_numpy(
            np.stack(pixel_direction_list)).float().to(device)
        latent_directions = torch.from_numpy(
            np.stack(latent_direction_list)).float().to(device)
        self.latent_directions = latent_directions.permute(0, 3, 1, 2)

        self._build_target_cameras()
        # Stitched pano depth stores per-cell source-camera z; divide by
        # cos(ray, camera axis) to get Euclidean range along the pano ray.
        if training_config["range_is_source_camera_z"]:
            self.range_correction = 1.0 / pano_source_camera_cosine_map()
        else:
            self.range_correction = np.ones((PANO_H, PANO_W))

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

    def _stitch_canvas(self, halves):
        """(N, 2, C, H, W) halves -> (N, C, H, 2W-494+pad) wrapped canvas."""
        canvas = torch.cat(
            [halves[:, 0], halves[:, 1, :, :, PANO_CANVAS_OVERLAP_COLUMNS:]],
            dim=-1)
        return torch.cat(
            [canvas, canvas[:, :, :, :self.canvas_pad_columns]], dim=-1)

    def _resample_targets(self, canvas):
        """Sample every target view from every canvas: (N, K, C, Hv, Wv)."""
        sample_count = canvas.shape[0]
        view_count = self.target_grids.shape[0]
        canvas_repeated = canvas.repeat_interleave(view_count, dim=0)
        grids = self.target_grids.repeat(sample_count, 1, 1, 1)
        sampled = F.grid_sample(
            canvas_repeated, grids, mode="bilinear", padding_mode="border",
            align_corners=False)
        return sampled.unflatten(0, (sample_count, view_count))

    def _load_frame_ranges(self, batch, frame_indices):
        """Range priors (N, 2, half_H, half_W) for the selected frames."""
        range_dir = self.training_config["range_dir"]
        hole_fill = self.training_config["range_hole_fill_m"]
        half_height = self.training_config["half_height"]
        half_width = self.training_config["half_width"]
        batch_size = batch["vae_images"].shape[0]
        half_list = []
        for sample_index in range(batch_size):
            for frame_index in frame_indices.tolist():
                timestamp = int(
                    batch["frame_timestamps_us"][sample_index, frame_index])
                with h5py.File(
                    os.path.join(range_dir, f"{timestamp}.h5"), "r"
                ) as handle:
                    depth_z = np.asarray(handle["depth"], dtype=np.float32)
                depth_tensor = torch.from_numpy(depth_z)
                full = torch.from_numpy(
                    (depth_z * self.range_correction).astype(np.float32))
                full = torch.where(
                    depth_tensor > 0.0, full,
                    torch.full_like(full, hole_fill))
                full = full[None, None]
                for column_range in (
                    PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE
                ):
                    half = full[:, :, :, column_range[0]:column_range[1]]
                    half = F.interpolate(
                        half, size=(half_height, half_width),
                        mode="bilinear", antialias=True)
                    half_list.append(half[0, 0])
        stacked = torch.stack(half_list).to(self.device)
        return stacked.unflatten(0, (batch_size * len(frame_indices), 2))

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
                    np.asarray(segment_image)[:, :, 0])
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

    def render_targets(self, gaussians, target_c2w=None):
        """Rasterize (N_gaussians, 14) at the target (or given) cameras."""
        import gsplat
        if target_c2w is None:
            target_c2w = self.target_c2w
        view_matrices = torch.linalg.inv(target_c2w)
        rendered, alphas, _ = gsplat.rasterization(
            means=gaussians[:, 0:3],
            quats=gaussians[:, 7:11],
            scales=gaussians[:, 11:14],
            opacities=gaussians[:, 6],
            colors=gaussians[:, 3:6],
            viewmats=view_matrices,
            Ks=self.target_intrinsics,
            width=self.training_config["target_width"],
            height=self.training_config["target_height"],
            near_plane=self.training_config["near_plane"],
            far_plane=self.training_config["far_plane"],
            render_mode="RGB+ED")
        rendered_rgb = rendered[..., 0:3].permute(0, 3, 1, 2)
        rendered_depth = rendered[..., 3]
        return rendered_rgb, rendered_depth, alphas[..., 0]

    def _select_frame_indices(self, frame_count):
        frames_per_step = self.training_config["frames_per_step"]
        permutation = torch.randperm(frame_count, generator=self.generator)
        return permutation[:frames_per_step]

    def train_step(self, batch: dict, global_step: int):
        step_start = time.time()
        self.decoder_wrapper.train()

        vae_images = batch["vae_images"]
        frame_indices = self._select_frame_indices(vae_images.shape[1])
        half_images = vae_images[:, frame_indices].flatten(0, 1)
        half_images = half_images.to(self.device)
        gt_ranges = self._load_frame_ranges(batch, frame_indices)
        gt_segments = self._load_frame_segments(batch, frame_indices)
        sample_count = half_images.shape[0]

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
        }
        # One frame per forward/backward keeps activation memory flat in
        # frames_per_step; gradients accumulate into a single optimizer step.
        for sample_index in range(sample_count):
            frame_images = half_images[sample_index:sample_index + 1]
            frame_gt_ranges = gt_ranges[sample_index:sample_index + 1]
            frame_gt_segments = gt_segments[sample_index:sample_index + 1]
            if refined_latents is None:
                latents = self.encode_half_latents(frame_images)
            else:
                latents = refined_latents[sample_index:sample_index + 1]

            is_last_frame = sample_index == sample_count - 1
            if isinstance(
                self.decoder_wrapper,
                torch.nn.parallel.DistributedDataParallel
            ) and not is_last_frame:
                sync_context = self.decoder_wrapper.no_sync()
            else:
                sync_context = contextlib.nullcontext()

            if self.decoder.use_range_prior:
                decoder_prior = frame_gt_ranges
            else:
                decoder_prior = None

            with sync_context:
                gaussians, predicted_range, predicted_segments = \
                    self.decoder_wrapper(
                        latents,
                        self.latent_directions.unsqueeze(0),
                        self.pixel_directions.unsqueeze(0),
                        decoder_prior)

                with torch.no_grad():
                    rgb_canvas = self._stitch_canvas(frame_images)
                    range_canvas = self._stitch_canvas(
                        frame_gt_ranges.unsqueeze(2))
                    target_images = self._resample_targets(rgb_canvas)[0]
                    target_ranges = self._resample_targets(
                        range_canvas)[0, :, 0]

                rendered_rgb, rendered_depth, _ = self.render_targets(
                    gaussians[0])
                rgb_loss = (rendered_rgb - target_images).abs().mean()
                if self.perceptual_metric is None:
                    perceptual_loss = torch.zeros((), device=self.device)
                else:
                    resolution = self.training_config[
                        "perceptual_resolution"]
                    # Clamp AFTER resizing: antialias lobes overshoot [0, 1].
                    rendered_resized = F.interpolate(
                        rendered_rgb, size=resolution,
                        mode="bilinear", antialias=True).clamp(0.0, 1.0)
                    target_resized = F.interpolate(
                        target_images, size=resolution,
                        mode="bilinear", antialias=True).clamp(0.0, 1.0)
                    perceptual_loss = self.perceptual_metric(
                        rendered_resized, target_resized)
                    self.perceptual_metric.reset()
                # gsplat depth is camera-space z; convert ray range via cos.
                target_view_depth = target_ranges * self.target_view_cosine
                render_depth_loss = (
                    rendered_depth - target_view_depth).abs().mean()
                range_loss = (
                    predicted_range - frame_gt_ranges).abs().mean()
                segment_loss = F.cross_entropy(
                    predicted_segments.reshape(-1, 2),
                    frame_gt_segments.reshape(-1))
                loss = self.training_config["weight_rgb_l1"] * rgb_loss \
                    + self.training_config["weight_perceptual"] \
                    * perceptual_loss \
                    + self.training_config["weight_render_depth"] \
                    * render_depth_loss \
                    + self.training_config["weight_range_l1"] \
                    * range_loss \
                    + self.training_config["weight_seg"] * segment_loss
                (loss / sample_count).backward()

            report["loss"] += loss.item() / sample_count
            report["rgb_l1"] += rgb_loss.item() / sample_count
            report["perceptual"] += perceptual_loss.item() / sample_count
            report["render_depth_l1"] += \
                render_depth_loss.item() / sample_count
            report["range_l1"] += range_loss.item() / sample_count
            report["segment_ce"] += segment_loss.item() / sample_count

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
        frame_count = min(
            self.training_config["preview_frame_count"],
            batch["vae_images"].shape[1])
        frame_grids = []
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

            latents = self.encode_half_latents(half_images)
            gaussians, _, _ = self.decoder_wrapper(
                latents, self.latent_directions.unsqueeze(0),
                self.pixel_directions.unsqueeze(0), decoder_prior)
            rendered_rgb, _, _ = self.render_targets(gaussians[0])
            target_images = self._resample_targets(
                self._stitch_canvas(half_images))[0]
            truth_row = torch.cat(list(target_images), dim=-1)
            rendered_row = torch.cat(list(rendered_rgb), dim=-1)
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
