"""WorldSplat-style 4D training for the PanDWM Gaussian decoder.

Subclasses ``dwm_pano.pipelines.ctsd_pano_gs_decoder.PanoGaussianDecoderPipeline``
and overrides only ``train_step``; rendering, depth loading, LPIPS, previews
and checkpointing are inherited unchanged.

Per step, F frames of one clip (stride ``frame_stride``) are decoded, each in
its own ego frame, with gradients kept for all of them.  Gaussians whose
centre lies inside a padded 3D box of their frame are *dynamic*.  Losses:

* ``own``       the original single-frame reconstruction (RGB L1, LPIPS and,
                when a depth mirror exists, render-depth / range L1);
* ``composite`` WorldSplat: static Gaussians of all F frames moved into frame
                t's ego frame + frame t's own dynamic Gaussians, rendered at
                frame t's cameras and compared with frame t (Wang et al.,
                WorldSplat, ``render_with_static_dynamic``);
* ``loo``       leave-one-out: only the *other* frames' static Gaussians
                rendered at frame t, on static, covered pixels -- a genuine
                novel-view signal that a single-frame decoder never receives;
* ``motion``    (off by default) dynamic Gaussians of frame s carried to frame
                t by the rigid motion of the same tracked box, supervised on
                frame t's dynamic pixels.

Batch size must be 1 (as in every G2 config).  Batches need the box fields of
``dwm_pano_4d.dataset.PanoNuScenesBoxDataset``.
"""

from __future__ import annotations

import time

import torch
import torch.utils.checkpoint
import torch.nn.functional as F

from dwm_pano.pipelines.ctsd_pano_gs_decoder import (
    PanoGaussianDecoderPipeline,
    masked_mean,
)
from dwm_pano_4d.geometry import (
    box_pose,
    points_in_boxes,
    relative_ego_transform,
    transform_gaussians,
)

DEFAULT_4D = {
    "frame_stride": [2, 3, 4],
    "composite_targets": 2,
    "weight_own": 1.0,
    "weight_composite": 1.0,
    "weight_composite_perceptual": 0.1,
    "weight_loo": 0.5,
    "weight_motion": 0.0,
    "box_margin_m": 0.3,
    "loo_min_alpha": 0.5,
    "dynamic_pixel_alpha": 0.3,
    # Keep only each frame's (N, 14) output between forward and backward and
    # recompute decoder activations in backward: F frames must stay in one
    # graph for the pooled losses, which otherwise exceeds a 48 GB GPU.
    "checkpoint_decoder": True,
    # Composite every training render over a random solid background colour.
    # With the default black background a transparent Gaussian matches a dark
    # pixel for free, and at 1000 steps night clips collapsed to ~40 %
    # coverage.  Evaluation and demos still render on black.
    "random_background": False,
    # Probability of the random background per step (1.0 = always); the
    # rest of the steps keep the black background that evaluation and demos
    # render on, so colours stay calibrated for black.
    "random_background_prob": 1.0,
    # Coverage regulariser on each frame's own render: mean(1 - alpha) over
    # valid pixels.  With the black background a transparent Gaussian matches
    # a dark pixel for free; this term makes transparency cost something
    # while keeping the background (and so the colour calibration) black.
    "weight_coverage": 0.0,
}


class PanoGaussianDecoder4DPipeline(PanoGaussianDecoderPipeline):
    def __init__(self, *args, gs4d: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gs4d = dict(DEFAULT_4D)
        self.gs4d.update(gs4d or {})
        unknown = set(self.gs4d) - set(DEFAULT_4D)
        if unknown:
            raise ValueError(f"unknown gs4d options: {sorted(unknown)}")
        strides = self.gs4d["frame_stride"]
        self.gs4d["frame_stride"] = [strides] if isinstance(strides, int) else list(strides)
        if self.training_config["frames_per_step"] < 2:
            raise ValueError("4D training needs frames_per_step >= 2")

    # ------------------------------------------------------------------
    _train_background = None

    def _render_own_supervision(self, gaussians):
        rgb, depth, alpha, valid = super()._render_own_supervision(gaussians)
        if self._train_background is not None:
            rgb = rgb + (1.0 - alpha.unsqueeze(1)) * self._train_background.view(1, 3, 1, 1)
        return rgb, depth, alpha, valid

    def _select_4d_frames(self, frame_count):
        frames = self.training_config["frames_per_step"]
        options = [s for s in self.gs4d["frame_stride"] if s * (frames - 1) < frame_count]
        if not options:
            raise ValueError("clip too short for frames_per_step x frame_stride")
        stride = options[int(torch.randint(len(options), (1,), generator=self.generator))]
        start = int(torch.randint(frame_count - stride * (frames - 1), (1,),
                                  generator=self.generator))
        return start + stride * torch.arange(frames)

    def _perceptual(self, rendered, target, valid):
        if self.perceptual_metric is None:
            return torch.zeros((), device=self.device)
        resolution = self.training_config["perceptual_resolution"]
        rendered = torch.where(valid.unsqueeze(1), rendered, target)
        rendered = F.interpolate(rendered, size=resolution, mode="bilinear",
                                 antialias=True).clamp(0.0, 1.0)
        target = F.interpolate(target, size=resolution, mode="bilinear",
                               antialias=True).clamp(0.0, 1.0)
        value = self.perceptual_metric(rendered, target)
        self.perceptual_metric.reset()
        return value

    def _update_aa_eps(self, global_step):
        if self.rasterize_mode != "antialiased":
            return
        if self.aa_eps2d_ramp_steps > 0:
            progress = min(max(float(global_step), 0.0) / self.aa_eps2d_ramp_steps, 1.0)
            self.active_rasterize_eps2d = self.aa_eps2d_start + progress * (
                self.aa_eps2d_end - self.aa_eps2d_start)
        else:
            self.active_rasterize_eps2d = self.aa_eps2d_end

    # ------------------------------------------------------------------
    def train_step(self, batch: dict, global_step: int):
        step_start = time.time()
        self.decoder_wrapper.train()
        self._update_aa_eps(global_step)
        self._prepare_nuscenes6_supervision(batch)
        self._train_background = None
        if self.gs4d["random_background"]:
            colour = torch.rand(3, generator=self.generator)
            prob = self.gs4d["random_background_prob"]
            # prob = 1 draws nothing extra, so the RNG stream (frame choice)
            # matches runs made before this option existed.
            if prob >= 1.0 or float(torch.rand((), generator=self.generator)) < prob:
                self._train_background = colour.to(self.device)
        vae_images = batch["vae_images"]
        if vae_images.shape[0] != 1:
            raise ValueError("4D training expects batch_size 1")
        frame_indices = self._select_4d_frames(vae_images.shape[1])
        frame_count = len(frame_indices)
        half_images = vae_images[:, frame_indices].flatten(0, 1).to(self.device)
        use_depth = (self.training_config.get("weight_render_depth", 0.0) > 0.0
                     or self.training_config.get("weight_range_l1", 0.0) > 0.0)
        if use_depth:
            gt_ranges, gt_range_valid = self._load_frame_ranges(
                batch, frame_indices, return_valid=True)
            gt_range_valid = gt_range_valid & self.pixel_content_ray_valid.unsqueeze(0)
        latent_origins, pixel_origins = self._frame_ray_origins(batch, frame_indices)
        camera_origin_scale = self._camera_origin_scale(global_step)
        ego_to_global = batch["ego_transforms"][0, frame_indices, 0].to(
            self.device, dtype=torch.float32)
        content = self.pixel_content_ray_valid.reshape(-1)

        gaussians, owners, targets = [], [], []
        own_total = torch.zeros((), device=self.device)
        report = {"loss": 0.0, "own_rgb_l1": 0.0, "own_perceptual": 0.0,
                  "own_render_depth_l1": 0.0, "own_range_l1": 0.0,
                  "composite_rgb_l1": 0.0, "composite_perceptual": 0.0,
                  "loo_rgb_l1": 0.0, "loo_coverage": 0.0, "motion_rgb_l1": 0.0,
                  "dynamic_fraction": 0.0,
                  "rasterize_eps2d": self.active_rasterize_eps2d,
                  "frame_span": float(frame_indices[-1] - frame_indices[0])}
        for k in range(frame_count):
            frame_images = half_images[k:k + 1]
            with torch.no_grad():
                latents = self.encode_half_latents(frame_images)
            decoder_kwargs = dict(
                previous_latents=None,
                temporal_context_mask=0.0,
                latent_origins=None if latent_origins is None else latent_origins[k:k + 1],
                pixel_origins=None if pixel_origins is None else pixel_origins[k:k + 1],
                camera_origin_scale=camera_origin_scale)
            decoder_args = (latents, self.latent_directions.unsqueeze(0),
                            self.pixel_directions.unsqueeze(0), None)
            if self.gs4d["checkpoint_decoder"]:
                # With one process DDP has nothing to synchronise; recomputing
                # through its wrapper would re-run its reducer bookkeeping, so
                # checkpoint the bare module.  Multi-GPU keeps the wrapper.
                single = (not torch.distributed.is_initialized()
                          or torch.distributed.get_world_size() == 1)
                module = self.decoder if single else self.decoder_wrapper
                g, predicted_range, _ = torch.utils.checkpoint.checkpoint(
                    module, *decoder_args, use_reentrant=False, **decoder_kwargs)
            else:
                g, predicted_range, _ = self.decoder_wrapper(*decoder_args, **decoder_kwargs)
            g = g[0]
            rendered_rgb, rendered_depth, own_alpha, render_valid = self._render_own_supervision(g)
            with torch.no_grad():
                if use_depth:
                    target_rgb, target_depth, target_valid = self._build_own_supervision_targets(
                        frame_images, gt_ranges[k:k + 1], gt_range_valid[k:k + 1],
                        None if pixel_origins is None else pixel_origins[k:k + 1],
                        camera_origin_scale)
                else:
                    target_rgb = self._target_rgb_for_supervision(frame_images)
            rgb_l1 = masked_mean((rendered_rgb - target_rgb).abs(), render_valid)
            perceptual = self._perceptual(rendered_rgb, target_rgb, render_valid)
            own = (self.training_config["weight_rgb_l1"] * rgb_l1
                   + self.training_config["weight_perceptual"] * perceptual)
            if use_depth:
                depth_l1 = masked_mean((rendered_depth - target_depth).abs(),
                                       target_valid & render_valid)
                range_l1 = masked_mean((predicted_range - gt_ranges[k:k + 1]).abs(),
                                       gt_range_valid[k:k + 1])
                own = own + self.training_config["weight_render_depth"] * depth_l1 \
                    + self.training_config["weight_range_l1"] * range_l1
                report["own_render_depth_l1"] += depth_l1.item() / frame_count
                report["own_range_l1"] += range_l1.item() / frame_count
            if self.gs4d["weight_coverage"] > 0.0:
                uncovered = masked_mean((1.0 - own_alpha).clamp_min(0.0), render_valid)
                own = own + self.gs4d["weight_coverage"] * uncovered
                report["own_uncovered"] = report.get("own_uncovered", 0.0) + uncovered.item() / frame_count
            own_total = own_total + own / frame_count
            report["own_rgb_l1"] += rgb_l1.item() / frame_count
            report["own_perceptual"] += perceptual.item() / frame_count

            frame = int(frame_indices[k])
            with torch.no_grad():
                owner = points_in_boxes(
                    g[:, 0:3].detach(),
                    batch["box_centers"][0, frame].to(self.device),
                    batch["box_rotations"][0, frame].to(self.device),
                    batch["box_half_sizes"][0, frame].to(self.device),
                    valid=batch["box_valid"][0, frame],
                    margin=self.gs4d["box_margin_m"])
                owner[~content.to(owner.device)] = -2   # invalid rays: never used
            report["dynamic_fraction"] += float((owner >= 0).float().mean()) / frame_count
            gaussians.append(g)
            owners.append(owner)
            targets.append((target_rgb, render_valid))

        composite_total = torch.zeros((), device=self.device)
        loo_total = torch.zeros((), device=self.device)
        motion_total = torch.zeros((), device=self.device)
        permutation = torch.randperm(frame_count, generator=self.generator)
        target_list = permutation[:min(self.gs4d["composite_targets"], frame_count)].tolist()
        for t in target_list:
            target_rgb, _ = targets[t]
            moved_static = []
            for s in range(frame_count):
                to_t = relative_ego_transform(ego_to_global[s], ego_to_global[t])
                moved_static.append(transform_gaussians(gaussians[s][owners[s] == -1], to_t))
            dynamic_t = gaussians[t][owners[t] >= 0]

            composite = torch.cat(moved_static + [dynamic_t])
            rgb, _, _, valid = self._render_own_supervision(composite)
            comp_l1 = masked_mean((rgb - target_rgb).abs(), valid)
            comp_lpips = self._perceptual(rgb, target_rgb, valid)
            composite_total = composite_total + (
                comp_l1 + self.gs4d["weight_composite_perceptual"] * comp_lpips
            ) / len(target_list)
            report["composite_rgb_l1"] += comp_l1.item() / len(target_list)
            report["composite_perceptual"] += comp_lpips.item() / len(target_list)

            with torch.no_grad():
                if dynamic_t.shape[0]:
                    _, _, dyn_alpha, _ = self._render_own_supervision(dynamic_t.detach())
                    dynamic_pixels = dyn_alpha > self.gs4d["dynamic_pixel_alpha"]
                else:
                    dynamic_pixels = None

            if self.gs4d["weight_loo"] > 0.0 and frame_count > 1:
                others = [moved_static[s] for s in range(frame_count) if s != t]
                scene = torch.cat(others + [dynamic_t.detach()])
                rgb, _, alpha, valid = self._render_own_supervision(scene)
                mask = valid & (alpha.detach() > self.gs4d["loo_min_alpha"])
                if dynamic_pixels is not None:
                    mask = mask & ~dynamic_pixels
                loo_l1 = masked_mean((rgb - target_rgb).abs(), mask)
                loo_total = loo_total + loo_l1 / len(target_list)
                report["loo_rgb_l1"] += loo_l1.item() / len(target_list)
                report["loo_coverage"] += float(mask.float().mean()) / len(target_list)

            if self.gs4d["weight_motion"] > 0.0 and dynamic_pixels is not None:
                moved_dynamic = []
                frame_t = int(frame_indices[t])
                tracks_t = batch["box_track"][0, frame_t].tolist()
                for s in range(frame_count):
                    if s == t:
                        continue
                    frame_s = int(frame_indices[s])
                    tracks_s = batch["box_track"][0, frame_s].tolist()
                    for slot_s, track in enumerate(tracks_s):
                        if track < 0 or track not in tracks_t:
                            continue
                        members = gaussians[s][owners[s] == slot_s]
                        if not members.shape[0]:
                            continue
                        slot_t = tracks_t.index(track)
                        pose_s = box_pose(batch["box_centers"][0, frame_s, slot_s].to(self.device),
                                          batch["box_rotations"][0, frame_s, slot_s].to(self.device))
                        pose_t = box_pose(batch["box_centers"][0, frame_t, slot_t].to(self.device),
                                          batch["box_rotations"][0, frame_t, slot_t].to(self.device))
                        moved_dynamic.append(transform_gaussians(
                            members, pose_t @ torch.linalg.inv(pose_s)))
                if moved_dynamic:
                    scene = torch.cat([m.detach() for m in moved_static] + moved_dynamic)
                    rgb, _, _, valid = self._render_own_supervision(scene)
                    motion_l1 = masked_mean((rgb - target_rgb).abs(), valid & dynamic_pixels)
                    motion_total = motion_total + motion_l1 / len(target_list)
                    report["motion_rgb_l1"] += motion_l1.item() / len(target_list)

        loss = (self.gs4d["weight_own"] * own_total
                + self.gs4d["weight_composite"] * composite_total
                + self.gs4d["weight_loo"] * loo_total
                + self.gs4d["weight_motion"] * motion_total)
        loss.backward()
        if "max_norm_for_grad_clip" in self.training_config:
            torch.nn.utils.clip_grad_norm_(self.decoder_wrapper.parameters(),
                                           self.training_config["max_norm_for_grad_clip"])
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._train_background = None
        report["loss"] = loss.item()
        self.loss_report_list.append(report)
        self.step_duration += time.time() - step_start
