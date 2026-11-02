"""4D GS decoder with a usable depth term (2026-10-17, depth pilot; report section 34).

Why: the review renders of the locked video model (v84i4250) show the GS problems are geometric (curtains at depth
edges, seam cracks, ripple stripes, holes), while the original-camera round trip is fine.  The 4D decoder's held-out
frame PSNR plateaued at ~21.5 dB from step 1000 of the coverage run (round trip ~32.6 dB) and it was trained without
depth (no labels on HPC2).  HPC3 holds dense estimated panorama depth for 20000 train frames (pandwm_data/train/depth,
600 x 9600 float16, 100 m = hole / sky fill); the inherited depth terms are metre L1 at weight 0.002, dominated by far
pixels.

What changes (everything else is PanoGaussianDecoder4DPipeline.train_step, copied): on the own term, two log-depth L1
terms on pixels whose label is below far_cut_m:
    weight_log_range  * mean |log r_decoder - log r_label|     (the decoder's per-pixel range)
    weight_log_render * mean |log d_render  - log d_label|     (the rendered own-view depth)
Defaults 0 / 0 are the parent bit for bit.  Missing label files still raise, as in the parent: the pilot trains on the
G2 depth subset (pandwm_gs_aa_experiment/inputs/nuscenes_interp_12Hz_infos_train_p2.pkl, 18638 frames, every frame has
a label).

2026-10-18 (report section 38, formal run): depth_optional=True makes the depth terms per batch: a batch gets them only
if every selected frame has a label file in range_dir, otherwise it trains on the colour terms alone (the parent's
targets).  This lets one run mix all 700 training scenes (colour) with the 86 labelled ones (colour + depth).  The
report gains depth_batch (1 / 0).  False (default) keeps the pilot's behaviour.
"""

from __future__ import annotations

import os
import time

import torch
import torch.utils.checkpoint

from dwm_pano.pipelines.ctsd_pano_gs_decoder import masked_mean
from dwm_pano_4d.geometry import box_pose, points_in_boxes, relative_ego_transform, transform_gaussians
from dwm_pano_4d.pipeline import PanoGaussianDecoder4DPipeline

DEFAULT_DEPTH = {"weight_log_range": 0.0, "weight_log_render": 0.0, "far_cut_m": 99.0, "depth_optional": False}


class PanoGaussianDecoder4DDepthPipeline(PanoGaussianDecoder4DPipeline):
    def __init__(self, *args, gs4d_depth: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gs4d_depth = dict(DEFAULT_DEPTH)
        self.gs4d_depth.update(gs4d_depth or {})
        unknown = set(self.gs4d_depth) - set(DEFAULT_DEPTH)
        if unknown:
            raise ValueError(f"unknown gs4d_depth options: {sorted(unknown)}")

    def _extra_own_loss(self, batch, frame, gaussians):
        """Hook for subclasses (2026-10-24): an extra own-frame loss of frame `frame` given its Gaussians, as
        (scalar tensor or None, {report key: float}).  None adds nothing, so the parent's loss is unchanged."""
        return None, {}

    # A copy of PanoGaussianDecoder4DPipeline.train_step; the only changes are marked "depth pilot".
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
                     or self.training_config.get("weight_range_l1", 0.0) > 0.0
                     or self.gs4d_depth["weight_log_range"] > 0.0        # depth pilot
                     or self.gs4d_depth["weight_log_render"] > 0.0)      # depth pilot
        if use_depth and self.gs4d_depth["depth_optional"]:                  # formal run: labelled batches only
            range_dir = self.training_config["range_dir"]
            use_depth = all(os.path.exists(os.path.join(
                range_dir, "{}.h5".format(int(batch["frame_timestamps_us"][0, int(f)])))) for f in frame_indices)
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
                  "own_log_range_l1": 0.0, "own_log_render_l1": 0.0,      # depth pilot
                  "composite_rgb_l1": 0.0, "composite_perceptual": 0.0,
                  "loo_rgb_l1": 0.0, "loo_coverage": 0.0, "motion_rgb_l1": 0.0,
                  "dynamic_fraction": 0.0,
                  "rasterize_eps2d": self.active_rasterize_eps2d,
                  "frame_span": float(frame_indices[-1] - frame_indices[0]),
                  "depth_batch": float(use_depth)}                           # formal run
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
                # depth pilot: log-depth L1 on the decoder range and on the rendered own-view depth, labels below
                # far_cut_m only (the labels hold 100 m for holes / sky)
                far, eps = self.gs4d_depth["far_cut_m"], 1e-3
                gt_k = gt_ranges[k:k + 1]
                log_range = masked_mean(
                    (predicted_range.clamp_min(eps).log() - gt_k.clamp_min(eps).log()).abs(),
                    gt_range_valid[k:k + 1] & (gt_k < far) & (gt_k > 0.0))
                log_render = masked_mean(
                    (rendered_depth.clamp_min(eps).log() - target_depth.clamp_min(eps).log()).abs(),
                    target_valid & render_valid & (target_depth < far) & (target_depth > 0.0))
                own = own + self.gs4d_depth["weight_log_range"] * log_range \
                    + self.gs4d_depth["weight_log_render"] * log_render
                report["own_log_range_l1"] += log_range.item() / frame_count
                report["own_log_render_l1"] += log_render.item() / frame_count
            extra, extra_report = self._extra_own_loss(batch, int(frame_indices[k]), g)   # 2026-10-24 hook
            if extra is not None:
                own = own + extra
                for key, value in extra_report.items():
                    report[key] = report.get(key, 0.0) + value / frame_count
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
