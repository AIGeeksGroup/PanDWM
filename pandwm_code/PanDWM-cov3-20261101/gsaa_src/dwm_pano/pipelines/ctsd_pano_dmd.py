"""OpenDWM pano adaptation of Causal-Forcing++ Stage 3 (Asymmetric DMD).

Three DiT instances (all `DiTCrossviewTemporalConditionModel`):
  generator   — trainable, AR causal mask, init from a Stage 2 ckpt
  real_score  — frozen bidirectional teacher (NO causal mask), init from a
                Stage 0 bidirectional ckpt; provides the "what's good" signal
  fake_score  — trainable critic, AR causal mask, init from a Stage 2 ckpt;
                tracks the generator's current distribution for DMD's KL grad

Each `train_step` does 1 critic update; every `dfake_gen_update_ratio` calls
also do 1 generator update with the DMD pseudo-loss. The math mirrors
`externals/Causal-Forcing/model/dmd.py` and `trainer/distillation.py`.

Simplifications vs official: no I2V, no KV cache, fixed 19-frame block, full
OpenDWM conditioning (text+3dbox+hdmap+camera) routed through all three nets.
"""
import copy
import os
import re
import time

import dwm.common
import dwm.functional
import einops
import torch
import torch.distributed.checkpoint.state_dict
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import dwm_pano.distributed as _dwm_distributed
from dwm_pano.pipelines.ctsd_pano import CrossviewTemporalSD, PanoCrossviewTemporalSD


class PanoCrossviewTemporalDMD(PanoCrossviewTemporalSD):
    """Causal-Forcing++ Stage 3 — Asymmetric Distribution Matching Distillation."""

    def __init__(
        self,
        output_path,
        config: dict,
        device,
        common_config: dict,
        training_config: dict,
        inference_config: dict,
        pretrained_model_name_or_path: str,
        model,
        model_dtype=None,
        model_checkpoint_path=None,
        model_load_state_args: dict = {},
        metrics: dict = {},
        resume_from=None,
        # DMD-specific fields
        real_teacher_checkpoint_path: str = None,
        fake_score_checkpoint_path: str = None,
        denoising_step_list=None,
        warp_denoising_step: bool = True,
        num_train_timestep: int = 1000,
        timestep_shift: float = 5.0,
        real_guidance_scale: float = 3.0,
        fake_guidance_scale: float = 0.0,
        dfake_gen_update_ratio: int = 5,
        lr_critic: float = 4e-7,
        beta1_critic: float = 0.0,
        beta2_critic: float = 0.999,
        weight_decay_critic: float = 0.01,
        max_norm_for_critic_grad_clip: float = 10.0,
        ema_weight: float = 0.99,
        ema_start_step: int = 200,
        min_score_timestep: int = 0,
        min_step: int = 20,
        max_step: int = 980,
        critic_gradient_checkpointing: bool = True,
        **unused_official_kwargs,
    ):
        super().__init__(
            output_path=output_path,
            config=config,
            device=device,
            common_config=common_config,
            training_config=training_config,
            inference_config=inference_config,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model=model,
            model_dtype=model_dtype,
            model_checkpoint_path=model_checkpoint_path,
            model_load_state_args=model_load_state_args,
            metrics=metrics,
            resume_from=resume_from)

        self.real_guidance_scale = real_guidance_scale
        self.fake_guidance_scale = fake_guidance_scale
        self.dfake_gen_update_ratio = dfake_gen_update_ratio
        self.ema_decay = ema_weight
        self.ema_start_step = ema_start_step
        self.min_score_timestep = min_score_timestep
        self.min_step = min_step
        self.max_step = max_step
        self.num_train_timestep = num_train_timestep
        self.timestep_shift = timestep_shift
        self.max_norm_for_critic_grad_clip = max_norm_for_critic_grad_clip
        # Whether the fake_score (critic) attends causally. Official DMD runs
        # fake_score bidirectional (`is_causal=False`, base.py:51). Default True
        # preserves the legacy causal-critic behavior; set False in config to
        # match official (must pair with a bidirectional fake_score init).
        self.fake_score_causal = self.training_config.get(
            "fake_score_causal", True)
        # Random-exit backward simulation in `_run_generator` (official
        # Self-Forcing `last_step_only=false`). False = legacy last-step only.
        self.random_exit_step = self.training_config.get(
            "random_exit_step", True)

        # Few-step denoising schedule (same as Stage 2): the 4 timesteps the
        # generator is trained to predict x0 directly at.
        raw_steps = denoising_step_list or [1000, 750, 500, 250]
        self.denoising_step_list = self._make_official_denoising_step_list(
            raw_steps, warp_denoising_step, num_train_timestep)
        self.denoising_sigmas = self._compute_denoising_sigmas()

        # Training-time scheduler with `num_train_timestep` discrete timesteps.
        # Used to sample arbitrary t for DMD KL grad noise injection and
        # critic flow-match loss.
        _train_sched = copy.deepcopy(self.test_scheduler)
        _train_sched.set_timesteps(num_train_timestep, self.device)
        self.train_ts = _train_sched.timesteps.to(self.device)
        self.train_sigmas = _train_sched.sigmas.to(self.device)

        # Build real_score (bidirectional teacher, frozen). Loaded before
        # FSDP wrap inside `_build_auxiliary_model`.
        self.real_score, self.real_score_wrapper = self._build_auxiliary_model(
            checkpoint_path=real_teacher_checkpoint_path,
            model_load_state_args=model_load_state_args,
            requires_grad=False,
            role="real_score")

        # Build fake_score (causal critic, trainable). On resume, swap the
        # init path to the saved critic ckpt; otherwise use the configured
        # init (typically the same Stage 2 ckpt as the generator).
        fake_init = fake_score_checkpoint_path or model_checkpoint_path
        if resume_from is not None:
            fake_init = os.path.join(
                output_path, "checkpoints", f"{resume_from}_critic.pth")
        self.fake_score, self.fake_score_wrapper = self._build_auxiliary_model(
            checkpoint_path=fake_init,
            model_load_state_args=model_load_state_args,
            requires_grad=True,
            role="fake_score")
        if critic_gradient_checkpointing:
            self.fake_score.enable_gradient_checkpointing()

        # Critic optimizer (separate Adam group from the generator's
        # `self.optimizer` created by the parent).
        self.critic_optimizer = torch.optim.AdamW(
            self.fake_score_wrapper.parameters(),
            lr=lr_critic,
            betas=(beta1_critic, beta2_critic),
            weight_decay=weight_decay_critic)

        if resume_from is not None:
            _dwm_distributed.distributed_load_optimizer_state(
                self.fake_score_wrapper, self.critic_optimizer,
                os.path.join(output_path, "optimizer"),
                f"{resume_from}_critic")

        # Cache for the last generator-step metrics so non-gen steps still emit
        # the same row schema in `loss_report_list` (base `log()` averages over
        # row[0].keys() and would KeyError on the first non-gen row otherwise).
        # Holding the last value (rather than 0-padding) keeps the running mean
        # representative instead of diluted by 1/dfake_gen_update_ratio.
        self._last_gen_loss = 0.0
        self._last_dmd_grad_norm = 0.0

        # EMA on generator full state_dict. Built lazily once we cross
        # `ema_start_step`. On resume past that step, restore from saved.
        self.ema_state = None
        if resume_from is not None and resume_from >= ema_start_step:
            ema_path = os.path.join(
                output_path, "checkpoints", f"{resume_from}_ema.pth")
            if os.path.exists(ema_path):
                self.ema_state = CrossviewTemporalSD.load_state(ema_path)

    # ----- Helpers (copied from CD pipeline patterns) -----

    def _make_official_denoising_step_list(
        self, denoising_step_list, warp_denoising_step, num_train_timestep
    ):
        # Warp raw [1000, 750, 500, 250] through the FlowMatch scheduler so
        # the timesteps actually match scheduler.timesteps at inference.
        raw_steps = torch.tensor(
            denoising_step_list, dtype=torch.long, device=self.device)
        if not warp_denoising_step:
            return raw_steps
        scheduler = copy.deepcopy(self.test_scheduler)
        scheduler.set_timesteps(num_train_timestep, self.device)
        timesteps = torch.cat((
            scheduler.timesteps.detach().cpu(),
            torch.tensor([0], dtype=torch.float32),
        )).to(self.device)
        indices = num_train_timestep - raw_steps
        return timesteps[indices]

    def _compute_denoising_sigmas(self):
        # Look up FlowMatch sigma at each denoising step so generator forwards
        # can convert velocity → x0 via `x0 = noisy + (-sigma) * velocity`.
        lookup = copy.deepcopy(self.test_scheduler)
        lookup.set_timesteps(self.num_train_timestep, self.device)
        lookup_t = lookup.timesteps.to(self.device)
        lookup_s = lookup.sigmas.to(self.device)
        return torch.stack([
            lookup_s[(lookup_t - t).abs().argmin()]
            for t in self.denoising_step_list
        ]).to(self.device)

    def _build_auxiliary_model(
        self, checkpoint_path, model_load_state_args, requires_grad, role
    ):
        model = dwm.common.create_instance_from_config(
            self.config["pipeline"]["model"])
        model = model.to(dtype=self.model_dtype)
        if requires_grad:
            model.enable_gradient_checkpointing()
        if checkpoint_path is not None:
            state_dict = CrossviewTemporalSD.load_state(checkpoint_path)
            model.load_state_dict(state_dict, **model_load_state_args)
        model.requires_grad_(requires_grad)
        wrapper = self._wrap_auxiliary_model(model)
        if not requires_grad:
            wrapper.eval()
        return model, wrapper

    def _wrap_auxiliary_model(self, model):
        distribution_framework = self.common_config.get(
            "distribution_framework", "ddp")
        if (
            not torch.distributed.is_initialized()
            or distribution_framework == "ddp"
        ):
            model.to(self.device)
            return model

        ignored_modules = None
        if "fsdp_ignored_module_pattern" in self.common_config:
            pattern = re.compile(self.common_config["fsdp_ignored_module_pattern"])
            ignored_modules = [
                module for name, module in model.named_modules()
                if pattern.match(name) is not None
            ]

        return FSDP(
            model,
            device_id=torch.cuda.current_device(),
            ignored_modules=ignored_modules,
            **self.common_config["ddp_wrapper_settings"])

    def _add_temporal_attention_mask(self, model_conditions: dict, latent_shape):
        # Causal BlockMask — only applied to generator and fake_score
        # (real_score stays bidirectional).
        tam_setting = self.training_config.get("temporal_attention_mask")
        if tam_setting is None:
            return
        sequence_length = latent_shape[1]
        attn_type = self.model.temporal_attention_type
        if attn_type == "pointwise":
            causal = torch.zeros(
                sequence_length, sequence_length,
                device=self.device, dtype=self.model_dtype)
            causal.masked_fill_(
                torch.triu(
                    torch.ones(sequence_length, sequence_length,
                               dtype=torch.bool, device=self.device),
                    diagonal=1),
                float("-inf"))
            model_conditions["temporal_attention_mask"] = causal
        elif attn_type in ("rowwise", "full"):
            from dwm_pano.models.flex_block_causal import \
                make_blockwise_causal_block_mask
            hp = latent_shape[4] // self.model.config.patch_size
            wp = latent_shape[5] // self.model.config.patch_size
            frame_seqlen = wp if attn_type == "rowwise" else hp * wp
            model_conditions["temporal_attention_mask"] = \
                make_blockwise_causal_block_mask(
                    self.device,
                    num_frames=sequence_length,
                    frame_seqlen=frame_seqlen)

    # ----- Batch / latent helpers -----

    def _get_latent_shape(self, batch, seq_len, view_count):
        H_lat = batch["vae_images"].shape[-2] // (
            2 ** (len(self.vae.config.down_block_types) - 1))
        W_lat = batch["vae_images"].shape[-1] // (
            2 ** (len(self.vae.config.down_block_types) - 1))
        return (
            batch["vae_images"].shape[0], seq_len, view_count,
            self.vae.config.latent_channels, H_lat, W_lat)

    @torch.no_grad()
    def _vae_encode(self, vae_images):
        sequence_length = vae_images.shape[1]
        view_count = vae_images.shape[2]
        image_tensor = self.image_processor.preprocess(
            vae_images.flatten(0, 2).to(self.device))
        if self.is_temporal_vae:
            image_tensor = einops.rearrange(
                image_tensor, "(b t v) c h w -> (b v) c t h w",
                t=sequence_length, v=view_count)
        shift_factor = (
            self.vae.config.shift_factor
            if self.vae.config.shift_factor is not None else 0)
        latents = dwm.functional.memory_efficient_split_call(
            self.vae, image_tensor,
            lambda block, tensor: (
                block.encode(tensor).latent_dist.sample() - shift_factor
            ) * block.config.scaling_factor,
            self.common_config.get("memory_efficient_batch", -1))
        if self.is_temporal_vae:
            latents = einops.rearrange(
                latents, "(b v) c t h w -> b t v c h w", v=view_count)
        else:
            latents = einops.rearrange(
                latents, "(b t v) c h w -> b t v c h w",
                t=sequence_length, v=view_count)
        return latents

    # ----- 4-step AR generator forward -----

    def _run_generator(
        self, latent_shape, cond, image_latents=None,
        reference_frame_count: int = 0, grad_on_last: bool = True,
    ):
        # 4-step AR backward simulation over `denoising_step_list`. The rollout
        # EXITS at a step `exit_idx` and returns that step's x0 prediction; the
        # gradient-carrying forward runs only at the exit step (when
        # `grad_on_last`), earlier steps re-noise under no_grad. `_critic_step`
        # calls with `grad_on_last=False` to get the SAME exit sample no_grad.
        #
        # `random_exit_step` (default) picks a RANDOM exit step (official
        # Self-Forcing `last_step_only=false`: `self_forcing_training.py:48-66`
        # `generate_and_sync_list` + `:182-228` break-at-exit). This is what
        # lets the critic see the generator's x0 across all noise levels (not
        # only the near-clean last step). `random_exit_step=false` reproduces
        # the legacy always-last-step behavior.
        if grad_on_last:
            self.model_wrapper.train()
        else:
            self.model_wrapper.eval()

        latents = torch.randn(
            latent_shape, generator=self.generator).to(self.device)

        cond_gen = dict(cond)
        self._add_temporal_attention_mask(cond_gen, latent_shape)

        n_steps = len(self.denoising_step_list)
        # One exit index shared across ranks — required so every FSDP rank runs
        # the same number of forwards and the grad step lands on the same index
        # (else the all-gather / reduce-scatter collectives desync). Matches
        # official `same_step_across_blocks=true` + `dist.broadcast`. Sampled
        # off `self.generator` so the per-step noise stream is unaffected.
        if self.random_exit_step:
            exit_idx = torch.randint(0, n_steps, (1,), device=self.device)
            torch.distributed.broadcast(exit_idx, src=0)
            exit_idx = int(exit_idx.item())
        else:
            exit_idx = n_steps - 1

        for i in range(n_steps):
            is_exit = (i == exit_idx)
            t_val = self.denoising_step_list[i]
            sigma_t = self.denoising_sigmas[i].to(dtype=latents.dtype)
            timesteps = torch.full(
                latent_shape[:3], float(t_val.item()),
                dtype=t_val.dtype, device=self.device)

            inp = latents
            if image_latents is not None and reference_frame_count > 0:
                inp = torch.cat([
                    image_latents[:, :reference_frame_count],
                    latents[:, reference_frame_count:],
                ], dim=1)
                timesteps[:, :reference_frame_count] = 0
            inp = inp.to(dtype=self.model_dtype)

            grad_ctx = (
                torch.enable_grad()
                if (grad_on_last and is_exit)
                else torch.no_grad())
            with grad_ctx, self.get_autocast_context():
                model_output, _, _ = self.model_wrapper(
                    inp, timesteps, **cond_gen)
            velocity = model_output[0]
            x0_pred = inp + (-sigma_t) * velocity

            if is_exit:
                return x0_pred

            sigma_next = self.denoising_sigmas[i + 1].to(dtype=latents.dtype)
            new_noise = torch.randn(
                latent_shape, generator=self.generator).to(self.device)
            latents = (1.0 - sigma_next) * x0_pred + sigma_next * new_noise

    # ----- DMD timestep sampling -----

    def _sample_dmd_timestep(self, shape, dtype):
        # Match `model/dmd.py:158-172`: uniform int t in
        # [min_score_timestep, num_train_timestep), apply timestep shift,
        # clamp to [min_step, max_step], snap to nearest scheduler step for
        # sigma lookup. `ts_schedule=False` semantics — we don't narrow the
        # range to the generator's last-step denoised window.
        #
        # ONE t per sample, broadcast across frames/views (uniform_timestep).
        # The frozen real_score teacher, the critic init, the generator, and
        # inference (`inference_pipeline` uses `torch.full(latent_shape[:3])`)
        # are all trained/run with `frame_prediction_style="ctsd"` = one shared
        # t per video. Sampling per-frame/per-view t here queries the FROZEN
        # bidirectional teacher out-of-distribution → corrupted `real_x0` DMD
        # target. Matches official `uniform_timestep = not is_causal`
        # (`wan_wrapper.py:134`) for the non-causal score networks.
        t_raw = torch.randint(
            self.min_score_timestep, self.num_train_timestep,
            (shape[0], 1, 1), device=self.device
        ).float().expand(shape).contiguous()
        if self.timestep_shift > 1:
            t = (self.timestep_shift * (t_raw / 1000)
                 / (1.0 + (self.timestep_shift - 1) * (t_raw / 1000))
                 * 1000)
        else:
            t = t_raw
        t = t.clamp(self.min_step, self.max_step)

        abs_diff = (self.train_ts.view(1, -1) - t.flatten().view(-1, 1)).abs()
        sigma_idx = abs_diff.argmin(dim=1)
        sigmas = self.train_sigmas[sigma_idx].view(*t.shape)
        sigmas_b = sigmas.view(*sigmas.shape, 1, 1, 1).to(dtype=dtype)
        return t, sigmas_b

    # ----- DMD KL grad + pseudo-loss -----

    def _compute_kl_grad(
        self, noisy_x0, ts, sigmas_b, x0_pred_detached, cond, cond_cfg
    ):
        # Fake score (no CFG: fake_guidance_scale=0 by default). Causal mask
        # only when `fake_score_causal` (default True); official runs it
        # bidirectional.
        cond_fake = dict(cond)
        if self.fake_score_causal:
            self._add_temporal_attention_mask(cond_fake, noisy_x0.shape)
        with self.get_autocast_context():
            fake_out, _, _ = self.fake_score_wrapper(
                noisy_x0.to(self.model_dtype), ts, **cond_fake)
        fake_v = fake_out[0]
        fake_x0 = noisy_x0.float() + (-sigmas_b).float() * fake_v.float()

        # Real score (bidirectional, CFG): forward the 2B uncond+cond stack
        # once. `cond_cfg` from `get_conditions(..., CFG=True)` is uncond-then-cond.
        # Strip the causal mask key — bidirectional teacher must NOT see it.
        noisy_cfg = torch.cat([noisy_x0, noisy_x0], dim=0)
        ts_cfg = torch.cat([ts, ts], dim=0)
        sigmas_cfg = torch.cat([sigmas_b, sigmas_b], dim=0)
        cond_real = {
            k: v for k, v in cond_cfg.items() if k != "temporal_attention_mask"
        }
        with self.get_autocast_context():
            real_out, _, _ = self.real_score_wrapper(
                noisy_cfg.to(self.model_dtype), ts_cfg, **cond_real)
        real_v = real_out[0]
        real_x0_cfg = noisy_cfg.float() + (-sigmas_cfg).float() * real_v.float()
        real_x0_uncond, real_x0_cond = real_x0_cfg.chunk(2, dim=0)
        real_x0 = real_x0_cond + (real_x0_cond - real_x0_uncond) * self.real_guidance_scale

        # DMD eq 7 with per-sample normalization (eq 8).
        grad = fake_x0 - real_x0
        p_real = x0_pred_detached.float() - real_x0
        normalizer = p_real.abs().mean(
            dim=list(range(1, p_real.ndim)), keepdim=True
        ).clamp(min=1e-8)
        grad = grad / normalizer
        return torch.nan_to_num(grad).to(dtype=x0_pred_detached.dtype)

    def compute_distribution_matching_loss(
        self, x0_pred, cond, cond_cfg, gradient_mask=None
    ):
        with torch.no_grad():
            ts, sigmas_b = self._sample_dmd_timestep(
                x0_pred.shape[:3], x0_pred.dtype)

            noise = torch.randn_like(x0_pred)
            x0_det = x0_pred.detach()
            noisy_x0 = sigmas_b * noise + (1.0 - sigmas_b) * x0_det

            grad = self._compute_kl_grad(
                noisy_x0, ts, sigmas_b, x0_det, cond, cond_cfg)

        # Pseudo-loss: grad flows ONLY through x0_pred. d(loss)/d(x0_pred) = grad.
        target = (x0_det.double() - grad.double()).detach()
        diff = x0_pred.double() - target
        if gradient_mask is not None:
            mask = gradient_mask.to(dtype=diff.dtype)
            loss = 0.5 * (diff.square() * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = 0.5 * diff.square().mean()
        return loss, grad.abs().mean().detach()

    # ----- Update steps -----

    def _generator_step(self, latent_shape, cond, cond_cfg, image_latents, ref_count):
        x0_pred = self._run_generator(
            latent_shape, cond, image_latents=image_latents,
            reference_frame_count=ref_count, grad_on_last=True)

        gradient_mask = None
        if image_latents is not None and ref_count > 0:
            gradient_mask = torch.ones(
                latent_shape, dtype=torch.bool, device=self.device)
            gradient_mask[:, :ref_count] = False

        gen_loss, grad_norm = self.compute_distribution_matching_loss(
            x0_pred, cond, cond_cfg, gradient_mask=gradient_mask)

        use_scaler = self.training_config.get("enable_grad_scaler", False)
        if use_scaler:
            self.grad_scaler.scale(gen_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
        else:
            gen_loss.backward()

        if "max_norm_for_grad_clip" in self.training_config:
            self.model_wrapper.clip_grad_norm_(
                self.training_config["max_norm_for_grad_clip"])

        if use_scaler:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

        self._last_gen_loss = gen_loss.item()
        self._last_dmd_grad_norm = float(grad_norm)
        return {
            "gen_loss": self._last_gen_loss,
            "dmd_grad_norm": self._last_dmd_grad_norm,
        }

    def _critic_step(self, latent_shape, cond, image_latents, ref_count):
        self.fake_score_wrapper.train()

        with torch.no_grad():
            x0_pred = self._run_generator(
                latent_shape, cond, image_latents=image_latents,
                reference_frame_count=ref_count, grad_on_last=False)

        ts, sigmas_b = self._sample_dmd_timestep(
            x0_pred.shape[:3], x0_pred.dtype)

        noise = torch.randn_like(x0_pred)
        noisy = sigmas_b * noise + (1.0 - sigmas_b) * x0_pred
        # Flow-match velocity target: v = noise − x0.
        target_v = noise - x0_pred

        cond_critic = dict(cond)
        if self.fake_score_causal:
            self._add_temporal_attention_mask(cond_critic, latent_shape)

        with self.get_autocast_context():
            critic_out, _, _ = self.fake_score_wrapper(
                noisy.to(self.model_dtype), ts, **cond_critic)
        pred_v = critic_out[0]

        critic_loss = F.mse_loss(
            pred_v.float(), target_v.float(), reduction="mean")

        use_scaler = self.training_config.get("enable_grad_scaler", False)
        if use_scaler:
            self.grad_scaler.scale(critic_loss).backward()
            self.grad_scaler.unscale_(self.critic_optimizer)
        else:
            critic_loss.backward()

        self.fake_score_wrapper.clip_grad_norm_(self.max_norm_for_critic_grad_clip)

        if use_scaler:
            self.grad_scaler.step(self.critic_optimizer)
            self.grad_scaler.update()
        else:
            self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        return {"critic_loss": critic_loss.item()}

    # ----- Train loop / EMA / save -----

    def train_step(self, batch: dict, global_step: int):
        self.real_score_wrapper.eval()
        t0 = time.time()

        seq_len = self.inference_config["sequence_length_per_iteration"]
        view_count = batch["vae_images"].shape[2]
        latent_shape = self._get_latent_shape(batch, seq_len, view_count)

        # Reference frames (clean latents to seed the AR). Mirrors Stage 2's
        # `generate_frames_for_reference=false` behavior — VAE-encode the
        # first `reference_frame_count` real frames; the 4-step loop injects
        # them at every step with t=0.
        image_latents = None
        ref_count = self.inference_config.get("reference_frame_count", 0)
        if (
            not self.inference_config.get("generate_frames_for_reference", True)
            and ref_count > 0
        ):
            image_latents = self._vae_encode(batch["vae_images"][:, :ref_count])

        # Build conditioning twice: single-B for generator/fake_score, CFG-2B
        # (uncond+cond stack) for real_score CFG forward.
        with torch.no_grad(), self.get_autocast_context():
            cond = CrossviewTemporalSD.get_conditions(
                self.model, self.text_encoders, self.tokenizers,
                self.common_config, latent_shape, batch, self.device,
                self.model_dtype,
                do_classifier_free_guidance=False,
                latents_shape=latent_shape)
            cond_cfg = CrossviewTemporalSD.get_conditions(
                self.model, self.text_encoders, self.tokenizers,
                self.common_config, latent_shape, batch, self.device,
                self.model_dtype,
                do_classifier_free_guidance=True,
                latents_shape=latent_shape)

        log = {
            "gen_loss": self._last_gen_loss,
            "dmd_grad_norm": self._last_dmd_grad_norm,
        }
        train_generator = (global_step % self.dfake_gen_update_ratio == 0)
        if train_generator:
            log.update(self._generator_step(
                latent_shape, cond, cond_cfg, image_latents, ref_count))
            if global_step >= self.ema_start_step:
                self._ema_update()

        log.update(self._critic_step(
            latent_shape, cond, image_latents, ref_count))

        self.step_duration += time.time() - t0
        self.loss_report_list.append({
            "loss": log.get("critic_loss", 0.0),
            **log,
        })

    @torch.no_grad()
    def _ema_update(self):
        # EMA on the generator's FULL state_dict (rank0 gathered, CPU-offloaded)
        # — cheap to save/load, decoupled from FSDP shard layout.
        if torch.distributed.is_initialized():
            options = torch.distributed.checkpoint.state_dict.StateDictOptions(
                full_state_dict=True, cpu_offload=True)
            cur_sd = torch.distributed.checkpoint.state_dict.get_model_state_dict(
                self.model_wrapper, options=options)
        else:
            cur_sd = self.model.state_dict()

        if self.should_save:
            if self.ema_state is None:
                self.ema_state = {k: v.detach().clone() for k, v in cur_sd.items()}
            else:
                for k, v in cur_sd.items():
                    self.ema_state[k].mul_(self.ema_decay).add_(
                        v.detach(), alpha=1.0 - self.ema_decay)

        # Drain the full-mesh all_gather kicked off by `get_model_state_dict`
        # before train_step returns — otherwise its residual stream state
        # trips the next save's optim plan-gather on the shard subgroup.
        # All ranks must reach this barrier (no early-return above).
        if torch.distributed.is_initialized():
            torch.cuda.synchronize()
            torch.distributed.barrier()

    def save_checkpoint(self, output_path: str, steps: int, upload: bool = True,
                        keep_last_n: int = 0):
        # Drain in-flight NCCL collectives before the first gather.
        if torch.distributed.is_initialized():
            torch.cuda.synchronize()
            torch.distributed.barrier()

        # Optimizer states are NOT persisted — distcp's per-rank torch.save
        # trips WekaFS-zipfile EOF mismatches (`unexpected pos X vs Y`) on
        # this cluster. DMD is short (~1K steps per CF README); on resume
        # Adam momentum rebuilds in ~100 steps. Save model/critic/ema only.
        # `CKPT_OUT_DIR` env redirects writes to compute-node local disk
        # (`/local`, 6.9 TB, no shared-FS quota); rank 0 only writes, then
        # `_upload_and_prune` uploads from same dir + prunes locally.
        ckpt_dir = os.environ.get("CKPT_OUT_DIR") \
            or os.path.join(output_path, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1. Generator state (full state dict, rank-0 writes file).
        if torch.distributed.is_initialized():
            options = torch.distributed.checkpoint.state_dict.StateDictOptions(
                full_state_dict=True, cpu_offload=True)
            gen_sd = torch.distributed.checkpoint.state_dict.get_model_state_dict(
                self.model_wrapper, options=options)
        elif self.should_save:
            gen_sd = self.model.state_dict()
        # Legacy pickle serialization for all 3 torch.save calls below:
        # PyTorch 2 default `new_zipfile_serialization=True` does a strict
        # `write_end_of_file()` size check on context exit. WekaFS on `/data`
        # intermittently drops ~100 bytes at the end of large writes, raising
        # `unexpected pos X vs Y`. Legacy pickle is streaming + no EOF check.
        if self.should_save:
            torch.save(
                gen_sd, os.path.join(ckpt_dir, f"{steps}.pth"),
                _use_new_zipfile_serialization=False)

        # 2. Critic state.
        if torch.distributed.is_initialized():
            critic_sd = torch.distributed.checkpoint.state_dict.get_model_state_dict(
                self.fake_score_wrapper, options=options)
        elif self.should_save:
            critic_sd = self.fake_score.state_dict()
        if self.should_save:
            torch.save(
                critic_sd, os.path.join(ckpt_dir, f"{steps}_critic.pth"),
                _use_new_zipfile_serialization=False)

        # EMA snapshot.
        if self.should_save and self.ema_state is not None:
            torch.save(
                self.ema_state,
                os.path.join(ckpt_dir, f"{steps}_ema.pth"),
                _use_new_zipfile_serialization=False)

        if upload and self.should_save:
            self._upload_and_prune(output_path, steps, keep_last_n=keep_last_n)

    def evaluate_4step_ema_pipeline(
        self, global_step, dataset_length, dataloader, datasampler
    ):
        # Eval with EMA weights. DMD's `inference_pipeline` is already 4-step
        # (no monkey-patch needed, unlike CD). Snapshot main generator full
        # state, load self.ema_state into model_wrapper, run base
        # evaluate_pipeline (routes to native 4-step), restore main.
        if self.ema_state is None:
            # Pre-ema_start_step: main is the only meaningful state.
            self.evaluate_pipeline(
                global_step, dataset_length, dataloader, datasampler)
            return

        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions, get_model_state_dict, set_model_state_dict)
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)

        # Snapshot main (rank 0 gets full dict, other ranks get {}).
        main_state = get_model_state_dict(self.model_wrapper, options=options)

        # Load EMA into model_wrapper. ema_state lives only on rank 0 (per
        # `_ema_update` early-return); non-rank-0 passes {} and lets the
        # FSDP-aware set handle the broadcast internally.
        ema_to_load = self.ema_state if self.should_save else {}
        set_model_state_dict(
            self.model_wrapper, ema_to_load, options=options)
        try:
            self.evaluate_pipeline(
                global_step, dataset_length, dataloader, datasampler)
        finally:
            main_to_load = main_state if self.should_save else {}
            set_model_state_dict(
                self.model_wrapper, main_to_load, options=options)

    # ----- 4-step distilled inference (preview / eval path) -----

    def inference_pipeline(
        self, latent_shape, batch, output_type, image_latents=None,
        reference_frame_count: int = 0, start_timestep: int = 0,
        stop_timestep=None, take_time: int = 0,
    ):
        # 4-step path mirroring `_run_generator` + VAE decode. The parent's
        # `autoregressive_inference_pipeline` (called by `preview_pipeline`
        # and the evaluation loop) goes through this transparently.
        self.model_wrapper.eval()

        latents = torch.randn(
            latent_shape, generator=self.generator).to(self.device)

        cond = CrossviewTemporalSD.get_conditions(
            self.model, self.text_encoders, self.tokenizers,
            self.common_config, latent_shape, batch, self.device,
            self.model_dtype,
            do_classifier_free_guidance=False,
            latents_shape=latents.shape)
        self._add_temporal_attention_mask(cond, latent_shape)

        for i in range(len(self.denoising_step_list)):
            t_val = self.denoising_step_list[i]
            sigma_t = self.denoising_sigmas[i].to(dtype=latents.dtype)
            timesteps = torch.full(
                latent_shape[:3], float(t_val.item()),
                dtype=t_val.dtype, device=self.device)

            inp = latents
            if image_latents is not None and reference_frame_count > 0:
                inp = torch.cat([
                    image_latents[:, :reference_frame_count],
                    latents[:, reference_frame_count:],
                ], dim=1)
                timesteps[:, :reference_frame_count] = 0
            inp = inp.to(dtype=self.model_dtype)

            with self.get_autocast_context():
                model_output, _, _ = self.model_wrapper(
                    inp, timesteps, **cond)
            velocity = model_output[0]
            x0_pred = inp + (-sigma_t) * velocity

            if i < len(self.denoising_step_list) - 1:
                sigma_next = self.denoising_sigmas[i + 1].to(dtype=latents.dtype)
                new_noise = torch.randn(
                    latent_shape, generator=self.generator).to(self.device)
                latents = (1.0 - sigma_next) * x0_pred + sigma_next * new_noise
            else:
                latents = x0_pred

        shift_factor = (
            self.vae.config.shift_factor
            if self.vae.config.shift_factor is not None else 0)
        image_tensor = dwm.functional.memory_efficient_split_call(
            self.vae,
            latents.flatten(0, 2).to(dtype=self.vae.dtype),
            lambda block, tensor: block.decode(
                tensor / block.config.scaling_factor + shift_factor,
                return_dict=False)[0],
            self.common_config.get("memory_efficient_batch", -1))
        images = self.image_processor.postprocess(
            image_tensor, output_type=output_type)
        return {"images": images, "latents": latents}
