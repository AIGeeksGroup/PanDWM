"""Context-Forced DMD (+ optional GAN) for the PanDWM 4-step student.

Builds on the repo's Stage-3 ``PanoCrossviewTemporalDMD`` (unchanged; this is
a subclass in a new package).  Following CausalDrive's Context-Forced DMD:

* every train step the student first rolls out ``context_windows`` 19-frame
  windows by itself (4-step sampler, no gradient), starting from the clip's 3
  real frames; the last 3 generated latents become the *history* of the
  trained window ("N-chunk self-rollout -> flawed memory context");
* the generator step, the DMD real/fake scores and the critic all condition on
  that student-made history: it is fed clean (t = 0) in the 3 reference slots
  and only the 16 target frames are noised.  The real score is the *causal* AR
  teacher (Stage-1) with the block-causal mask, so it answers "given this
  flawed past, what should come next" -- the gradient that teaches the student
  to recover from its own drift.  Losses never touch the reference slots;
* optionally, a small discriminator head on an intermediate block of the fake
  score separates real clips (ground-truth window + ground-truth history) from
  generated ones; the generator receives a non-saturating adversarial term
  (DMD2-style; CausalDrive attaches the discriminator to the fake score too).

All ranks run the same number of forwards every step (rollout length is fixed,
random choices are broadcast from rank 0), as FSDP requires.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

import dwm.functional
from dwm_pano.pipelines.ctsd_pano import CrossviewTemporalSD
from dwm_pano.pipelines.ctsd_pano_dmd import PanoCrossviewTemporalDMD

CONTEXT = 3
WINDOW = 19
STRIDE = 16


def _broadcast_scalar(value: float, device) -> float:
    t = torch.tensor([value], device=device, dtype=torch.float32)
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(t, src=0)
    return float(t.item())


class PanoContextForcedDMD(PanoCrossviewTemporalDMD):
    def __init__(
        self, *args,
        context_windows: int = 2,
        context_forcing_prob: float = 1.0,
        real_score_causal: bool = True,
        gan_generator_weight: float = 0.0,
        gan_discriminator_weight: float = 0.0,
        gan_feature_block: int = 12,
        gan_head_lr: float = 1e-5,
        autoregression_exceptions=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.context_windows = int(context_windows)
        self.context_forcing_prob = float(context_forcing_prob)
        self.real_score_causal = bool(real_score_causal)
        self.gan_generator_weight = float(gan_generator_weight)
        self.gan_discriminator_weight = float(gan_discriminator_weight)
        self.use_gan = self.gan_generator_weight > 0 or self.gan_discriminator_weight > 0
        self.window_exceptions = list(autoregression_exceptions or self.inference_config.get(
            "autoregression_data_exception_for_take_sequence", []))
        self._last_adv = 0.0
        self._features = {}
        self._capture = False
        if self.use_gan:
            hidden = (self.fake_score.config.num_attention_heads
                      * self.fake_score.config.attention_head_dim)
            self.disc_head = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, 256),
                torch.nn.SiLU(), torch.nn.Linear(256, 1)).to(self.device)
            self.disc_optimizer = torch.optim.AdamW(
                self.disc_head.parameters(), lr=gan_head_lr, betas=(0.0, 0.999),
                weight_decay=0.01)
            block = self.fake_score.transformer_blocks[gan_feature_block]
            block.register_forward_hook(self._feature_hook)
            resume = kwargs.get("resume_from")
            head_path = os.path.join(kwargs.get("output_path", ""), "checkpoints",
                                     f"{resume}_disc_head.pth")
            if resume is not None and os.path.exists(head_path):
                self.disc_head.load_state_dict(torch.load(head_path, map_location="cpu"))

    # ------------------------------------------------------------------
    def _feature_hook(self, module, inputs, output):
        if self._capture:
            # JointTransformerBlock returns (encoder_hidden_states, hidden_states).
            self._features["h"] = output[1] if isinstance(output, tuple) else output

    def _window_batch(self, batch, start):
        result = {
            key: (value if key in self.window_exceptions else
                  dwm.functional.take_sequence_clip(value, start, start + WINDOW))
            for key, value in batch.items()}
        result.pop("vae_images", None)
        return result

    def _conditions(self, batch, latent_shape, cfg):
        with torch.no_grad(), self.get_autocast_context():
            return CrossviewTemporalSD.get_conditions(
                self.model, self.text_encoders, self.tokenizers, self.common_config,
                latent_shape, batch, self.device, self.model_dtype,
                do_classifier_free_guidance=cfg, latents_shape=latent_shape)

    @torch.no_grad()
    def _rollout_window(self, latent_shape, cond, history):
        """Full 4-step sample of one window from `history` (deployed sampler)."""
        self.model_wrapper.eval()
        latents = torch.randn(latent_shape, generator=self.generator).to(self.device)
        cond_gen = dict(cond)
        self._add_temporal_attention_mask(cond_gen, latent_shape)
        for i in range(len(self.denoising_step_list)):
            t_val = self.denoising_step_list[i]
            sigma_t = self.denoising_sigmas[i].to(dtype=latents.dtype)
            timesteps = torch.full(latent_shape[:3], float(t_val.item()),
                                   dtype=t_val.dtype, device=self.device)
            timesteps[:, :CONTEXT] = 0
            inp = torch.cat([history.to(latents.dtype), latents[:, CONTEXT:]], dim=1)
            with self.get_autocast_context():
                out, _, _ = self.model_wrapper(inp.to(self.model_dtype), timesteps, **cond_gen)
            x0 = inp + (-sigma_t) * out[0]
            if i < len(self.denoising_step_list) - 1:
                sigma_next = self.denoising_sigmas[i + 1].to(dtype=latents.dtype)
                noise = torch.randn(latent_shape, generator=self.generator).to(self.device)
                latents = (1.0 - sigma_next) * x0 + sigma_next * noise
        return x0

    def _noised_with_clean_history(self, x0, history, noise=None):
        """Noise the 16 targets at one shared t; keep the 3 history slots clean."""
        ts, sigmas_b = self._sample_dmd_timestep(x0.shape[:3], x0.dtype)
        noise = torch.randn_like(x0) if noise is None else noise
        noisy = sigmas_b * noise + (1.0 - sigmas_b) * x0
        noisy = torch.cat([history.to(noisy.dtype), noisy[:, CONTEXT:]], dim=1)
        ts = ts.clone()
        ts[:, :CONTEXT] = 0
        return noisy, ts, sigmas_b, noise

    # ------------------------------------------------------------------
    def _kl_grad_with_history(self, noisy, ts, sigmas_b, x0_det, cond, cond_cfg, target_mask):
        cond_fake = dict(cond)
        if self.fake_score_causal:
            self._add_temporal_attention_mask(cond_fake, noisy.shape)
        with self.get_autocast_context():
            fake_out, _, _ = self.fake_score_wrapper(noisy.to(self.model_dtype), ts, **cond_fake)
        fake_x0 = noisy.float() + (-sigmas_b).float() * fake_out[0].float()

        noisy_cfg = torch.cat([noisy, noisy], dim=0)
        ts_cfg = torch.cat([ts, ts], dim=0)
        sigmas_cfg = torch.cat([sigmas_b, sigmas_b], dim=0)
        cond_real = {k: v for k, v in cond_cfg.items() if k != "temporal_attention_mask"}
        if self.real_score_causal:
            self._add_temporal_attention_mask(cond_real, noisy.shape)
        with self.get_autocast_context():
            real_out, _, _ = self.real_score_wrapper(noisy_cfg.to(self.model_dtype), ts_cfg, **cond_real)
        real_x0_cfg = noisy_cfg.float() + (-sigmas_cfg).float() * real_out[0].float()
        real_uncond, real_cond = real_x0_cfg.chunk(2, dim=0)
        real_x0 = real_cond + (real_cond - real_uncond) * self.real_guidance_scale

        grad = fake_x0 - real_x0
        mask = target_mask.to(grad.dtype)
        while mask.ndim < grad.ndim:
            mask = mask.unsqueeze(-1)
        # Per-sample DMD normaliser (eq. 8), over the target frames only.
        p_real = (x0_det.float() - real_x0).abs() * mask
        normalizer = (p_real.flatten(1).sum(1) / mask.expand_as(p_real).flatten(1).sum(1).clamp(min=1.0))
        grad = grad / normalizer.clamp(min=1e-8).view(-1, *([1] * (grad.ndim - 1)))
        return torch.nan_to_num(grad).to(dtype=x0_det.dtype)

    def _disc_logits(self, noisy, ts, cond):
        """Discriminator logit per sample from the fake score's block features."""
        cond_fake = dict(cond)
        if self.fake_score_causal:
            self._add_temporal_attention_mask(cond_fake, noisy.shape)
        self._capture = True
        try:
            with self.get_autocast_context():
                self.fake_score_wrapper(noisy.to(self.model_dtype), ts, **cond_fake)
        finally:
            self._capture = False
        h = self._features.pop("h")                      # ((b t v), tokens, c)
        b, t, v = noisy.shape[:3]
        h = h.float().unflatten(0, (b, t, v))[:, CONTEXT:]   # target frames only
        return self.disc_head(h.mean(dim=(1, 2, 3))).squeeze(-1)

    # ------------------------------------------------------------------
    def train_step(self, batch: dict, global_step: int):
        self.real_score_wrapper.eval()
        t0 = time.time()
        view_count = batch["vae_images"].shape[2]
        frames = batch["vae_images"].shape[1]
        need = CONTEXT + STRIDE * (self.context_windows + 1)
        if frames < need:
            raise ValueError(f"clip has {frames} frames, context forcing needs {need}")
        latent_shape = self._get_latent_shape(batch, WINDOW, view_count)
        start = STRIDE * self.context_windows                # first frame of the trained window

        gt = self._vae_encode(batch["vae_images"][:, :start + WINDOW])
        history = gt[:, :CONTEXT]
        forced = _broadcast_scalar(float(torch.rand(()) < self.context_forcing_prob), self.device)
        if forced:
            for w in range(self.context_windows):
                cond_w = self._conditions(self._window_batch(batch, STRIDE * w), latent_shape, False)
                history = self._rollout_window(latent_shape, cond_w, history)[:, -CONTEXT:]
        else:
            history = gt[:, start:start + CONTEXT]
        history = history.detach()
        gt_window = gt[:, start:start + WINDOW]

        wbatch = self._window_batch(batch, start)
        cond = self._conditions(wbatch, latent_shape, False)
        cond_cfg = self._conditions(wbatch, latent_shape, True)
        target_mask = torch.ones(latent_shape[:3], dtype=torch.bool, device=self.device)
        target_mask[:, :CONTEXT] = False

        log = {"gen_loss": self._last_gen_loss, "dmd_grad_norm": self._last_dmd_grad_norm,
               "adv_gen": self._last_adv, "context_forced": forced}
        x0_sample = None
        if global_step % self.dfake_gen_update_ratio == 0:
            log.update(self._generator_step_cf(latent_shape, cond, cond_cfg, history, target_mask))
            if global_step >= self.ema_start_step:
                self._ema_update()
        log.update(self._critic_step_cf(latent_shape, cond, history, gt_window, target_mask))
        self.step_duration += time.time() - t0
        self.loss_report_list.append({"loss": log.get("critic_loss", 0.0), **log})

    def _generator_step_cf(self, latent_shape, cond, cond_cfg, history, target_mask):
        x0_pred = self._run_generator(latent_shape, cond, image_latents=history,
                                      reference_frame_count=CONTEXT, grad_on_last=True)
        with torch.no_grad():
            x0_det = x0_pred.detach()
            noisy, ts, sigmas_b, _ = self._noised_with_clean_history(x0_det, history)
            grad = self._kl_grad_with_history(noisy, ts, sigmas_b, x0_det, cond, cond_cfg, target_mask)
        target = (x0_det.double() - grad.double()).detach()
        mask = target_mask.to(torch.float64)
        while mask.ndim < x0_pred.ndim:
            mask = mask.unsqueeze(-1)
        diff = (x0_pred.double() - target) * mask
        loss = 0.5 * diff.square().sum() / mask.expand_as(diff).sum().clamp(min=1.0)
        adv = torch.zeros((), device=self.device)
        if self.gan_generator_weight > 0:
            noisy_g, ts_g, _, _ = self._noised_with_clean_history(x0_pred, history)
            adv = F.softplus(-self._disc_logits(noisy_g, ts_g, cond)).mean()
            loss = loss + self.gan_generator_weight * adv.double()

        use_scaler = self.training_config.get("enable_grad_scaler", False)
        if use_scaler:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        if "max_norm_for_grad_clip" in self.training_config:
            self.model_wrapper.clip_grad_norm_(self.training_config["max_norm_for_grad_clip"])
        if use_scaler:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        # The adversarial term back-propagated through the fake score; those
        # gradients belong to the generator update only.
        self.critic_optimizer.zero_grad()
        if self.use_gan:
            self.disc_optimizer.zero_grad()
        self._last_gen_loss = float(loss.item())
        self._last_dmd_grad_norm = float(grad.abs().mean().item())
        self._last_adv = float(adv.item())
        return {"gen_loss": self._last_gen_loss, "dmd_grad_norm": self._last_dmd_grad_norm,
                "adv_gen": self._last_adv}

    def _critic_step_cf(self, latent_shape, cond, history, gt_window, target_mask):
        self.fake_score_wrapper.train()
        with torch.no_grad():
            x0 = self._run_generator(latent_shape, cond, image_latents=history,
                                     reference_frame_count=CONTEXT, grad_on_last=False).detach()
        noisy, ts, sigmas_b, noise = self._noised_with_clean_history(x0, history)
        cond_critic = dict(cond)
        if self.fake_score_causal:
            self._add_temporal_attention_mask(cond_critic, latent_shape)
        with self.get_autocast_context():
            out, _, _ = self.fake_score_wrapper(noisy.to(self.model_dtype), ts, **cond_critic)
        target_v = noise - x0
        mask = target_mask.to(torch.float32)
        while mask.ndim < x0.ndim:
            mask = mask.unsqueeze(-1)
        err = (out[0].float() - target_v.float()).square() * mask
        critic_loss = err.sum() / mask.expand_as(err).sum().clamp(min=1.0)
        report = {"critic_loss": float(critic_loss.item())}

        total = critic_loss
        if self.gan_discriminator_weight > 0:
            # Real: ground-truth window behind its ground-truth history.
            real_noisy, real_ts, _, _ = self._noised_with_clean_history(
                gt_window, gt_window[:, :CONTEXT])
            real_logit = self._disc_logits(real_noisy, real_ts, cond)
            fake_logit = self._disc_logits(noisy.detach(), ts, cond)
            disc_loss = F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()
            total = total + self.gan_discriminator_weight * disc_loss
            report.update(disc_loss=float(disc_loss.item()),
                          disc_real_acc=float((real_logit > 0).float().mean().item()),
                          disc_fake_acc=float((fake_logit < 0).float().mean().item()))

        use_scaler = self.training_config.get("enable_grad_scaler", False)
        loss_scale = float(self.grad_scaler.get_scale()) if use_scaler else 1.0
        if use_scaler:
            self.grad_scaler.scale(total).backward()
            self.grad_scaler.unscale_(self.critic_optimizer)
        else:
            total.backward()
        self.fake_score_wrapper.clip_grad_norm_(self.max_norm_for_critic_grad_clip)
        if use_scaler:
            self.grad_scaler.step(self.critic_optimizer)
            self.grad_scaler.update()
        else:
            self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        if self.gan_discriminator_weight > 0:
            self._step_disc_head(loss_scale)
        return report

    def _step_disc_head(self, loss_scale):
        params = [p for p in self.disc_head.parameters() if p.grad is not None]
        # The head is outside the critic optimizer the shared scaler handles;
        # undo the loss scale that was in force for this backward by hand.
        for p in params:
            p.grad.div_(loss_scale)
        if torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
            for p in params:
                torch.distributed.all_reduce(p.grad)
                p.grad.div_(world)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if all(torch.isfinite(p.grad).all() for p in params):
            self.disc_optimizer.step()
        self.disc_optimizer.zero_grad()

    def save_checkpoint(self, output_path: str, steps: int, *args, **kwargs):
        result = super().save_checkpoint(output_path, steps, *args, **kwargs)
        if self.use_gan and self.should_save:
            target = os.environ.get("CKPT_OUT_DIR") or os.path.join(output_path, "checkpoints")
            os.makedirs(target, exist_ok=True)
            torch.save(self.disc_head.state_dict(), os.path.join(target, f"{steps}_disc_head.pth"))
        return result
