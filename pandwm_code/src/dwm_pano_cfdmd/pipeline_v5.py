"""Context-Forced DMD with a history-blind discriminator and per-rank noise.

Changes relative to the resumable pipeline:
- Every discriminator input (real and fake, generator and critic step) replaces the 3 history slots with pure
  noise at the first 4-step timestep, so the discriminator only sees the 16 target frames and cannot separate
  real from generated clips by the history alone. DMD scores and the critic flow-matching loss still use the
  clean student history.
- After the models and discriminator head are built, torch / CUDA / the pipeline generator are reseeded with
  seed + 1009 * rank so ranks do not share rollout noise and timesteps.
- Student weights of every `milestone_steps`-th committed checkpoint are hard-linked into <run>/milestones so
  they survive checkpoint rotation.
- EMA is skipped when the GradScaler skipped the student step, and dmd_grad_norm averages over the target
  frames only. Skipped generator steps are counted cumulatively and stored as log/v5_counters_<step>.json.

Critic step (three backward passes, one fake-score graph alive at a time): flow matching on the clean student
history; softplus(D(noise history, noised student target)); softplus(-D(noise history, noised GT target)).
"""
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint_io import atomic_json
from .pipeline import CONTEXT
from .pipeline_resumable import PanoContextForcedDMDResumable


class PanoContextForcedDMDv5(PanoContextForcedDMDResumable):
    def __init__(self, *args, disc_history="noise", per_rank_seed=True, milestone_steps=250, **kwargs):
        super().__init__(*args, **kwargs)
        if disc_history not in ("noise", "clean"):
            raise ValueError(f"disc_history must be 'noise' or 'clean', got {disc_history}")
        self.disc_history = disc_history
        self._milestone_steps = int(milestone_steps)
        self._cur_step = None
        self._skipped_gen_steps = 0
        resume = kwargs.get("resume_from")
        if resume is not None:
            counters = self._output / "log" / f"v5_counters_{resume}.json"
            if counters.exists():
                import json
                self._skipped_gen_steps = int(json.loads(counters.read_text())["gen_steps_skipped"])
                note = "restored"
            else:
                note = "missing (checkpoint predates the counter file), starting at 0"
            if self.should_save:
                print(f"V5_COUNTER gen_steps_skipped={self._skipped_gen_steps} at resume {resume}: {note}",
                      flush=True)
        if per_rank_seed:
            seed = int(kwargs.get("seed", 0)) + 1009 * self._rank
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            self.generator.manual_seed(seed)
        if self.should_save:
            print(f"V5 disc_history={disc_history} per_rank_seed={per_rank_seed} "
                  f"milestone_steps={self._milestone_steps}", flush=True)

    def train_step(self, batch, global_step):
        self._cur_step = global_step
        return super().train_step(batch, global_step)

    # ------------------------------------------------------------------
    def _disc_input(self, x0):
        """Noise the 16 targets at one shared t (as in training); history slots become pure noise."""
        if self.disc_history == "clean":
            raise RuntimeError("clean history must go through _noised_with_clean_history")
        ts, sigmas_b = self._sample_dmd_timestep(x0.shape[:3], x0.dtype)
        noisy = sigmas_b * torch.randn_like(x0) + (1.0 - sigmas_b) * x0
        sigma0 = self.denoising_sigmas[0].to(dtype=noisy.dtype)
        hist = sigma0 * torch.randn_like(noisy[:, :CONTEXT])
        noisy = torch.cat([hist, noisy[:, CONTEXT:]], dim=1)
        ts = ts.clone()
        ts[:, :CONTEXT] = float(self.denoising_step_list[0].item())
        return noisy, ts

    def _disc_sample(self, x0, history):
        if self.disc_history == "clean":
            noisy, ts, _, _ = self._noised_with_clean_history(x0, history)
            return noisy, ts
        return self._disc_input(x0)

    # ------------------------------------------------------------------
    def _generator_step_cf(self, latent_shape, cond, cond_cfg, history, target_mask):
        # As in the base pipeline, but with a history-blind adversarial input.
        self._touch_fake_output = True
        try:
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
                noisy_g, ts_g = self._disc_sample(x0_pred, history)
                adv = F.softplus(-self._disc_logits(noisy_g, ts_g, cond)).mean()
                loss = loss + self.gan_generator_weight * adv.double()
        finally:
            self._touch_fake_output = False

        use_scaler = self.training_config.get("enable_grad_scaler", False)
        scale_before = float(self.grad_scaler.get_scale()) if use_scaler else 1.0
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
            # update() lowers the scale exactly when the step was skipped for inf/nan.
            self._last_gen_step_ok = float(self.grad_scaler.get_scale()) >= scale_before
        else:
            self.optimizer.step()
            self._last_gen_step_ok = True
        self.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        if self.use_gan:
            self.disc_optimizer.zero_grad()
        if not self._last_gen_step_ok:
            self._skipped_gen_steps += 1
            if self.should_save:
                print(f"GEN_STEP_SKIPPED global_step={self._cur_step} total={self._skipped_gen_steps}",
                      flush=True)
        self._last_gen_loss = float(loss.item())
        self._last_dmd_grad_norm = float((grad * mask.to(grad.dtype)).abs().sum().item()
                                         / mask.expand_as(grad).sum().clamp(min=1.0).item())
        self._last_adv = float(adv.item())
        return {"gen_loss": self._last_gen_loss, "dmd_grad_norm": self._last_dmd_grad_norm,
                "adv_gen": self._last_adv}

    def _ema_update(self):
        if getattr(self, "_last_gen_step_ok", True):
            super()._ema_update()

    # ------------------------------------------------------------------
    def _critic_step_cf(self, latent_shape, cond, history, gt_window, target_mask):
        if self.gan_discriminator_weight <= 0 or self.disc_history == "clean":
            return super()._critic_step_cf(latent_shape, cond, history, gt_window, target_mask)

        self.fake_score_wrapper.train()
        with torch.no_grad():
            x0 = self._run_generator(latent_shape, cond, image_latents=history,
                                     reference_frame_count=CONTEXT, grad_on_last=False).detach()
        use_scaler = self.training_config.get("enable_grad_scaler", False)
        loss_scale = float(self.grad_scaler.get_scale()) if use_scaler else 1.0
        w = self.gan_discriminator_weight

        def backward(loss):
            if use_scaler:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

        # 1) flow matching on the clean student history (unchanged objective).
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
        backward(critic_loss)
        del out, err, critic_loss

        # 2) and 3) history-blind discriminator; the touch switch keeps FSDP's backward hooks firing.
        self._touch_fake_output = True
        try:
            fake_noisy, fake_ts = self._disc_input(x0)
            fake_logit = self._disc_logits(fake_noisy, fake_ts, cond)
            fake_term = F.softplus(fake_logit).mean()
            report["disc_fake_acc"] = float((fake_logit < 0).float().mean().item())
            fake_value = float(fake_term.item())
            backward(w * fake_term)
            del fake_logit, fake_term

            real_noisy, real_ts = self._disc_input(gt_window)
            real_logit = self._disc_logits(real_noisy, real_ts, cond)
            real_term = F.softplus(-real_logit).mean()
            report.update(disc_loss=float(real_term.item()) + fake_value,
                          disc_real_acc=float((real_logit > 0).float().mean().item()))
            backward(w * real_term)
            del real_logit, real_term
        finally:
            self._touch_fake_output = False

        if use_scaler:
            self.grad_scaler.unscale_(self.critic_optimizer)
        self.fake_score_wrapper.clip_grad_norm_(self.max_norm_for_critic_grad_clip)
        if use_scaler:
            self.grad_scaler.step(self.critic_optimizer)
            self.grad_scaler.update()
        else:
            self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        self._step_disc_head(loss_scale)
        return report

    # ------------------------------------------------------------------
    def save_checkpoint(self, output_path, steps, upload=False, keep_last_n=3):
        already = steps == self._last_saved
        super().save_checkpoint(output_path, steps, upload=upload, keep_last_n=keep_last_n)
        if already or not self.should_save:
            return
        atomic_json({"step": steps, "gen_steps_skipped": self._skipped_gen_steps},
                    Path(output_path) / "log" / f"v5_counters_{steps}.json")
        if self._milestone_steps <= 0:
            return
        if steps % self._milestone_steps != 0 and steps < self._target_steps:
            return
        root = Path(output_path)
        keep = root / "milestones"
        keep.mkdir(exist_ok=True)
        for name in (f"{steps}.pth", f"{steps}_ema.pth", f"{steps}_critic.pth",
                     f"{steps}_disc_head.pth", f"{steps}.complete.json"):
            source = root / "checkpoints" / name
            if source.exists():
                target = keep / name
                if target.exists():
                    target.unlink()
                os.link(source, target)
        print(f"MILESTONE_LINKED step={steps}", flush=True)
