"""Lower-memory critic step for the GAN variant of Context-Forced DMD.

The discriminator forward on the generated sample is the same call as the flow-matching forward of the critic
step (same noisy latents, timesteps and causal mask), so the fake logit is taken from block features of that
single forward. Two backward passes are used: (1) critic loss plus the fake-sample discriminator term, (2) the
real-window discriminator term. Their summed gradient equals that of the combined loss, while peak memory is
one graph instead of several.
"""
import torch
import torch.nn.functional as F

from dwm_pano.student.pipeline import CONTEXT
from dwm_pano.student.pipeline_v2 import PanoContextForcedDMDv2


class PanoContextForcedDMDv4(PanoContextForcedDMDv2):
    def _logit_from_features(self, h, shape):
        b, t, v = shape[:3]
        h = h.float().unflatten(0, (b, t, v))[:, CONTEXT:]          # target frames only
        return self.disc_head(h.mean(dim=(1, 2, 3))).squeeze(-1)

    def _critic_step_cf(self, latent_shape, cond, history, gt_window, target_mask):
        if self.gan_discriminator_weight <= 0:
            return super()._critic_step_cf(latent_shape, cond, history, gt_window, target_mask)

        self.fake_score_wrapper.train()
        with torch.no_grad():
            x0 = self._run_generator(latent_shape, cond, image_latents=history,
                                     reference_frame_count=CONTEXT, grad_on_last=False).detach()
        noisy, ts, sigmas_b, noise = self._noised_with_clean_history(x0, history)
        cond_critic = dict(cond)
        if self.fake_score_causal:
            self._add_temporal_attention_mask(cond_critic, latent_shape)

        # One fake-score forward serves both the flow-matching loss and the fake discriminator logit.
        self._capture = True
        try:
            with self.get_autocast_context():
                out, _, _ = self.fake_score_wrapper(noisy.to(self.model_dtype), ts, **cond_critic)
        finally:
            self._capture = False
        h_fake = self._features.pop("h")
        fake_logit = self._logit_from_features(h_fake, noisy.shape)

        target_v = noise - x0
        mask = target_mask.to(torch.float32)
        while mask.ndim < x0.ndim:
            mask = mask.unsqueeze(-1)
        err = (out[0].float() - target_v.float()).square() * mask
        critic_loss = err.sum() / mask.expand_as(err).sum().clamp(min=1.0)
        report = {"critic_loss": float(critic_loss.item())}

        use_scaler = self.training_config.get("enable_grad_scaler", False)
        loss_scale = float(self.grad_scaler.get_scale()) if use_scaler else 1.0
        w = self.gan_discriminator_weight

        loss1 = critic_loss + w * F.softplus(fake_logit).mean()
        report["disc_fake_acc"] = float((fake_logit < 0).float().mean().item())
        fake_term = float(F.softplus(fake_logit).mean().item())
        if use_scaler:
            self.grad_scaler.scale(loss1).backward()
        else:
            loss1.backward()
        del out, err, critic_loss, loss1, h_fake, fake_logit

        # Real window behind its ground-truth history; the touch switch keeps FSDP's backward hooks firing.
        real_noisy, real_ts, _, _ = self._noised_with_clean_history(gt_window, gt_window[:, :CONTEXT])
        self._touch_fake_output = True
        try:
            real_logit = self._disc_logits(real_noisy, real_ts, cond)
        finally:
            self._touch_fake_output = False
        real_term = F.softplus(-real_logit).mean()
        report.update(disc_loss=float(real_term.item()) + fake_term,
                      disc_real_acc=float((real_logit > 0).float().mean().item()))
        loss2 = w * real_term
        if use_scaler:
            self.grad_scaler.scale(loss2).backward()
        else:
            loss2.backward()
        del real_logit, real_term, loss2

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
