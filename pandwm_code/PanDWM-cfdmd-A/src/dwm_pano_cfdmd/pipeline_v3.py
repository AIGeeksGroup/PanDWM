"""Lower-memory critic step for the GAN variant (pipeline.py / pipeline_v2.py are untouched).

Job 700295 (HPC3, 4x H100 80GB): the generator step passed (the v2 FSDP fix works) but the critic step ran
out of memory in `total.backward()` (75.9 GiB allocated, 1 GiB more requested).  The original step keeps three
fake-score autograd graphs alive at once: the flow-matching forward, plus the discriminator forwards on the
real and the fake window, then back-propagates `critic_loss + w * disc_loss` together.

v3 back-propagates the two terms one after the other.  Gradients accumulate in the same parameters, so the
update is mathematically identical (sum of the two gradients), but the flow-matching graph is freed before the
discriminator forwards are built: peak = max(graph) instead of sum(graphs).

Because the second backward has no fake-score OUTPUT in its loss, FSDP's pre-backward hook would not fire
again (the failure fixed in pipeline_v2.py).  The discriminator forwards therefore run with the v2
`_touch_fake_output` switch on (adds 0 * output; values and gradients unchanged).
"""
import torch
import torch.nn.functional as F

from dwm_pano_cfdmd.pipeline import CONTEXT
from dwm_pano_cfdmd.pipeline_v2 import PanoContextForcedDMDv2


class PanoContextForcedDMDv3(PanoContextForcedDMDv2):
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
        with self.get_autocast_context():
            out, _, _ = self.fake_score_wrapper(noisy.to(self.model_dtype), ts, **cond_critic)
        target_v = noise - x0
        mask = target_mask.to(torch.float32)
        while mask.ndim < x0.ndim:
            mask = mask.unsqueeze(-1)
        err = (out[0].float() - target_v.float()).square() * mask
        critic_loss = err.sum() / mask.expand_as(err).sum().clamp(min=1.0)
        report = {"critic_loss": float(critic_loss.item())}

        use_scaler = self.training_config.get("enable_grad_scaler", False)
        loss_scale = float(self.grad_scaler.get_scale()) if use_scaler else 1.0

        # (1) flow-matching loss alone; its graph is released before the discriminator forwards.
        if use_scaler:
            self.grad_scaler.scale(critic_loss).backward()
        else:
            critic_loss.backward()
        del out, err, critic_loss

        # (2) discriminator on the real window and on the generator's noised sample.
        real_noisy, real_ts, _, _ = self._noised_with_clean_history(gt_window, gt_window[:, :CONTEXT])
        self._touch_fake_output = True
        try:
            real_logit = self._disc_logits(real_noisy, real_ts, cond)
            fake_logit = self._disc_logits(noisy.detach(), ts, cond)
        finally:
            self._touch_fake_output = False
        disc_loss = F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()
        report.update(disc_loss=float(disc_loss.item()),
                      disc_real_acc=float((real_logit > 0).float().mean().item()),
                      disc_fake_acc=float((fake_logit < 0).float().mean().item()))
        weighted = self.gan_discriminator_weight * disc_loss
        if use_scaler:
            self.grad_scaler.scale(weighted).backward()
        else:
            weighted.backward()
        del real_logit, fake_logit, disc_loss, weighted

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
