"""v6: optional pull toward the real footage on top of v5 (all earlier pipeline files are untouched).

Why: on the 149 eval clips the old CF-DMD student (run A, step 1585) beats B0 on FID/FVD but is further from
the real future (PSNR -0.42 / -0.48 dB, worse than B0 on 132/149 clips in window 1) and over-sharpened
(Laplacian variance 371 vs GT 329, B0 352).  Nothing in CF-DMD ties the student to the real clip: DMD only asks
for samples the teacher finds likely.  Two optional knobs, both OFF by default (v6 with defaults == v5):

1. ``gt_regression_weight`` (default 0.0): on every student update, also roll out the FIRST window from the
   clip's real 3-frame history with the full 4-step sampler (the deployed one, no random exit) and add
   ``w * mean((x0 - gt)^2)`` over the 16 target latents.  Only this window has a real history, so only here
   is the real future the right target.  It is back-propagated in its own pass before the DMD/GAN pass, so
   both gradients land in the same student step and peak memory stays that of one graph.  Caveat: the future
   is not deterministic, so a large weight pulls toward the average future (blur); keep it small (~0.1)
   and watch the ``gt_reg`` log value and sharpness.
2. Teacher guidance needs no code: ``real_guidance_scale`` already sets ``x_T = x_c + s * (x_c - x_u)``
   (s = 3 means an effective guidance of 4).  A lower s (e.g. 1.0, effective 2) is a config change.

``gt_reg`` is reported from the critic step, which runs every iteration (the trainer averages every key over
all logged rows; a key that only student steps return crashed job 703667).
"""
import torch

from .pipeline import CONTEXT, WINDOW
from .pipeline_v5 import PanoContextForcedDMDv5


class PanoContextForcedDMDv6(PanoContextForcedDMDv5):
    def __init__(self, *args, gt_regression_weight=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gt_regression_weight = float(gt_regression_weight)
        self._stash = {}
        self._last_gt_reg = 0.0
        if self.should_save:
            print(f"V6 gt_regression_weight={self.gt_regression_weight} "
                  f"real_guidance_scale={self.real_guidance_scale}", flush=True)

    # Capture the clip's real latents and the first window's conditions from the parent train_step.
    def train_step(self, batch, global_step):
        self._stash = {}
        return super().train_step(batch, global_step)

    def _vae_encode(self, vae_images):
        out = super()._vae_encode(vae_images)
        self._stash.setdefault("gt", out)
        return out

    @torch.no_grad()
    def _rollout_window(self, latent_shape, cond, history):
        if "cond0" not in self._stash:
            self._stash["cond0"], self._stash["latent_shape"] = cond, latent_shape
        return super()._rollout_window(latent_shape, cond, history)

    # ------------------------------------------------------------------
    def _gt_regression_backward(self):
        """Backward w * MSE(full 4-step first window from real history, real future); returns the MSE."""
        gt, cond0 = self._stash.get("gt"), self._stash.get("cond0")
        if gt is None or cond0 is None or gt.shape[1] < WINDOW:
            return None
        latent_shape = self._stash["latent_shape"]
        saved = self.random_exit_step
        self.random_exit_step = False          # deployed sampler: every rank exits at the last step
        try:
            x0 = self._run_generator(latent_shape, cond0, image_latents=gt[:, :CONTEXT],
                                     reference_frame_count=CONTEXT, grad_on_last=True)
        finally:
            self.random_exit_step = saved
        target = gt[:, CONTEXT:WINDOW].detach()
        mse = (x0[:, CONTEXT:].float() - target.float()).square().mean()
        loss = self.gt_regression_weight * mse
        if self.training_config.get("enable_grad_scaler", False):
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        return float(mse.item())

    def _generator_step_cf(self, latent_shape, cond, cond_cfg, history, target_mask):
        if self.gt_regression_weight > 0:
            value = self._gt_regression_backward()
            if value is not None:
                self._last_gt_reg = value
        # v5 back-propagates DMD + GAN into the same (not yet zeroed) student gradients, then unscales,
        # clips and steps once, so the regression term is part of this student update.
        return super()._generator_step_cf(latent_shape, cond, cond_cfg, history, target_mask)

    def _critic_step_cf(self, latent_shape, cond, history, gt_window, target_mask):
        report = super()._critic_step_cf(latent_shape, cond, history, gt_window, target_mask)
        report["gt_reg"] = float(self._last_gt_reg)
        return report
