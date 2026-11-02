"""v8.3 = v8.2 with a reworked frame-to-frame change match (2026-10-12).

Why (v8.2 diagnosis, eval_v7/V82_EVAL_HPC3_20261011.md section 6):
  * the remaining flicker is static-scene "boiling": in the third of clips with the least real motion the generated
    frame change is 1.75x GT (VAE round trip 1.10x); scenes with motion are at ~1.0x;
  * the step between windows (last history frame -> first generated frame) carries the same excess but v8.2's term
    never sees it: latent_motion(x[:, CONTEXT:]) covers the 15 steps inside the 16 generated frames only;
  * v8.2 matched raw latent L2 change, which falls when detail is removed: from v8.1 to v8.2 high-frequency change fell
    more than low-frequency change and per-clip sharpness loss correlates with change loss (+0.45);
  * the term was on/off: zero when r <= 1, otherwise always clipped to 0.35 x DMD (its uncapped gradient was 1.7-10x
    DMD), and it only covered the first window (real history), not windows built on the student's own history.

What changes (everything else is v8.2):
  1. Seam: the change sequence starts at the last history frame, so the history -> first generated step counts.
  2. Blur-proof split: each latent frame is split into a low-pass part (average over `mm3_pool` x `mm3_pool` latent
     pixels) and the high-frequency residual.
       r_hf = (HF frame change / HF energy)_student / (same)_real      -- removing detail does not lower it
       r_lf = (LF frame change)_student / (LF frame change)_real
     loss = w_hf * relu(log r_hf)^2 + w_lf * [relu(log r_lf)^2 + relu(log floor - log r_lf)^2]
     The LF part is two-sided (below `mm3_lf_floor` x the real change counts too), so the video is not pushed static.
  3. Smooth push: squared log ratios grow with the excess; the 0.35 x DMD cap (CapGrad) stays as a safety bound.
  4. Rollout window: when the student's main pass exited at the final step (its x0 is the deployed 4-step sample),
     the same term is applied to the trained window, whose history is the student's own rollout (Self-Forcing's
     train-on-own-history point), against the real window's change; earlier exits are skipped (blurry by design).
Report keys: mm3_hf_ratio / mm3_lf_ratio (first window, before the update), mm3_loss, mm3_cap_frac, mm3_dmd_ratio,
mm3_roll_hf_ratio / mm3_roll_lf_ratio / mm3_roll_on (rollout term), mm_weight.  The v8.2 keys keep their names.
"""
import math

import torch
import torch.nn.functional as F

from .pipeline import CONTEXT, STRIDE, WINDOW
from .pipeline_v7 import CapGrad
from .pipeline_v8_2 import PanoContextForcedDMDv8_2

_EPS = 1e-12


def split_lf_hf(x, pool):
    """x: [B, T, V, C, H, W] -> (low-pass [B, T, V, C, H/pool, W/pool], high-frequency residual like x)."""
    b, t, v, c, h, w = x.shape
    flat = x.float().flatten(0, 2)
    lf = F.avg_pool2d(flat, pool)
    hf = flat - F.interpolate(lf, size=(h, w), mode="nearest")
    return lf.unflatten(0, (b, t, v)), hf.unflatten(0, (b, t, v))


def change_stats(seq, pool):
    """seq: [B, 1 + T, ...] = last history frame followed by T generated frames.

    Returns per sample (lf_change, hf_relative_change): mean squared frame-to-frame change over the T steps (the
    first one is the seam step) of the low-pass part, and of the high-frequency part divided by the mean squared
    high-frequency energy of the generated frames."""
    lf, hf = split_lf_hf(seq, pool)
    lf_change = (lf[:, 1:] - lf[:, :-1]).square().flatten(1).mean(1)
    hf_change = (hf[:, 1:] - hf[:, :-1]).square().flatten(1).mean(1)
    hf_energy = hf[:, 1:].square().flatten(1).mean(1)
    return lf_change, hf_change / hf_energy.clamp(min=_EPS)


class PanoContextForcedDMDv8_3(PanoContextForcedDMDv8_2):
    def __init__(self, *args, mm3_pool=4, mm3_w_lf=1.0, mm3_w_hf=1.0, mm3_lf_floor=0.9, mm3_rollout=True,
                 **kwargs):
        self.mm3_pool, self.mm3_w_lf, self.mm3_w_hf = int(mm3_pool), float(mm3_w_lf), float(mm3_w_hf)
        self.mm3_lf_floor, self.mm3_rollout = float(mm3_lf_floor), bool(mm3_rollout)
        self._last_mm3 = {"mm3_hf_ratio": 1.0, "mm3_lf_ratio": 1.0, "mm3_loss": 0.0, "mm3_cap_frac": 0.0,
                          "mm3_dmd_ratio": 0.0, "mm3_roll_hf_ratio": 1.0, "mm3_roll_lf_ratio": 1.0,
                          "mm3_roll_on": 0.0}
        super().__init__(*args, **kwargs)
        if self.should_save:
            print(f"V8_3 mm3_pool={self.mm3_pool} w_lf={self.mm3_w_lf} w_hf={self.mm3_w_hf} "
                  f"lf_floor={self.mm3_lf_floor} rollout={self.mm3_rollout} (seam step included, smooth log-ratio "
                  f"penalty, cap {self.motion_match_cap} x DMD)", flush=True)

    def _motion_match3(self, x_gen, hist_last, real_seq, cap):
        """Weight-free v8.3 term.  x_gen: student frames [B, T, ...] (gradient capped); hist_last: [B, 1, ...] the
        history frame before them (no gradient); real_seq: [B, 1 + T, ...] the real last history frame + future."""
        stats = {}
        x = CapGrad.apply(x_gen, cap, stats)
        fake_lf, fake_hf = change_stats(torch.cat([hist_last.detach().to(x.dtype), x], dim=1), self.mm3_pool)
        with torch.no_grad():
            real_lf, real_hf = change_stats(real_seq, self.mm3_pool)
        log_lf = torch.log(fake_lf.clamp(min=_EPS)) - torch.log(real_lf.clamp(min=_EPS))
        log_hf = torch.log(fake_hf.clamp(min=_EPS)) - torch.log(real_hf.clamp(min=_EPS))
        floor = math.log(self.mm3_lf_floor)
        loss = (self.mm3_w_hf * F.relu(log_hf).square()
                + self.mm3_w_lf * (F.relu(log_lf).square() + F.relu(floor - log_lf).square())).mean()
        return loss, log_lf.detach().exp(), log_hf.detach().exp(), stats

    # A copy of PanoContextForcedDMDv8_2._generator_step_cf with the v8.3 term ("v8.3") in place of v8.2's
    # _motion_match on the first window, plus the rollout-window term; the rest is unchanged.
    def _generator_step_cf(self, latent_shape, cond, cond_cfg, history, target_mask):
        use_scaler = self.training_config.get("enable_grad_scaler", False)
        scale = float(self.grad_scaler.get_scale()) if use_scaler else 1.0
        w_adv = self._adv_weight()
        self._last_gan_weight = w_adv
        adv_pairs = self._adv_pairs_this_step()
        self._stash["cond_main"] = cond

        x0_pred = self._run_generator(latent_shape, cond, image_latents=history, reference_frame_count=CONTEXT,
                                      grad_on_last=True)
        exit_main = self._last_exit_idx
        with torch.no_grad():
            x0_det = x0_pred.detach()
            noisy, ts, sigmas_b, _ = self._noised_with_clean_history(x0_det, history)
            grad = self._kl_grad_with_history(noisy, ts, sigmas_b, x0_det, cond, cond_cfg, target_mask)
        target = (x0_det.double() - grad.double()).detach()
        mask = target_mask.to(torch.float64)
        while mask.ndim < x0_pred.ndim:
            mask = mask.unsqueeze(-1)
        diff = (x0_pred.double() - target) * mask
        count = mask.expand_as(diff).sum().clamp(min=1.0)
        loss = 0.5 * diff.square().sum() / count
        dmd_norm = float((grad.double() * mask).norm() / count)
        cap = None if self.gan_ratio_cap is None else self.gan_ratio_cap * dmd_norm * scale
        cap_mm = None if self.motion_match_cap is None else self.motion_match_cap * dmd_norm * scale   # v8.3
        adv = torch.zeros((), device=self.device)
        stats = {}
        if w_adv > 0 and "rollout" in adv_pairs:
            adv, touch = self._adv_term(x0_pred, lambda xs: self._rollout_inputs(xs, exit_main), cap, stats)
            loss = loss + w_adv * adv.double() + touch.double()
        w_mm = self._motion_weight()                                                   # v8.3
        st = self._stash
        roll_stats = None
        if w_mm > 0:
            self._last_mm3["mm3_roll_on"] = 0.0
        if (w_mm > 0 and self.mm3_rollout and exit_main == len(self.denoising_step_list) - 1 and "gt" in st):
            start = STRIDE * self.context_windows                                      # first frame of this window
            real_seq = st["gt"][:, start + CONTEXT - 1:start + WINDOW]
            if real_seq.shape[1] == WINDOW - CONTEXT + 1:
                roll, r_lf, r_hf, roll_stats = self._motion_match3(x0_pred[:, CONTEXT:], history[:, -1:],
                                                                   real_seq, cap_mm)
                loss = loss + w_mm * roll.double()
                self._last_mm3.update(mm3_roll_lf_ratio=float(r_lf.mean()), mm3_roll_hf_ratio=float(r_hf.mean()),
                                      mm3_roll_on=1.0)
        self._backward(loss)
        loss_value = float(loss.item())
        del x0_pred, diff, loss, target, noisy, x0_det
        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.use_gan:
            self.disc_optimizer.zero_grad(set_to_none=True)

        use_first = w_adv > 0 and "first" in adv_pairs
        mm_stats = None
        if (use_first or self.gt_regression_weight > 0 or w_mm > 0) and "cond0" in st and "gt" in st:
            gt = st["gt"]
            saved = self.random_exit_step
            self.random_exit_step = False
            try:
                x1 = self._run_generator(st["latent_shape"], st["cond0"], image_latents=gt[:, :CONTEXT],
                                         reference_frame_count=CONTEXT, grad_on_last=True)
            finally:
                self.random_exit_step = saved
            loss1 = torch.zeros((), device=self.device)
            if self.gt_regression_weight > 0:
                mse = (x1[:, CONTEXT:].float() - gt[:, CONTEXT:WINDOW].float()).square().mean()
                loss1 = loss1 + self.gt_regression_weight * mse
                self._last_gt_reg = float(mse.item())
            if w_mm > 0:                                                               # v8.3
                mm, r_lf, r_hf, mm_stats = self._motion_match3(x1[:, CONTEXT:], gt[:, CONTEXT - 1:CONTEXT],
                                                               gt[:, CONTEXT - 1:WINDOW], cap_mm)
                loss1 = loss1 + w_mm * mm
                self._last_mm3.update(mm3_lf_ratio=float(r_lf.mean()), mm3_hf_ratio=float(r_hf.mean()),
                                      mm3_loss=float(mm.item()))
                self._mm_w = w_mm
            if use_first:
                adv1, touch1 = self._adv_term(x1, lambda xs: self._first_inputs(xs, gt[:, :CONTEXT]), cap, {})
                loss1 = loss1 + w_adv * adv1 + touch1
                self._last_adv_first = float(adv1.item())
            self._backward(loss1)
            if mm_stats:                                                               # v8.3: after backward
                self._last_mm3.update(mm3_dmd_ratio=mm_stats["norm"] / max(scale, 1e-30) / max(dmd_norm, 1e-30),
                                      mm3_cap_frac=float(mm_stats["scale"] < 1.0))
            elif w_mm > 0:
                self._last_mm3.update(mm3_dmd_ratio=0.0, mm3_cap_frac=0.0)
            del x1, loss1

        if use_scaler:
            self.grad_scaler.unscale_(self.optimizer)
        if "max_norm_for_grad_clip" in self.training_config:
            self.model_wrapper.clip_grad_norm_(self.training_config["max_norm_for_grad_clip"])
        if use_scaler:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self._last_gen_step_ok = float(self.grad_scaler.get_scale()) >= scale
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
                print(f"GEN_STEP_SKIPPED global_step={self._cur_step} total={self._skipped_gen_steps}", flush=True)
        if stats:
            self._last_gan_ratio = stats["norm"] / max(scale, 1e-30) / max(dmd_norm, 1e-30)
            self._last_cap_frac = float(stats["scale"] < 1.0)
        self._last_gen_loss = loss_value
        self._last_dmd_grad_norm = float((grad * mask.to(grad.dtype)).abs().sum().item()
                                         / mask.expand_as(grad).sum().clamp(min=1.0).item())
        self._last_adv = float(adv.item())
        return {"gen_loss": self._last_gen_loss, "dmd_grad_norm": self._last_dmd_grad_norm,
                "adv_gen": self._last_adv}

    @torch.no_grad()
    def _report(self, stats):
        out = super()._report(stats)
        if self.motion_match_weight > 0:
            out.update(self._last_mm3)
        return out
