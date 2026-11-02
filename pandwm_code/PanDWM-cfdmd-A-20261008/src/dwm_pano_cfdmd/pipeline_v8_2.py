"""v8.2 = v8.1 plus a direct frame-to-frame change match on the student's first window (2026-10-11).

Why: v7 step 1000 (149-clip eval) changes about 15% more from frame to frame than the real footage (B0: 5-8%).  The
excess is spread over all scales (low-pass structure 1.15x GT, high-frequency detail 1.13x, regional brightness 1.17x;
the first and second temporal differences carry the same excess), so it is not a particular texture shimmer or a
brightness flicker.  The v8.1 temporal discriminator heads cut the latent flicker monitor by only ~0.5% (same-step
paired, steps 1310-1450): their real/fake gap stayed near 0.02.

What: on the first rollout window (the student's deployed 4-step sample from the clip's real history, which v7 already
computes for the "first" adversarial pair) compare the student's latent frame-to-frame change with the real future's:
    r = mean_k |x_k - x_{k-1}|^2 (student, 16 target frames) / same for the real frames      (per sample)
and add  w_m * mean(relu(r - 1))  to the student loss.  Only the excess over the real change is penalised, so the
student is never pushed below the real amount of change (no "frozen video"); this is the training monitor
motion_ratio_w1 turned into a loss.  The gradient this term sends to the student output is capped at
`motion_match_cap` x the DMD gradient (CapGrad, as for the adversarial terms), and w_m ramps linearly from 0 at
`motion_match_from` to `motion_match_weight` over `motion_match_ramp` steps.

Everything else as v8.1.  Report adds mm_ratio (r before the update, mean over samples), mm_loss, mm_dmd_ratio
(uncapped gradient norm of the term / DMD gradient norm) and mm_cap_frac.
"""
import torch
import torch.nn.functional as F

from .pipeline import CONTEXT, WINDOW
from .pipeline_v7 import CapGrad, latent_motion
from .pipeline_v8_1 import PanoContextForcedDMDv8_1


class PanoContextForcedDMDv8_2(PanoContextForcedDMDv8_1):
    def __init__(self, *args, motion_match_weight=0.0, motion_match_cap=0.35, motion_match_from=0,
                 motion_match_ramp=0, **kwargs):
        self.motion_match_weight = float(motion_match_weight)
        self.motion_match_cap = None if motion_match_cap is None else float(motion_match_cap)
        self.motion_match_from, self.motion_match_ramp = int(motion_match_from), int(motion_match_ramp)
        self._last_mm = {"mm_ratio": 1.0, "mm_loss": 0.0, "mm_dmd_ratio": 0.0, "mm_cap_frac": 0.0}
        self._mm_stats, self._mm_w = None, 0.0
        super().__init__(*args, **kwargs)
        if self.should_save:
            print(f"V8_2 motion_match_weight={self.motion_match_weight} cap={self.motion_match_cap} "
                  f"from={self.motion_match_from} ramp={self.motion_match_ramp}", flush=True)

    def _motion_weight(self):
        if self.motion_match_weight <= 0:
            return 0.0
        step = self._cur_step or 0
        if step < self.motion_match_from:
            return 0.0
        if self.motion_match_ramp <= 0:
            return self.motion_match_weight
        return self.motion_match_weight * min(1.0, (step - self.motion_match_from + 1) / self.motion_match_ramp)

    def _motion_match(self, x1, gt, cap):
        """w_m-free motion-match loss on the first window and its stats; gradient to x1 capped."""
        stats = {}
        x = CapGrad.apply(x1, cap, stats)
        fake = latent_motion(x[:, CONTEXT:])
        real = latent_motion(gt[:, CONTEXT:WINDOW]).detach().clamp(min=1e-12)
        ratio = fake / real
        return F.relu(ratio - 1.0).mean(), ratio.detach(), stats

    # A copy of PanoContextForcedDMDv7._generator_step_cf with the motion-match term added to the first-window pass
    # (marked "v8.2"); the rest is unchanged.
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
        adv = torch.zeros((), device=self.device)
        stats = {}
        if w_adv > 0 and "rollout" in adv_pairs:
            adv, touch = self._adv_term(x0_pred, lambda xs: self._rollout_inputs(xs, exit_main), cap, stats)
            loss = loss + w_adv * adv.double() + touch.double()
        self._backward(loss)
        loss_value = float(loss.item())
        del x0_pred, diff, loss, target, noisy, x0_det
        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.use_gan:
            self.disc_optimizer.zero_grad(set_to_none=True)

        st = self._stash
        use_first = w_adv > 0 and "first" in adv_pairs
        w_mm = self._motion_weight()                                                   # v8.2
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
            if w_mm > 0:                                                               # v8.2
                cap_mm = None if self.motion_match_cap is None else self.motion_match_cap * dmd_norm * scale
                mm, ratio, mm_stats = self._motion_match(x1, gt, cap_mm)
                loss1 = loss1 + w_mm * mm
                self._last_mm.update(mm_ratio=float(ratio.mean()), mm_loss=float(mm.item()))
                self._mm_stats, self._mm_w = mm_stats, w_mm
                if not mm_stats:                       # no backward reached the term (e.g. r <= 1 everywhere)
                    self._last_mm.update(mm_dmd_ratio=0.0, mm_cap_frac=0.0)
            else:
                self._mm_stats = None
            if use_first:
                adv1, touch1 = self._adv_term(x1, lambda xs: self._first_inputs(xs, gt[:, :CONTEXT]), cap, {})
                loss1 = loss1 + w_adv * adv1 + touch1
                self._last_adv_first = float(adv1.item())
            self._backward(loss1)
            if self._mm_stats:                                                         # v8.2: after backward
                # weighted, uncapped gradient of the term over the DMD gradient (same convention as gan_dmd_ratio)
                self._last_mm.update(mm_dmd_ratio=self._mm_stats["norm"] / max(scale, 1e-30) / max(dmd_norm, 1e-30),
                                     mm_cap_frac=float(self._mm_stats["scale"] < 1.0))
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
            out.update(self._last_mm)
            out["mm_weight"] = self._motion_weight()
        return out
