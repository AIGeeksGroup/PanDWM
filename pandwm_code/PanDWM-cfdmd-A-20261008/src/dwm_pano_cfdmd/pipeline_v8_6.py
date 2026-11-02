"""v8.6 = v8.4 with its own mm3 low-pass floor for the rollout window (2026-10-16; prepared, not yet run).

Why (eval_v7/V82_EVAL_HPC3_20261011.md section 30, appendix on window 1 / window 7 of the student and the teacher):
  * from the same drifted history the 50-step teacher's window 7 beats the student's (FVD 176.8 vs 188.4);
  * in pixels the student's window-7 frame change is 0.92 x the recording (low-pass 0.90), the teacher's 1.00;
  * the student moves ~9% less than the teacher at window 1 as well (1.02 vs 1.10 x real: the v8.2 / v8.3
    anti-flicker effect, right there), but at depth both lose ~10% and the student lands below the recording;
    window-7 FVD is lowest for arms near 1.0 (over- and under-motion are both worse);
  * v8.4 training monitors (steps 4251-4500, branches I and K): latent low-pass change ratio ~0.97 at rollout windows
    1-2 and ~0.90 at windows 5-9, i.e. pinned at mm3_lf_floor 0.9, the lower edge of mm3's two-sided low-pass band.
    mm3 ratios are of mean SQUARED frame change, so 0.9 is ~0.95 in amplitude and 0.97 is ~0.985.

What changes (everything else is v8.4): the rollout-window mm3 term (the trained window on the student's own
history; final-step exits only) uses mm3_lf_floor_rollout as its low-pass floor,
    loss = w_hf * relu(log r_hf)^2 + w_lf * [relu(log r_lf)^2 + relu(log floor_rollout - log r_lf)^2]
while the first-window term keeps mm3_lf_floor.  mm3_lf_floor_rollout None (the default) calls v8.3's term unchanged,
i.e. v8.4 bit for bit.  The floor is relative to each clip's own real change, so static scenes are not pushed to move.
Note: the term's gradient at the output is capped (CapGrad, motion_match_cap x DMD) as a whole (LF + HF parts).  When
only the low-pass floor part is active and the student is below both floors, the gradient direction does not depend on
the floor, so a capped gradient is the same for 0.9 and 0.97; when the HF part is active too, a higher floor shifts
the combined direction toward more low-pass motion even when capped (Codex review 2026-10-16: 3.4% different at LF
ratio 0.64).  The new floor also acts on ratios between the two floors and on uncapped steps.  Report keys added when
the knob is on (from the first row): mm3_roll_cap_frac (1 when the rollout term's gradient was capped on its last
run), mm3_roll_dmd_ratio (its output-gradient norm BEFORE the cap / the DMD gradient norm, as mm3_dmd_ratio for the
first window).  Both are rank-local values carried from the last run of the rollout term (not reduced over ranks), so
a pilot sees whether the cap binds on that rank.

2026-10-16 addition (branch M): motion_match_cap_rollout (default None = motion_match_cap, i.e. the code of branch L
bit for bit) caps the rollout-window term at motion_match_cap_rollout x DMD instead.  Why: branch L (floor 0.97 only)
showed mm3_roll_cap_frac 1.0 on every run of the rollout term, uncapped 1.3-19 x DMD, roll HF ratio 0.84-0.90 (HF part
inactive) and roll LF ratio ~0.77 (below both floors), so its capped gradient equalled v8.4's; same-step deep motion
vs branch I +0.003 / -0.004 (null).  The first-window term, the sharpness floor and the GAN keep their caps.
"""
import torch

from .pipeline import CONTEXT, STRIDE, WINDOW
from .pipeline_v7 import TARGET
from .pipeline_v8_4 import _EPS, PanoContextForcedDMDv8_4, hf_energy


class PanoContextForcedDMDv8_6(PanoContextForcedDMDv8_4):
    # Class default keeps instances built without __init__ (the CPU logic tests) usable.
    mm3_lf_floor_rollout = None
    motion_match_cap_rollout = None

    def __init__(self, *args, mm3_lf_floor_rollout=None, motion_match_cap_rollout=None, **kwargs):
        if mm3_lf_floor_rollout is not None:
            mm3_lf_floor_rollout = float(mm3_lf_floor_rollout)
            if not 0.0 < mm3_lf_floor_rollout <= 1.0:
                raise ValueError(f"mm3_lf_floor_rollout must be in (0, 1], got {mm3_lf_floor_rollout}")
        self.mm3_lf_floor_rollout = mm3_lf_floor_rollout
        if motion_match_cap_rollout is not None:
            motion_match_cap_rollout = float(motion_match_cap_rollout)
            if not motion_match_cap_rollout > 0.0:
                raise ValueError(f"motion_match_cap_rollout must be > 0, got {motion_match_cap_rollout}")
        self.motion_match_cap_rollout = motion_match_cap_rollout
        super().__init__(*args, **kwargs)
        if self._v86_on():                                  # report keys exist from the first row on
            self._last_mm3.update(mm3_roll_dmd_ratio=0.0, mm3_roll_cap_frac=0.0)
        if self.should_save:
            roll = self.mm3_lf_floor if self.mm3_lf_floor_rollout is None else self.mm3_lf_floor_rollout
            rcap = self.motion_match_cap if self.motion_match_cap_rollout is None else self.motion_match_cap_rollout
            print(f"V8_6 mm3 low-pass floor: first window {self.mm3_lf_floor}, rollout window {roll}; rollout cap "
                  f"{rcap} x DMD (first window {self.motion_match_cap})", flush=True)

    def _v86_on(self):
        return self.mm3_lf_floor_rollout is not None or self.motion_match_cap_rollout is not None

    def _motion_match3(self, x_gen, hist_last, real_seq, cap, floor=None):
        """v8.3's term; `floor` replaces mm3_lf_floor for this call only (None: v8.3's call unchanged)."""
        if floor is None:
            return super()._motion_match3(x_gen, hist_last, real_seq, cap)
        saved = self.mm3_lf_floor
        self.mm3_lf_floor = float(floor)
        try:
            return super()._motion_match3(x_gen, hist_last, real_seq, cap)
        finally:
            self.mm3_lf_floor = saved

    # A copy of PanoContextForcedDMDv8_4._generator_step_cf; the only change is the rollout-window mm3 floor
    # ("v8.6").  The rest is unchanged.
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
        deep = self.deep_dmd_scale != 1.0 and self.context_windows >= self.deep_dmd_from_k             # v8.4
        g_used = grad.double() * self.deep_dmd_scale if deep else grad.double()                       # v8.4
        target = (x0_det.double() - g_used).detach()
        mask = target_mask.to(torch.float64)
        while mask.ndim < x0_pred.ndim:
            mask = mask.unsqueeze(-1)
        diff = (x0_pred.double() - target) * mask
        count = mask.expand_as(diff).sum().clamp(min=1.0)
        loss = 0.5 * diff.square().sum() / count
        dmd_norm = float((grad.double() * mask).norm() / count)        # unscaled: the caps below do not shrink
        cap = None if self.gan_ratio_cap is None else self.gan_ratio_cap * dmd_norm * scale
        cap_mm = None if self.motion_match_cap is None else self.motion_match_cap * dmd_norm * scale
        cap_roll = (cap_mm if self.motion_match_cap_rollout is None                               # v8.6 (M)
                    else self.motion_match_cap_rollout * dmd_norm * scale)
        self._v84_last["gan_cap"] = 0.0 if cap is None else float(cap)                                # v8.4
        adv = torch.zeros((), device=self.device)
        stats = {}
        if w_adv > 0 and "rollout" in adv_pairs:
            adv, touch = self._adv_term(x0_pred, lambda xs: self._rollout_inputs(xs, exit_main), cap, stats)
            loss = loss + w_adv * adv.double() + touch.double()
        w_mm = self._motion_weight()
        st = self._stash
        roll_stats = None
        if w_mm > 0:
            self._last_mm3["mm3_roll_on"] = 0.0
        final_exit = exit_main == len(self.denoising_step_list) - 1
        if (w_mm > 0 and self.mm3_rollout and final_exit and "gt" in st):
            start = STRIDE * self.context_windows                                      # first frame of this window
            real_seq = st["gt"][:, start + CONTEXT - 1:start + WINDOW]
            if real_seq.shape[1] == WINDOW - CONTEXT + 1:
                roll, r_lf, r_hf, roll_stats = self._motion_match3(x0_pred[:, CONTEXT:], history[:, -1:],
                                                                   real_seq, cap_roll,
                                                                   floor=self.mm3_lf_floor_rollout)  # v8.6
                loss = loss + w_mm * roll.double()
                self._last_mm3.update(mm3_roll_lf_ratio=float(r_lf.mean()), mm3_roll_hf_ratio=float(r_hf.mean()),
                                      mm3_roll_on=1.0)
        sf_stats = None                                                                # v8.4: sharpness floor
        self._v84_last["sf_on"] = 0.0
        if final_exit and "gt" in st and (self.sharp_floor_weight > 0 or self.v84_monitors):
            start = STRIDE * self.context_windows
            real_t = st["gt"][:, start + CONTEXT:start + WINDOW]
            if real_t.shape[1] == TARGET:
                if self.sharp_floor_weight > 0:
                    sf, r_e, sf_stats = self._sharp_floor(x0_pred[:, CONTEXT:], real_t, cap_mm)
                    loss = loss + self.sharp_floor_weight * sf.double()
                    self._v84_last.update(sf_on=1.0, sf_loss=float(sf.item()))
                else:
                    with torch.no_grad():
                        r_e = (hf_energy(x0_det[:, CONTEXT:], self.mm3_pool)
                               / hf_energy(real_t, self.mm3_pool).clamp(min=_EPS))
                self._v84_last.update(sf_ratio=float(r_e.mean()), sf_fresh=1.0)
        self._backward(loss)
        loss_value = float(loss.item())
        if self._v84_last["sf_on"]:                                                    # v8.4: after backward
            if sf_stats:
                self._v84_last.update(sf_dmd_ratio=sf_stats["norm"] / max(scale, 1e-30) / max(dmd_norm, 1e-30),
                                      sf_cap_frac=float(sf_stats["scale"] < 1.0))
            else:
                self._v84_last.update(sf_dmd_ratio=0.0, sf_cap_frac=0.0)
        if self._v86_on() and roll_stats is not None:                                 # v8.6: after backward
            if roll_stats:
                self._last_mm3.update(
                    mm3_roll_dmd_ratio=roll_stats["norm"] / max(scale, 1e-30) / max(dmd_norm, 1e-30),
                    mm3_roll_cap_frac=float(roll_stats["scale"] < 1.0))
            else:
                self._last_mm3.update(mm3_roll_dmd_ratio=0.0, mm3_roll_cap_frac=0.0)
        del x0_pred, diff, loss, target, noisy, x0_det, g_used
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
            if w_mm > 0:
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
            if mm_stats:
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
