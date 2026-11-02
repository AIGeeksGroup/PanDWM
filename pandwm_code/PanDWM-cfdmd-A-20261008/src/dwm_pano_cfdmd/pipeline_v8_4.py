"""v8.4 = v8.3 trained on deep self-rollouts (2026-10-13).

Why (eval_v7/V82_EVAL_HPC3_20261011.md sections 21-24): v8.3 step 4000 is the best short-clip model (Dev-149 FVD sum
102.0), but its 20 s videos go blurry and static after ~10 s in some scenes (8 dev scenes, windows 7-12: sharpness /
GT 0.85, frame change / GT 0.90; step 2000: 0.99 / 1.00).  The collapse grows with training steps (onset ~2250-2500)
while every step rolls out only `context_windows: 2` windows; inference chains 11.  anchor_sink at inference made it
worse and the history latents' std does not drop, so the fix has to come from training.

What changes (everything else is v8.3):
  1. Rollout depth.  Every step draws k uniformly from [context_windows_min, context_windows_max] (2..10) with a
     generator seeded by (rollout_depth_seed, global_step): every rank (FSDP needs the same forwards) and every resume
     draw the same k without communication, and the global torch RNG (per-rank seeds) is untouched.
     self.context_windows = k for the whole step, so the rollout loop, the trained window's start, the truncated D
     conditions and the mm3 real window all follow it.  The clip must hold 3 + 16 * (k_max + 1) frames (179 for
     k_max 10); this is checked on every step, so a short dataset fails on the first step, not when a large k is drawn.
  2. Sharpness floor on the trained window, when the student's main pass exited at the final step (its x0 is the
     deployed sample, as for mm3's rollout term):
         r_e = HF energy (student) / HF energy (real window)      (latent residual of an mm3_pool average, 16 frames)
         loss = sharp_floor_weight * relu(log sharp_floor - log r_e)^2           zero for r_e >= sharp_floor (0.9)
     Nothing in v8.3 penalises blur: the mm3 HF part only limits relative change (flicker).  The gradient this term
     sends to the student output is capped at motion_match_cap x DMD (its own CapGrad).
  3. Monitors (v84_monitors).  Every step and every rollout window: HF energy ratio and low-pass change ratio against
     the window's real frames, averaged over ranks; the report carries the deepest window, rank 0 prints a V84 line
     with all windows.  On student steps the teacher direction on the trained window, from the copied
     _kl_grad_with_history:
         tdir_proj   = <HF(real_x0 - fake_x0), HF(x0)> / <HF(x0), HF(x0)>   (> 0: the DMD target adds the student's
                       own detail = sharper; < 0: removes it = blurrier)
         tdir_eratio = HF energy (real_x0) / HF energy (fake_x0)
     HorizonDrive (arXiv 2605.11596) reports that a standard teacher's supervision is contaminated by drifted
     histories; these monitors show whether ours is.
  4. Contingency knob, off by default: for k >= deep_dmd_from_k the DMD gradient is multiplied by deep_dmd_scale; the
     GAN, mm3 and floor caps keep using the unscaled DMD norm, so D and the floors carry the deep windows (APT2 trains
     long videos with a discriminator only).
Report keys added: rollout_k, hf_ratio_deep, motion_ratio_deep, sf_ratio, sf_loss, sf_dmd_ratio, sf_cap_frac,
tdir_proj, tdir_eratio (student-step values are carried between student steps).
"""
import math

import torch
import torch.nn.functional as F

from .pipeline import CONTEXT, STRIDE, WINDOW
from .pipeline_v7 import TARGET, CapGrad
from .pipeline_v8_3 import PanoContextForcedDMDv8_3, change_stats, split_lf_hf

_EPS = 1e-12


def draw_rollout_depth(seed, step, k_min, k_max):
    """Uniform integer in [k_min, k_max] determined by (seed, step) alone; the global torch RNG is not touched."""
    if k_max <= k_min:
        return int(k_min)
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(step))
    return int(k_min) + int(torch.randint(0, int(k_max) - int(k_min) + 1, (1,), generator=g).item())


def hf_energy(x, pool):
    """Mean squared high-frequency residual (x minus its pool x pool average) per sample.  x: [B, T, V, C, H, W]."""
    _, hf = split_lf_hf(x, pool)
    return hf.square().flatten(1).mean(1)


def teacher_direction(real_x0, fake_x0, x0, pool):
    """Per sample on the target frames: (projection of the DMD update direction real_x0 - fake_x0 onto the student's
    own high-frequency detail, relative to that detail's energy; HF energy of real_x0 over that of fake_x0)."""
    _, h_real = split_lf_hf(real_x0[:, CONTEXT:].float(), pool)
    _, h_fake = split_lf_hf(fake_x0[:, CONTEXT:].float(), pool)
    _, h_x = split_lf_hf(x0[:, CONTEXT:].float(), pool)
    proj = ((h_real - h_fake) * h_x).flatten(1).sum(1) / h_x.square().flatten(1).sum(1).clamp(min=_EPS)
    e_ratio = h_real.square().flatten(1).mean(1) / h_fake.square().flatten(1).mean(1).clamp(min=_EPS)
    return proj, e_ratio


class PanoContextForcedDMDv8_4(PanoContextForcedDMDv8_3):
    # Class defaults keep instances built without __init__ (the CPU logic tests) usable.
    context_windows_min, context_windows_max, rollout_depth_seed = 2, 10, 0
    sharp_floor_weight, sharp_floor = 1.0, 0.9
    deep_dmd_scale, deep_dmd_from_k = 1.0, 99
    v84_monitors = True

    def __init__(self, *args, context_windows_min=2, context_windows_max=10, rollout_depth_seed=0,
                 sharp_floor_weight=1.0, sharp_floor=0.9, deep_dmd_scale=1.0, deep_dmd_from_k=99, v84_monitors=True,
                 **kwargs):
        self.context_windows_min, self.context_windows_max = int(context_windows_min), int(context_windows_max)
        if not 1 <= self.context_windows_min <= self.context_windows_max:
            raise ValueError(f"need 1 <= context_windows_min <= context_windows_max, got "
                             f"{context_windows_min}, {context_windows_max}")
        self.rollout_depth_seed = int(rollout_depth_seed)
        self.sharp_floor_weight, self.sharp_floor = float(sharp_floor_weight), float(sharp_floor)
        if not 0.0 < self.sharp_floor <= 1.0:
            raise ValueError(f"sharp_floor must be in (0, 1], got {sharp_floor}")
        self.deep_dmd_scale, self.deep_dmd_from_k = float(deep_dmd_scale), int(deep_dmd_from_k)
        self.v84_monitors = bool(v84_monitors)
        self._v84_reset()
        super().__init__(*args, **kwargs)
        if self.should_save:
            print(f"V8_4 rollout depth k in [{self.context_windows_min}, {self.context_windows_max}] "
                  f"(seed {self.rollout_depth_seed}), clips of {self._frames_needed()} frames, sharp_floor "
                  f"w={self.sharp_floor_weight} floor={self.sharp_floor} (cap {self.motion_match_cap} x DMD), "
                  f"deep_dmd scale={self.deep_dmd_scale} from k={self.deep_dmd_from_k}, "
                  f"monitors={self.v84_monitors}", flush=True)

    def _v84_reset(self):
        self._v84_last = {"sf_on": 0.0, "sf_fresh": 0.0, "sf_ratio": 1.0, "sf_loss": 0.0, "sf_dmd_ratio": 0.0,
                          "sf_cap_frac": 0.0, "tdir_fresh": 0.0, "tdir_proj": 0.0, "tdir_eratio": 1.0,
                          "gan_cap": 0.0}
        self._v84_carry = {"hf_ratio_deep": 1.0, "motion_ratio_deep": 1.0, "sf_ratio": 1.0, "tdir_proj": 0.0,
                           "tdir_eratio": 1.0}

    def _frames_needed(self):
        return CONTEXT + STRIDE * (self.context_windows_max + 1)

    # ------------------------------------------------------------------ rollout depth
    def train_step(self, batch, global_step):
        frames = batch["vae_images"].shape[1]
        if frames < self._frames_needed():
            raise ValueError(f"v8.4 needs clips of {self._frames_needed()} frames for context_windows_max "
                             f"{self.context_windows_max}, got {frames}; set the dataset sequence_length")
        self.context_windows = draw_rollout_depth(self.rollout_depth_seed, global_step, self.context_windows_min,
                                                  self.context_windows_max)
        return super().train_step(batch, global_step)

    # ------------------------------------------------------------------ sharpness floor
    def _sharp_floor(self, x_gen, real_target, cap):
        """Weight-free floor term.  x_gen: student target frames [B, 16, ...] (gradient capped); real_target: the real
        window's target frames."""
        stats = {}
        x = CapGrad.apply(x_gen, cap, stats)
        fake = hf_energy(x, self.mm3_pool)
        with torch.no_grad():
            real = hf_energy(real_target, self.mm3_pool).clamp(min=_EPS)
        log_r = torch.log(fake.clamp(min=_EPS)) - torch.log(real)
        loss = F.relu(math.log(self.sharp_floor) - log_r).square().mean()
        return loss, log_r.detach().exp(), stats

    # ------------------------------------------------------------------ monitors
    @torch.no_grad()
    def _depth_monitors(self):
        """Per rollout window w < k: (HF energy ratio, low-pass change ratio) of the window's deployed sample against
        its real frames; NaN-padded to context_windows_max entries."""
        kmax = max(int(self.context_windows_max), 1)
        hf, mot = [math.nan] * kmax, [math.nan] * kmax
        st = getattr(self, "_stash", {}) or {}
        gt, roll = st.get("gt"), st.get("roll", [])
        if gt is None:
            return hf, mot
        for w, x in enumerate(roll[:kmax]):
            a = STRIDE * w + CONTEXT
            real = gt[:, a:a + TARGET].to(torch.float32)
            if real.shape[1] != x.shape[1]:
                continue
            x = x.to(torch.float32)
            hf[w] = float((hf_energy(x, self.mm3_pool) / hf_energy(real, self.mm3_pool).clamp(min=_EPS)).mean())
            lf_f, _ = change_stats(x, self.mm3_pool)
            lf_r, _ = change_stats(real, self.mm3_pool)
            mot[w] = float((lf_f / lf_r.clamp(min=_EPS)).mean())
        return hf, mot

    # A copy of PanoContextForcedDMD._kl_grad_with_history (pipeline.py) that also records the teacher direction
    # ("v8.4"); the returned gradient is unchanged.
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
        if self.v84_monitors:                                                        # v8.4
            with torch.no_grad():
                proj, e_ratio = teacher_direction(real_x0, fake_x0, x0_det, self.mm3_pool)
            self._v84_last.update(tdir_proj=float(proj.mean()), tdir_eratio=float(e_ratio.mean()), tdir_fresh=1.0)

        grad = fake_x0 - real_x0
        mask = target_mask.to(grad.dtype)
        while mask.ndim < grad.ndim:
            mask = mask.unsqueeze(-1)
        # Per-sample DMD normaliser (eq. 8), over the target frames only.
        p_real = (x0_det.float() - real_x0).abs() * mask
        normalizer = (p_real.flatten(1).sum(1) / mask.expand_as(p_real).flatten(1).sum(1).clamp(min=1.0))
        grad = grad / normalizer.clamp(min=1e-8).view(-1, *([1] * (grad.ndim - 1)))
        return torch.nan_to_num(grad).to(dtype=x0_det.dtype)

    # A copy of PanoContextForcedDMDv8_3._generator_step_cf plus the sharpness floor, the deep DMD scaling and the
    # recorded GAN cap ("v8.4"); the rest is unchanged.
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
                                                                   real_seq, cap_mm)
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

    # ------------------------------------------------------------------ logging
    @torch.no_grad()
    def _report(self, stats):
        out = super()._report(stats)
        if not self.v84_monitors:
            return out
        kmax = max(int(self.context_windows_max), 1)
        k = int(self.context_windows)
        last = self._v84_last
        hf, mot = self._depth_monitors()
        nan = math.nan
        scalars = [last["sf_ratio"] if last["sf_fresh"] else nan,
                   last["tdir_proj"] if last["tdir_fresh"] else nan,
                   last["tdir_eratio"] if last["tdir_fresh"] else nan]
        vec = torch.tensor(hf + mot + scalars, device=self.device, dtype=torch.float32)
        if torch.distributed.is_initialized():
            rows = torch.empty(self._world * vec.numel(), device=self.device)
            torch.distributed.all_gather_into_tensor(rows, vec)
            rows = rows.view(self._world, -1)
        else:
            rows = vec.view(1, -1)
        finite = torch.isfinite(rows)
        means = [float(rows[finite[:, j], j].mean()) if finite[:, j].any() else nan for j in range(rows.shape[1])]
        hf_m, mot_m = means[:kmax], means[kmax:2 * kmax]
        carry = self._v84_carry
        for key, value in (("hf_ratio_deep", hf_m[k - 1] if 1 <= k <= kmax else nan),
                           ("motion_ratio_deep", mot_m[k - 1] if 1 <= k <= kmax else nan),
                           ("sf_ratio", means[2 * kmax]), ("tdir_proj", means[2 * kmax + 1]),
                           ("tdir_eratio", means[2 * kmax + 2])):
            if math.isfinite(value):
                carry[key] = value
        out.update(rollout_k=float(k), hf_ratio_deep=carry["hf_ratio_deep"],
                   motion_ratio_deep=carry["motion_ratio_deep"], sf_ratio=carry["sf_ratio"],
                   sf_loss=last["sf_loss"], sf_dmd_ratio=last["sf_dmd_ratio"], sf_cap_frac=last["sf_cap_frac"],
                   tdir_proj=carry["tdir_proj"], tdir_eratio=carry["tdir_eratio"])
        if self.should_save:
            def fmt(values):
                return " ".join("nan" if not math.isfinite(v) else f"{v:.3f}" for v in values)
            gen = "-" if not last["tdir_fresh"] else f"{means[2 * kmax + 1]:+.4f}/{means[2 * kmax + 2]:.3f}"
            sf = "-" if not last["sf_fresh"] else f"{means[2 * kmax]:.3f}"
            print(f"V84 step={self._cur_step} k={k} hf=[{fmt(hf_m[:k])}] mot=[{fmt(mot_m[:k])}] sf_ratio={sf} "
                  f"sf_loss={last['sf_loss']:.4f} sf_dmd={last['sf_dmd_ratio']:.3f} tdir={gen}", flush=True)
        last.update(sf_fresh=0.0, tdir_fresh=0.0)
        return out
