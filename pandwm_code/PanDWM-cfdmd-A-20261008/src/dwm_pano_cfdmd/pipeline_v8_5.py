"""v8.5 = v8.4 with a noise-truncated, decaying CFG in the DMD real score (2026-10-14; prepared, not yet run).

Why (eval_v7/V82_EVAL_HPC3_20261011.md sections 26-27): every checkpoint loses ~27 FID points between the first and
the second half of the 19.5 s long videos.  HorizonDrive (arXiv 2605.11596, Table 3; nuScenes ~20 s rollouts, a
4-step DMD student, CFG scale 6) finds CFG in the DMD real score harmful for long rollouts: FID / FVD 20.84 / 184.06
with CFG at every noise level, 14.59 / 110.81 without CFG, 13.82 / 92.99 when CFG is restricted to noise levels below a
threshold tau_th that decays 1000 -> 0 after a 100-step warm-up.  v8.4 guides the real score at every DMD timestep
with effective scale 1 + real_guidance_scale = 3 on text + 3D boxes + HD map (the unconditional branch blanks all
three; camera poses and the 3 history latents are kept in both branches).

What changes (everything else is v8.4):
    real_x0 = cond + (cond - uncond) * s_i,    s_i = real_guidance_scale if t_i < tau_th(step) else 0
  t_i is the DMD timestep shared by sample i's target frames; tau_th(step) = cfg_tau_start up to cfg_decay_from, then
  linear to cfg_tau_end at cfg_decay_to, held after.  The defaults (tau 1001 > every t) are v8.4 exactly.  The
  unconditional half is still computed when no sample is guided (same forwards on every rank and step; the cost is
  unchanged, the result is the conditional score).
Report keys added unless the schedule is v8.4's (both thresholds > 1000): cfg_tau (the threshold this step), cfg_frac
(fraction of this rank's samples guided on the last student step).  Rank 0 then prints a V85 line on student steps.
"""
import torch

from .pipeline import CONTEXT
from .pipeline_v8_4 import PanoContextForcedDMDv8_4, teacher_direction


def cfg_threshold(step, tau_start, tau_end, decay_from, decay_to):
    """tau_start until decay_from, then linear to tau_end at decay_to, then tau_end."""
    step = int(step)
    if step <= decay_from:
        return float(tau_start)
    if step >= decay_to:
        return float(tau_end)
    frac = (step - decay_from) / float(decay_to - decay_from)
    return float(tau_start) + frac * (float(tau_end) - float(tau_start))


class PanoContextForcedDMDv8_5(PanoContextForcedDMDv8_4):
    # Class defaults keep instances built without __init__ (the CPU logic tests) usable; they equal v8.4.
    cfg_tau_start, cfg_tau_end, cfg_decay_from, cfg_decay_to = 1001.0, 1001.0, 0, 0

    def __init__(self, *args, cfg_tau_start=1001.0, cfg_tau_end=1001.0, cfg_decay_from=0, cfg_decay_to=0,
                 **kwargs):
        self.cfg_tau_start, self.cfg_tau_end = float(cfg_tau_start), float(cfg_tau_end)
        self.cfg_decay_from, self.cfg_decay_to = int(cfg_decay_from), int(cfg_decay_to)
        if self.cfg_decay_to < self.cfg_decay_from:
            raise ValueError(f"cfg_decay_to {cfg_decay_to} < cfg_decay_from {cfg_decay_from}")
        self._v85_last = {"cfg_tau": self.cfg_tau_start, "cfg_frac": 1.0}
        super().__init__(*args, **kwargs)
        if self.should_save:
            print(f"V8_5 DMD real-score CFG scale {self.real_guidance_scale} (effective {1 + self.real_guidance_scale}) "
                  f"only for t < tau_th; tau_th {self.cfg_tau_start} until step {self.cfg_decay_from}, linear to "
                  f"{self.cfg_tau_end} at step {self.cfg_decay_to}", flush=True)

    def cfg_tau(self):
        return cfg_threshold(getattr(self, "_cur_step", 0) or 0, self.cfg_tau_start, self.cfg_tau_end,
                             self.cfg_decay_from, self.cfg_decay_to)

    # A copy of PanoContextForcedDMDv8_4._kl_grad_with_history with the per-sample guidance scale ("v8.5").
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
        tau = self.cfg_tau()                                                                     # v8.5
        t_target = ts[:, CONTEXT].reshape(ts.shape[0], -1)[:, 0].float()
        guided = (t_target < tau).to(real_cond.dtype)
        s = (guided * float(self.real_guidance_scale)).view(-1, *([1] * (real_cond.ndim - 1)))
        real_x0 = real_cond + (real_cond - real_uncond) * s
        self.__dict__.setdefault("_v85_last", {}).update(cfg_tau=tau, cfg_frac=float(guided.mean()))
        if self.v84_monitors:
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

    @torch.no_grad()
    def _report(self, stats):
        out = super()._report(stats)
        if self.cfg_tau_start > 1000 and self.cfg_tau_end > 1000:
            return out                                   # the v8.4 schedule: same report as v8.4
        frac = float(self.__dict__.get("_v85_last", {}).get("cfg_frac", 1.0))
        out.update(cfg_tau=float(self.cfg_tau()), cfg_frac=frac)
        step = getattr(self, "_cur_step", 0) or 0
        if self.should_save and step % max(int(getattr(self, "dfake_gen_update_ratio", 1)), 1) == 0:
            print(f"V85 step={step} cfg_tau={self.cfg_tau():.1f} cfg_frac={frac:.2f}", flush=True)
        return out
