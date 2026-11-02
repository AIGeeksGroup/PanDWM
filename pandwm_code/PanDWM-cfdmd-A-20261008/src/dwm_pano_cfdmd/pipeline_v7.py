"""v7: a discriminator that judges the generated frames fairly, a capped GAN push, and teacher guidance matched
to how the student is used.  All earlier pipeline files are untouched; v7 subclasses v5 (per-rank noise,
milestones, EMA gate and the skip counter are kept) but owns both update steps.

Review of v5 before it started (2026-10-10; code, the run-A diagnostic, a literature survey, Codex):

1. Teacher guidance stronger than anything the student was built for.  The DMD real score is
   x_c + s * (x_c - x_u) with s = real_guidance_scale = 3: effective guidance 1 + s = 4.  The teacher's own
   sampler (ctsd_pano.py, x_u + 3 * (x_c - x_u)) and the consistency distillation that made B0 both use 3.
   Run A step 1585 is sharper than the real footage (Laplacian variance 371 vs GT 329, B0 352) and further
   from the real future (PSNR -0.42 dB).  Fix: config real_guidance_scale = 2.0 (effective 3).  No code.
2. The discriminator mostly looked at noise.  It reused the DMD sampler (shift 5): median t = 833, 83% of
   inputs above sigma 0.5, where the targets are invisible (run-A logit gap 1.71 at sigma 0.21, 0.05 at 0.95).
   And 75% of its fakes are blurry early-exit predictions, so at low noise "blurry = fake" is the easy rule,
   which pushes the student towards sharper output.  Fix: one sigma per sample tied to the exit step the fake
   came from (Self-Forcing's ts_schedule): a sample that left the 4-step sampler at step s is judged with
   sigma between the sampler's next and current noise levels (training steps 1000/900/751/502: the final step
   covers [disc_sigma_min, 0.502]).
   Real and fake of a pair share sigma and the noise tensor.
3. D trained the fake score.  Its loss went into F, the network DMD relies on, on inputs F never saw (history
   at t = 1000, targets lower).  Fix: F is trained by flow matching only; the discriminator forwards run under
   no_grad and only small heads learn (LingBot-World, LADD).  This also makes the critic step cheaper than v5.
4. One mean-pooled logit per clip: the gradient was the same vector for every token, a global "sharper"
   push that cannot point at an edge or seam.  Fix: per-token heads (2 x 2 pooled tokens) on several blocks
   (`disc_layers`), hinge loss for D, -mean(D) for the student.
5. Real and fake now differ only in the frames being judged:
     "rollout" pair, trained (third) window: the 16 target frames alone as a 16-frame clip with conditions
       rebuilt for those frames (`disc_rollout_input` "truncate"; "noise" = v5's noise history), so the
       student-made history is not visible;
     "first" pair, first window: real history clean in both; fake = the first rollout window the step already
       sampled with the deployed 4-step sampler, real = the real future.  Here the discriminator may also judge
       whether the future follows the real past.
6. Nothing bounded the GAN push.  Fix: the adversarial gradient that reaches the student's output is rescaled
   to at most `gan_ratio_cap` x the DMD gradient (closed form), per sample; adversarial weight 0 for the first
   `gan_warmup[0]` steps (heads learn first), then a linear ramp over `gan_warmup[1]` steps.
   `adv_schedule` "both" applies the rollout and the first-window term on every student step (two passes, one
   student graph at a time); "alternate" uses one of them per student step (rollout on even, first on odd
   student steps), so the student step peaks like v5's.

Optional, off by default: `gt_regression_weight` (MSE of the student's first window to the real future).

Every critic row reports the same keys (the trainer averages each key over all rows; job 703667).  Rollout
pair: disc_loss / disc_real_acc / disc_fake_acc / d_real / d_fake / d_gap_lo / d_gap_hi; first pair: disc1_*;
student (value of the last student step): gan_dmd_ratio, gan_cap_frac, adv_first, gt_reg, gan_weight;
monitors (real footage = 1.0): hf_ratio_w1 / hf_ratio_w2 (latent high-frequency energy of the deployed first /
second rollout window over GT; > 1 = sharper than real), motion_ratio_w1 (frame-to-frame change over GT;
< 1 = less motion), mem_peak_gb (largest allocated GPU memory of any rank so far in this process).  Metrics are
averaged over all ranks.
"""
import math

import torch
import torch.nn.functional as F

import dwm.functional

from .pipeline import CONTEXT, STRIDE, WINDOW
from .pipeline_v5 import PanoContextForcedDMDv5

TARGET = WINDOW - CONTEXT
PAIRS = ("rollout", "first")
LOW_SIGMA = 0.4


class CapGrad(torch.autograd.Function):
    """Identity forward; backward rescales the incoming gradient to norm <= max_norm (None = no cap).

    max_norm already includes the GradScaler scale.  A non-finite gradient passes unchanged so the scaler
    skips the step.
    """

    @staticmethod
    def forward(ctx, x, max_norm, stats):
        ctx.max_norm, ctx.stats = max_norm, stats
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        norm = float(grad.detach().double().norm())
        scale = 1.0
        if ctx.max_norm is not None and math.isfinite(norm) and norm > ctx.max_norm > 0:
            scale = ctx.max_norm / norm
        ctx.stats["norm"], ctx.stats["scale"] = norm, scale
        return (grad * scale if scale != 1.0 else grad), None, None


def make_heads(layers, hidden, width=256):
    """One small per-token head per feature block (same layer types as the v1-v5 head)."""
    return torch.nn.ModuleDict({
        str(int(layer)): torch.nn.Sequential(torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, width),
                                             torch.nn.SiLU(), torch.nn.Linear(width, 1))
        for layer in layers})


def hinge_d(real_logits, fake_logits):
    real = sum(F.relu(1.0 - x).mean() for x in real_logits) / len(real_logits)
    fake = sum(F.relu(1.0 + x).mean() for x in fake_logits) / len(fake_logits)
    return real + fake


def gen_adv(fake_logits):
    return -sum(x.mean() for x in fake_logits) / len(fake_logits)


def latent_hf_energy(x):
    """Mean squared finite difference over latent H and W per sample.  x: [B, T, V, C, H, W]."""
    x = x.float()
    dh = (x[..., 1:, :] - x[..., :-1, :]).square().flatten(1).mean(1)
    dw = (x[..., :, 1:] - x[..., :, :-1]).square().flatten(1).mean(1)
    return dh + dw


def latent_motion(x):
    """Mean squared frame-to-frame difference per sample.  x: [B, T, V, C, H, W]."""
    x = x.float()
    return (x[:, 1:] - x[:, :-1]).square().flatten(1).mean(1)


class PanoContextForcedDMDv7(PanoContextForcedDMDv5):
    # pipeline.PanoContextForcedDMD.__init__ assigns its mean-pool head to `self.disc_head`, builds the head
    # optimizer from it and, on resume, loads <step>_disc_head.pth into it.  The property swaps in the v7 heads
    # at that assignment, so the optimizer, resume, save_checkpoint and milestones all use them unchanged.
    @property
    def disc_head(self):
        return self._v7_heads

    @disc_head.setter
    def disc_head(self, _parent_head):
        hidden = self.fake_score.config.num_attention_heads * self.fake_score.config.attention_head_dim
        self._v7_heads = make_heads(self._v7_layers, hidden).to(self.device)

    def __init__(self, *args, disc_pairs=PAIRS, disc_rollout_input="truncate", disc_layers=(6, 12, 18),
                 disc_pool=2, disc_sigma_min=0.05, disc_sigma_max=0.8, disc_exit_matched=True,
                 gan_ratio_cap=0.5, gan_warmup=(100, 200), adv_schedule="both", gt_regression_weight=0.0,
                 disc_history="noise", **kwargs):
        if disc_history != "noise":
            raise ValueError("v7 ignores v5's disc_history; use disc_rollout_input")
        self._v7_layers = tuple(int(layer) for layer in disc_layers)
        super().__init__(*args, disc_history="noise", **kwargs)
        pairs = tuple(disc_pairs)
        if not pairs or len(set(pairs)) != len(pairs) or any(p not in PAIRS for p in pairs):
            raise ValueError(f"disc_pairs must be a non-empty subset of {PAIRS}, got {disc_pairs}")
        if disc_rollout_input not in ("truncate", "noise"):
            raise ValueError(f"disc_rollout_input must be 'truncate' or 'noise', got {disc_rollout_input}")
        if adv_schedule not in ("both", "alternate"):
            raise ValueError(f"adv_schedule must be 'both' or 'alternate', got {adv_schedule}")
        self.adv_schedule = adv_schedule
        self.disc_pairs = pairs
        self.disc_rollout_input = disc_rollout_input
        self.disc_pool = int(disc_pool)
        self.disc_sigma_min, self.disc_sigma_max = float(disc_sigma_min), float(disc_sigma_max)
        self.disc_exit_matched = bool(disc_exit_matched)
        self.gan_ratio_cap = None if gan_ratio_cap is None else float(gan_ratio_cap)
        self.gan_warmup = (int(gan_warmup[0]), int(gan_warmup[1]))
        self.gt_regression_weight = float(gt_regression_weight)
        self._stash, self._batch, self._trunc_cond = {}, None, None
        self._last_exit_idx = len(self.denoising_step_list) - 1
        self._last_gan_ratio = self._last_cap_frac = self._last_adv_first = self._last_gt_reg = 0.0
        self._last_gan_weight = 0.0
        self._mem_peak = 0.0
        self._carry = {"d_gap_lo": 0.0, "d_gap_hi": 0.0, "hf_ratio_w1": 1.0, "hf_ratio_w2": 1.0,
                       "motion_ratio_w1": 1.0}
        if self.use_gan:
            # The parent hooked block `gan_feature_block`; hook the v7 blocks with the same (overridden) method.
            blocks = self.fake_score.transformer_blocks
            hooked = {int(kwargs.get("gan_feature_block", 12))}
            self._v7_block_ids = {id(blocks[layer]): layer for layer in self._v7_layers}
            for layer in self._v7_layers:
                if layer not in hooked:
                    blocks[layer].register_forward_hook(self._feature_hook)
                    hooked.add(layer)
        if self.should_save:
            print(f"V7 disc_pairs={list(pairs)} rollout_input={disc_rollout_input} layers={self._v7_layers} "
                  f"pool={self.disc_pool} sigma=[{self.disc_sigma_min},{self.disc_sigma_max}] "
                  f"exit_matched={self.disc_exit_matched} cap={self.gan_ratio_cap} warmup={self.gan_warmup} "
                  f"adv_schedule={adv_schedule} "
                  f"gt_regression_weight={self.gt_regression_weight} "
                  f"real_guidance_scale={self.real_guidance_scale} gan_generator_weight={self.gan_generator_weight} "
                  f"steps={[round(float(t), 1) for t in self.denoising_step_list]}", flush=True)

    # ------------------------------------------------------------------ data captured from the step
    def _feature_hook(self, module, inputs, output):
        if self._capture:
            layer = getattr(self, "_v7_block_ids", {}).get(id(module))
            if layer is not None:
                # JointTransformerBlock returns (encoder_hidden_states, hidden_states).
                self._features[layer] = output[1] if isinstance(output, tuple) else output

    def train_step(self, batch, global_step):
        self._stash, self._batch, self._trunc_cond = {}, batch, None
        return super().train_step(batch, global_step)

    def _vae_encode(self, vae_images):
        out = super()._vae_encode(vae_images)
        if isinstance(getattr(self, "_stash", None), dict):
            self._stash.setdefault("gt", out)
        return out

    @torch.no_grad()
    def _rollout_window(self, latent_shape, cond, history):
        x0 = super()._rollout_window(latent_shape, cond, history)
        stash = getattr(self, "_stash", None)
        if isinstance(stash, dict):
            if "cond0" not in stash:         # first call of the step = first window, from the real history
                stash.update(cond0=cond, latent_shape=latent_shape)
            stash.setdefault("roll", []).append(x0[:, CONTEXT:].detach())
        return x0

    def _run_generator(self, latent_shape, cond, image_latents=None, reference_frame_count: int = 0,
                       grad_on_last: bool = True):
        """Same as PanoCrossviewTemporalDMD._run_generator, plus `self._last_exit_idx` (the step it exited at)."""
        if grad_on_last:
            self.model_wrapper.train()
        else:
            self.model_wrapper.eval()
        latents = torch.randn(latent_shape, generator=self.generator).to(self.device)
        cond_gen = dict(cond)
        self._add_temporal_attention_mask(cond_gen, latent_shape)
        n_steps = len(self.denoising_step_list)
        if self.random_exit_step:
            exit_idx = torch.randint(0, n_steps, (1,), device=self.device)
            torch.distributed.broadcast(exit_idx, src=0)
            exit_idx = int(exit_idx.item())
        else:
            exit_idx = n_steps - 1
        self._last_exit_idx = exit_idx
        for i in range(n_steps):
            is_exit = i == exit_idx
            t_val = self.denoising_step_list[i]
            sigma_t = self.denoising_sigmas[i].to(dtype=latents.dtype)
            timesteps = torch.full(latent_shape[:3], float(t_val.item()), dtype=t_val.dtype, device=self.device)
            inp = latents
            if image_latents is not None and reference_frame_count > 0:
                inp = torch.cat([image_latents[:, :reference_frame_count], latents[:, reference_frame_count:]],
                                dim=1)
                timesteps[:, :reference_frame_count] = 0
            inp = inp.to(dtype=self.model_dtype)
            grad_ctx = torch.enable_grad() if (grad_on_last and is_exit) else torch.no_grad()
            with grad_ctx, self.get_autocast_context():
                model_output, _, _ = self.model_wrapper(inp, timesteps, **cond_gen)
            x0_pred = inp + (-sigma_t) * model_output[0]
            if is_exit:
                return x0_pred
            sigma_next = self.denoising_sigmas[i + 1].to(dtype=latents.dtype)
            new_noise = torch.randn(latent_shape, generator=self.generator).to(self.device)
            latents = (1.0 - sigma_next) * x0_pred + sigma_next * new_noise

    # ------------------------------------------------------------------ discriminator inputs
    def _disc_sigma(self, shape, exit_idx, dtype):
        """One (t, sigma) per sample on the scheduler grid; range tied to the exit step when exit-matched."""
        steps = [float(t) / 1000.0 for t in self.denoising_step_list]
        if self.disc_exit_matched:
            hi = steps[exit_idx]
            lo = steps[exit_idx + 1] if exit_idx + 1 < len(steps) else self.disc_sigma_min
        else:
            lo, hi = self.disc_sigma_min, self.disc_sigma_max
        hi = min(hi, float(self.max_step) / 1000.0)
        lo = min(max(lo, self.disc_sigma_min), hi)
        u = torch.rand((shape[0], 1, 1), device=self.device)
        wanted = ((lo + (hi - lo) * u) * 1000.0).expand(shape).contiguous()
        idx = (self.train_ts.view(1, -1).float() - wanted.flatten().view(-1, 1)).abs().argmin(dim=1)
        t = self.train_ts[idx].view(*shape).float()
        sigma = self.train_sigmas[idx].view(*shape)
        return t, sigma.view(*shape, 1, 1, 1).to(dtype)

    def _truncated_cond(self):
        """Conditions of the trained window's 16 target frames as their own 16-frame clip (once per step)."""
        if self._trunc_cond is None:
            batch = self._batch
            first = STRIDE * self.context_windows + CONTEXT
            sub = {key: (value if key in self.window_exceptions else
                         dwm.functional.take_sequence_clip(value, first, first + TARGET))
                   for key, value in batch.items()}
            sub.pop("vae_images", None)
            shape = self._get_latent_shape(batch, TARGET, batch["vae_images"].shape[2])
            cond = self._conditions(sub, shape, False)
            if self.fake_score_causal:
                self._add_temporal_attention_mask(cond, shape)
            self._trunc_cond = cond
        return self._trunc_cond

    def _with_mask(self, cond, shape):
        cond = dict(cond)
        if self.fake_score_causal:
            self._add_temporal_attention_mask(cond, shape)
        return cond

    def _rollout_inputs(self, xs, exit_idx):
        """Rollout pair inputs for window tensors xs (19 frames each), sharing sigma and noise.

        Returns ([noisy...], t, sigma, cond_for_F, drop_context)."""
        if self.disc_rollout_input == "truncate":
            targets = [x[:, CONTEXT:] for x in xs]
            t, sigma = self._disc_sigma(targets[0].shape[:3], exit_idx, targets[0].dtype)
            eps = torch.randn_like(targets[0])
            noisy = [sigma * eps + (1.0 - sigma) * x for x in targets]
            return noisy, t, sigma, self._truncated_cond(), False
        t, sigma = self._disc_sigma(xs[0].shape[:3], exit_idx, xs[0].dtype)
        eps = torch.randn_like(xs[0])
        hist = self.denoising_sigmas[0].to(dtype=xs[0].dtype) * torch.randn_like(xs[0][:, :CONTEXT])
        noisy = [torch.cat([hist, (sigma * eps + (1.0 - sigma) * x)[:, CONTEXT:]], dim=1) for x in xs]
        t = t.clone()
        t[:, :CONTEXT] = float(self.denoising_step_list[0].item())
        return noisy, t, sigma, self._with_mask(self._stash["cond_main"], xs[0].shape), True

    def _first_inputs(self, xs, history):
        """First pair inputs: clean real history (t = 0) in every sample, final-exit sigma, shared noise."""
        t, sigma = self._disc_sigma(xs[0].shape[:3], len(self.denoising_step_list) - 1, xs[0].dtype)
        eps = torch.randn_like(xs[0])
        noisy = [torch.cat([history.to(x.dtype), (sigma * eps + (1.0 - sigma) * x)[:, CONTEXT:]], dim=1)
                 for x in xs]
        t = t.clone()
        t[:, :CONTEXT] = 0
        return noisy, t, sigma, self._with_mask(self._stash["cond0"], xs[0].shape), True

    def _disc_features(self, noisy, ts, cond):
        """F forward; returns ({layer: block output}, model output).  cond already carries the causal mask."""
        self._features = {}
        self._capture = True
        try:
            with self.get_autocast_context():
                out, _, _ = self.fake_score_wrapper(noisy.to(self.model_dtype), ts, **cond)
        finally:
            self._capture = False
        feats = {layer: self._features[layer] for layer in self._v7_layers}
        self._features = {}
        return feats, out

    def _token_logits(self, feats, shape, drop_context):
        """Per-token logits, one tensor per head: [(b t v), hp / pool, wp / pool]."""
        b, t, v = shape[:3]
        p = self.fake_score.config.patch_size
        hp, wp = shape[-2] // p, shape[-1] // p
        out = []
        for name, head in self.disc_head.items():
            h = feats[int(name)]
            if drop_context:
                h = h.unflatten(0, (b, t, v))[:, CONTEXT:].flatten(0, 2)
            n, tokens, c = h.shape
            if tokens != hp * wp:
                raise ValueError(f"block {name}: {tokens} tokens per view, expected {hp} x {wp}")
            h = h.reshape(n, hp, wp, c).permute(0, 3, 1, 2)       # tokens are row-major (see unpatchify)
            if self.disc_pool > 1:
                h = F.avg_pool2d(h, self.disc_pool)
            out.append(head(h.permute(0, 2, 3, 1).float()).squeeze(-1))
        return out

    def _adv_pairs_this_step(self):
        if self.adv_schedule == "both" or len(self.disc_pairs) == 1:
            return self.disc_pairs
        k = (self._cur_step or 0) // max(int(self.dfake_gen_update_ratio), 1)   # student-step index, resume-safe
        return ("rollout",) if k % 2 == 0 else ("first",)

    def _adv_weight(self):
        start, ramp = self.gan_warmup
        step = self._cur_step or 0
        if not self.use_gan or step < start:
            return 0.0
        return self.gan_generator_weight * (min(1.0, (step - start + 1) / ramp) if ramp > 0 else 1.0)

    # ------------------------------------------------------------------ student
    def _backward(self, loss):
        if self.training_config.get("enable_grad_scaler", False):
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def _adv_term(self, x0, inputs_fn, cap, stats):
        """-mean(D) on the student's own sample, gradient to x0 capped; includes FSDP's 0 * output term."""
        x_adv = CapGrad.apply(x0, cap, stats)
        noisy, ts, _, cond, drop = inputs_fn([x_adv])
        feats, out = self._disc_features(noisy[0], ts, cond)
        adv = gen_adv(self._token_logits(feats, noisy[0].shape, drop))
        return adv, 0.0 * out[0].float().sum()

    def _generator_step_cf(self, latent_shape, cond, cond_cfg, history, target_mask):
        use_scaler = self.training_config.get("enable_grad_scaler", False)
        scale = float(self.grad_scaler.get_scale()) if use_scaler else 1.0
        w_adv = self._adv_weight()
        self._last_gan_weight = w_adv
        adv_pairs = self._adv_pairs_this_step()
        self._stash["cond_main"] = cond

        # Main pass: DMD exactly as v5 / pipeline.py, plus the capped rollout adversarial term.
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
        # d(DMD loss)/d(x0_pred) = grad * mask / count: its norm needs no extra backward.
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
        # Fake-score / head gradients of the adversarial path are discarded at the end of the student step;
        # free them now so the first-window pass does not carry them.
        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.use_gan:
            self.disc_optimizer.zero_grad(set_to_none=True)

        # First-window pass (own backward, one student graph at a time): student's deployed sample from the
        # clip's real history, capped adversarial term and the optional regression to the real future.
        st = self._stash
        use_first = w_adv > 0 and "first" in adv_pairs
        if (use_first or self.gt_regression_weight > 0) and "cond0" in st and "gt" in st:
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
            if use_first:
                adv1, touch1 = self._adv_term(x1, lambda xs: self._first_inputs(xs, gt[:, :CONTEXT]), cap, {})
                loss1 = loss1 + w_adv * adv1 + touch1
                self._last_adv_first = float(adv1.item())
            self._backward(loss1)
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
        self.critic_optimizer.zero_grad()          # F / head gradients of the adversarial path are not theirs
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

    # ------------------------------------------------------------------ critic
    def _critic_step_cf(self, latent_shape, cond, history, gt_window, target_mask):
        if not self.use_gan:
            return super()._critic_step_cf(latent_shape, cond, history, gt_window, target_mask)
        self._stash["cond_main"] = cond
        self.fake_score_wrapper.train()
        with torch.no_grad():
            x0 = self._run_generator(latent_shape, cond, image_latents=history, reference_frame_count=CONTEXT,
                                     grad_on_last=False).detach()
        exit_c = self._last_exit_idx
        use_scaler = self.training_config.get("enable_grad_scaler", False)

        # 1) flow matching on the clean student history: unchanged, and now F's only objective.
        noisy, ts, sigmas_b, noise = self._noised_with_clean_history(x0, history)
        cond_critic = self._with_mask(cond, latent_shape)
        with self.get_autocast_context():
            out, _, _ = self.fake_score_wrapper(noisy.to(self.model_dtype), ts, **cond_critic)
        target_v = noise - x0
        mask = target_mask.to(torch.float32)
        while mask.ndim < x0.ndim:
            mask = mask.unsqueeze(-1)
        err = (out[0].float() - target_v.float()).square() * mask
        critic_loss = err.sum() / mask.expand_as(err).sum().clamp(min=1.0)
        report = {"critic_loss": float(critic_loss.item())}
        self._backward(critic_loss)
        del out, err, critic_loss, noisy, noise

        # 2) heads on detached F features; each pair shares sigma and noise.
        stats = {}
        d_loss = torch.zeros((), device=self.device)
        if "rollout" in self.disc_pairs:
            with torch.no_grad():
                (nf, nr), t_d, sigma_d, cond_d, drop = self._rollout_inputs([x0, gt_window.to(x0.dtype)], exit_c)
                feats_f, _ = self._disc_features(nf, t_d, cond_d)
                feats_r, _ = self._disc_features(nr, t_d, cond_d)
            lf, lr = self._token_logits(feats_f, nf.shape, drop), self._token_logits(feats_r, nr.shape, drop)
            del feats_f, feats_r
            pair_loss = hinge_d(lr, lf)
            d_loss = d_loss + pair_loss
            stats["roll"] = (float(pair_loss.item()), lr, lf, float(sigma_d.flatten()[0]))
        st = self._stash
        if "first" in self.disc_pairs and "roll" in st and "gt" in st:
            real1 = st["gt"][:, :WINDOW]
            hist = real1[:, :CONTEXT]
            fake1 = torch.cat([hist, st["roll"][0].to(real1.dtype)], dim=1)
            with torch.no_grad():
                (nf, nr), t1, _, cond1, drop = self._first_inputs([fake1, real1], hist)
                feats_f, _ = self._disc_features(nf, t1, cond1)
                feats_r, _ = self._disc_features(nr, t1, cond1)
            lf1, lr1 = self._token_logits(feats_f, nf.shape, drop), self._token_logits(feats_r, nr.shape, drop)
            del feats_f, feats_r
            pair_loss = hinge_d(lr1, lf1)
            d_loss = d_loss + pair_loss
            stats["first"] = (float(pair_loss.item()), lr1, lf1)
        if d_loss.requires_grad:
            (self.gan_discriminator_weight * d_loss).backward()     # heads only: fp32, no FSDP, no scaler

        if use_scaler:
            self.grad_scaler.unscale_(self.critic_optimizer)
        self.fake_score_wrapper.clip_grad_norm_(self.max_norm_for_critic_grad_clip)
        if use_scaler:
            self.grad_scaler.step(self.critic_optimizer)
            self.grad_scaler.update()
        else:
            self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        self._step_disc_head(1.0)
        report.update(self._report(stats))
        return report

    # ------------------------------------------------------------------ logging
    @torch.no_grad()
    def _report(self, stats):
        nan = float("nan")

        def means(logits):
            return (torch.stack([x.float().mean() for x in logits]).mean().item(),
                    torch.stack([(x > 0).float().mean() for x in logits]).mean().item(),
                    torch.stack([(x < 0).float().mean() for x in logits]).mean().item())

        row = [nan] * 13
        if "roll" in stats:
            loss, lr, lf, sigma = stats["roll"]
            (r, acc_r, _), (f, _, acc_f) = means(lr), means(lf)
            row[0:6] = [loss, r, f, acc_r, acc_f, sigma]
        if "first" in stats:
            loss, lr, lf = stats["first"]
            (r, acc_r, _), (f, _, acc_f) = means(lr), means(lf)
            row[6:11] = [loss, r, f, acc_r, acc_f]
        st, roll = self._stash, self._stash.get("roll", [])
        if "gt" in st and len(roll) >= 1:
            gt1 = st["gt"][:, CONTEXT:WINDOW]
            row[11] = float((latent_hf_energy(roll[0]) / latent_hf_energy(gt1).clamp(min=1e-12)).mean())
            row[12] = float((latent_motion(roll[0]) / latent_motion(gt1).clamp(min=1e-12)).mean())
        hf2 = nan
        if "gt" in st and len(roll) >= 2:
            gt2 = st["gt"][:, STRIDE + CONTEXT:STRIDE + WINDOW]
            hf2 = float((latent_hf_energy(roll[1]) / latent_hf_energy(gt2).clamp(min=1e-12)).mean())
        row.append(hf2)
        row.append(torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else nan)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        vec = torch.tensor(row, device=self.device, dtype=torch.float32)
        if torch.distributed.is_initialized():
            rows = torch.empty(self._world * vec.numel(), device=self.device)
            torch.distributed.all_gather_into_tensor(rows, vec)
            rows = rows.view(self._world, -1)
        else:
            rows = vec.view(1, -1)

        def avg(col, default=0.0):
            v = rows[:, col]
            ok = torch.isfinite(v)
            return float(v[ok].mean()) if ok.any() else default

        out = {}
        if "rollout" in self.disc_pairs:
            out.update(disc_loss=avg(0), d_real=avg(1), d_fake=avg(2), disc_real_acc=avg(3, 0.5),
                       disc_fake_acc=avg(4, 0.5))
            gap, sig = rows[:, 1] - rows[:, 2], rows[:, 5]
            for key, sel in (("d_gap_lo", sig < LOW_SIGMA), ("d_gap_hi", sig >= LOW_SIGMA)):
                sel = sel & torch.isfinite(gap)
                if sel.any():
                    self._carry[key] = float(gap[sel].mean())
            out.update(d_gap_lo=self._carry["d_gap_lo"], d_gap_hi=self._carry["d_gap_hi"])
        if "first" in self.disc_pairs:
            out.update(disc1_loss=avg(6), d1_real=avg(7), d1_fake=avg(8), disc1_real_acc=avg(9, 0.5),
                       disc1_fake_acc=avg(10, 0.5))
        for key, col in (("hf_ratio_w1", 11), ("motion_ratio_w1", 12), ("hf_ratio_w2", 13)):
            self._carry[key] = avg(col, self._carry[key])
            out[key] = self._carry[key]
        mem = rows[:, 14]
        if torch.isfinite(mem).any():
            self._mem_peak = max(self._mem_peak, float(mem[torch.isfinite(mem)].max()))
        out["mem_peak_gb"] = self._mem_peak
        out.update(gan_dmd_ratio=self._last_gan_ratio, gan_cap_frac=self._last_cap_frac,
                   adv_first=self._last_adv_first, gt_reg=self._last_gt_reg, gan_weight=self._last_gan_weight)
        return out
