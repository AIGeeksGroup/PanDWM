"""CPU logic test for pipeline_v7 (no GPU, no real model): python3 scripts/test_v7_logic.py

Builds a v7 instance without __init__ on tiny tensors, replaces the heavy parts (fake score, student, scaler,
DMD scores, condition builder) with stand-ins and checks:
  1. discriminator sigma: exit-matched ranges (final step [0.05, 0.625], earlier steps higher), t = 1000 sigma
     on the scheduler grid; plain range when not exit-matched;
  2. the parent's head assignment yields one per-token head per configured block;
  3. per-token logits: shapes, context frames dropped, each pooled token = exactly its own 2 x 2 block;
  4. truncated conditions = frames 35..50 of the clip, with the causal mask, built once per step;
  5. critic: F forward order (flow matching with grad, then no_grad pairs: rollout on the 16-frame clip, first
     with clean real history), a pair shares sigma and noise (first pair differs only in the future), F's
     gradient is the flow-matching one only, heads get gradients, identical keys on every row, no NaN;
  6. student: no adversarial forwards during warm-up, then a linear ramp; rollout and first-window terms run,
     the first window uses the deployed sampler (random exit off, restored) from the real history, the
     adversarial gradient reaching the student output is capped at ratio x DMD; head gradients are cleared.
"""
import contextlib

import torch

torch.distributed.broadcast = lambda tensor, src=0: None       # single process: exit index stays local

from dwm_pano_cfdmd.pipeline import CONTEXT, WINDOW              # noqa: E402
from dwm_pano_cfdmd.pipeline_v7 import CapGrad, PanoContextForcedDMDv7   # noqa: E402

B, V, C, H, W, P = 1, 2, 4, 8, 16, 2          # token grid 4 x 8, 16 channels per token
HP, WP, CH = H // P, W // P, C * P * P
LAYERS = (1, 3)


class Scaler:
    value = 1024.0

    def scale(self, loss):
        return loss * self.value

    def unscale_(self, opt):
        pass

    def step(self, opt):
        opt.step()

    def update(self):
        pass

    def get_scale(self):
        return self.value


class Opt:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1

    def zero_grad(self, set_to_none=True):
        pass


def make(pairs=("rollout", "first"), rollout_input="truncate", warmup=(0, 1), cap=0.5, exit_matched=True,
         d_weight=0.01, g_weight=0.003, big_gan=False, adv_schedule="both"):
    torch.manual_seed(0)
    p = object.__new__(PanoContextForcedDMDv7)
    p.device = torch.device("cpu")
    p._v7_layers = LAYERS
    p.fake_score = type("FS", (), {"config": type("Cfg", (), {"patch_size": P, "num_attention_heads": 2,
                                                              "attention_head_dim": CH // 2})()})()
    p.disc_head = "parent head"                                   # goes through the v7 setter
    p.disc_pairs, p.disc_rollout_input, p.disc_pool = tuple(pairs), rollout_input, 2
    p.disc_sigma_min, p.disc_sigma_max, p.disc_exit_matched = 0.05, 0.8, exit_matched
    p.gan_ratio_cap, p.gan_warmup, p.gt_regression_weight = cap, warmup, 0.0
    p.gan_generator_weight, p.gan_discriminator_weight, p.use_gan = g_weight, d_weight, True
    p.min_score_timestep, p.num_train_timestep, p.max_step = 0, 1000, 980
    p.train_ts = torch.arange(999, -1, -1).float()
    p.train_sigmas = p.train_ts / 1000
    p.denoising_step_list = torch.tensor([1000.0, 937.5, 833.3, 625.0])
    p.denoising_sigmas = p.denoising_step_list / 1000
    p.context_windows, p.window_exceptions = 2, ["clip_text"]
    p.fake_score_causal, p.model_dtype = True, torch.float32
    p.training_config = {"enable_grad_scaler": True}
    p.grad_scaler, p.optimizer, p.critic_optimizer = Scaler(), Opt(), Opt()
    p.disc_optimizer = torch.optim.AdamW(p.disc_head.parameters(), lr=1e-5)
    p.max_norm_for_critic_grad_clip, p.should_save, p.random_exit_step = 10.0, False, True
    p.generator = torch.Generator().manual_seed(0)
    p._touch_fake_output, p._capture, p._features = False, False, {}
    p._cur_step, p._world, p._skipped_gen_steps = 0, 1, 0
    p._last_exit_idx = 3
    p._last_gan_ratio = p._last_cap_frac = p._last_adv_first = p._last_gt_reg = p._last_gan_weight = 0.0
    p._last_gen_loss = p._last_dmd_grad_norm = p._last_adv = 0.0
    p._mem_peak = 0.0
    p.adv_schedule, p.dfake_gen_update_ratio = adv_schedule, 5
    p._carry = {"d_gap_lo": 0.0, "d_gap_hi": 0.0, "hf_ratio_w1": 1.0, "hf_ratio_w2": 1.0, "motion_ratio_w1": 1.0}
    p.proj = torch.nn.Parameter(torch.eye(CH))                    # stands in for the fake-score weights
    p.gen_param = torch.nn.Parameter(torch.zeros(()))             # stands in for the student weights
    p.calls = {"F": [], "gen": [], "head_steps": 0, "F_grad_at_clip": None, "head_grad": 0.0}
    gain = 1e4 if big_gan else 1.0

    def fake_score_wrapper(noisy, ts, **cond):
        p.calls["F"].append({"x": noisy.detach().clone(), "t": ts.detach().clone(), "tag": cond.get("tag"),
                             "mask": cond.get("mask", False), "grad": torch.is_grad_enabled()})
        n = noisy.shape[0] * noisy.shape[1] * noisy.shape[2]
        tokens = noisy.flatten(0, 2).unflatten(2, (HP, P)).unflatten(4, (WP, P))
        tokens = tokens.permute(0, 2, 4, 1, 3, 5).reshape(n, HP * WP, CH)
        h = (tokens @ p.proj) * gain
        if p._capture:
            for layer in LAYERS:
                p._features[layer] = h + layer
        return [noisy * p.proj[0, 0]], None, None

    fake_score_wrapper.train = lambda: None

    def clip(n):
        p.calls["F_grad_at_clip"] = None if p.proj.grad is None else p.proj.grad.detach().clone()

    fake_score_wrapper.clip_grad_norm_ = clip
    p.fake_score_wrapper = fake_score_wrapper

    def model_wrapper(inp, timesteps, **cond):
        p.calls["gen"].append({"tag": cond.get("tag"), "grad": torch.is_grad_enabled(), "rand": p.random_exit_step})
        return [inp * 0.1 + p.gen_param], None, None

    model_wrapper.train = model_wrapper.eval = lambda: None
    model_wrapper.clip_grad_norm_ = lambda n: None
    p.model_wrapper = model_wrapper
    p._add_temporal_attention_mask = lambda cond, shape: cond.__setitem__("mask", tuple(shape[:3]))
    p.get_autocast_context = contextlib.nullcontext
    p._conditions = lambda sub, shape, cfg: {"tag": "trunc", "frames": sub["frame_id"], "shape": shape}
    p._get_latent_shape = lambda batch, seq, views: (B, seq, views, C, H, W)

    def noised(x0, history, noise=None):
        noise = torch.randn_like(x0)
        s = torch.full((x0.shape[0], x0.shape[1], x0.shape[2], 1, 1, 1), 0.5)
        noisy = torch.cat([history, (s * noise + (1 - s) * x0)[:, CONTEXT:]], dim=1)
        ts = torch.full(x0.shape[:3], 500.0)
        ts[:, :CONTEXT] = 0
        return noisy, ts, s, noise

    p._noised_with_clean_history = noised
    p._kl_grad_with_history = lambda noisy, ts, s, x0, cond, cond_cfg, mask: torch.full_like(x0, 0.01)

    def step_head(scale):
        p.calls["head_steps"] += 1
        p.calls["head_grad"] = sum(float(q.grad.abs().sum()) for q in p.disc_head.parameters() if q.grad is not None)
        p.disc_optimizer.zero_grad()

    p._step_disc_head = step_head
    gt = torch.randn(B, 51, V, C, H, W)
    frame_id = torch.arange(51).view(1, 51, 1).float().expand(B, 51, V)
    p._batch = {"vae_images": torch.zeros(B, 51, V, 3, H * 8, W * 8), "frame_id": frame_id, "clip_text": ["x"]}
    p._trunc_cond = None
    p._stash = {"gt": gt, "cond0": {"tag": "w0"}, "latent_shape": (B, WINDOW, V, C, H, W),
                "roll": [torch.randn(B, WINDOW - CONTEXT, V, C, H, W), torch.randn(B, WINDOW - CONTEXT, V, C, H, W)]}
    return p, gt


def masks():
    history = torch.randn(B, CONTEXT, V, C, H, W)
    mask = torch.ones(B, WINDOW, V, dtype=torch.bool)
    mask[:, :CONTEXT] = False
    return history, mask


def critic(p, gt, step):
    p._cur_step = step
    p.calls["F"].clear()
    history, mask = masks()
    return p._critic_step_cf((B, WINDOW, V, C, H, W), {"tag": "w2"}, history, gt[:, 32:51], mask)


def student(p, step):
    p._cur_step = step
    p.calls["F"].clear()
    p.calls["gen"].clear()
    history, mask = masks()
    return p._generator_step_cf((B, WINDOW, V, C, H, W), {"tag": "w2"}, {"tag": "w2cfg"}, history, mask)


def main():
    # 1) sigma ranges
    p, _ = make()
    for exit_idx, (lo, hi) in {3: (0.05, 0.625), 2: (0.625, 0.8333), 1: (0.8333, 0.9375), 0: (0.9375, 0.98)}.items():
        t, s = p._disc_sigma((3000, 1, 1), exit_idx, torch.float32)
        s = s.flatten()
        assert s.min() >= lo - 1e-3 and s.max() <= hi + 1e-3, (exit_idx, float(s.min()), float(s.max()))
        assert torch.allclose(t.flatten() / 1000, s, atol=1e-6)
    p2, _ = make(exit_matched=False)
    s = p2._disc_sigma((3000, 1, 1), 0, torch.float32)[1].flatten()
    assert s.min() >= 0.05 - 1e-3 and s.max() <= 0.8 + 1e-3 and float(s.median()) < 0.5
    print("ok: sigma exit-matched (final [0.05, 0.625], earlier steps higher), t = 1000 sigma on the grid")

    # 2) heads via the parent's assignment
    assert isinstance(p.disc_head, torch.nn.ModuleDict) and list(p.disc_head.keys()) == ["1", "3"]
    print("ok: parent head assignment builds one per-token head per block")

    # 3) per-token logits
    h = torch.zeros(B * WINDOW * V, HP * WP, CH)
    rows = torch.arange(HP).view(HP, 1).expand(HP, WP).reshape(-1).float()
    cols = torch.arange(WP).view(1, WP).expand(HP, WP).reshape(-1).float()
    h[:, :, 0], h[:, :, 1] = rows, cols
    p.disc_head = "reset"
    for head in p.disc_head.values():                              # head -> (row, col) of each pooled token
        head.forward = (lambda x: x[..., :2])
    logits = p._token_logits({1: h, 3: h}, (B, WINDOW, V, C, H, W), drop_context=True)
    assert len(logits) == 2 and logits[0].shape == (B * (WINDOW - CONTEXT) * V, HP // 2, WP // 2, 2)
    want = torch.tensor([[[r * 2 + 0.5, c * 2 + 0.5] for c in range(WP // 2)] for r in range(HP // 2)])
    assert torch.allclose(logits[0][0], want)
    logits = p._token_logits({1: h[: B * 16 * V], 3: h[: B * 16 * V]}, (B, 16, V, C, H, W), drop_context=False)
    assert logits[0].shape[0] == B * 16 * V
    print("ok: per-token logits, context frames dropped, each pooled token = its own 2 x 2 block")

    # 4) truncated conditions
    p, gt = make()
    cond = p._truncated_cond()
    assert torch.equal(cond["frames"][0, :, 0], torch.arange(35, 51).float()) and cond["mask"] == (B, 16, V)
    assert p._truncated_cond() is cond
    print("ok: truncated conditions = frames 35..50 with the causal mask, built once per step")

    # 5) critic
    p, gt = make()
    rep = critic(p, gt, 0)
    calls = p.calls["F"]
    assert [c["tag"] for c in calls] == ["w2", "trunc", "trunc", "w0", "w0"], [c["tag"] for c in calls]
    assert calls[0]["grad"] and not any(c["grad"] for c in calls[1:]), "discriminator forwards must be no_grad"
    assert all(c["mask"] for c in calls)
    rf, rr, ff, fr = calls[1], calls[2], calls[3], calls[4]
    assert rf["x"].shape[1] == 16 and torch.equal(rf["t"], rr["t"]) and not torch.equal(rf["x"], rr["x"])
    hist = gt[:, :CONTEXT]
    assert torch.equal(ff["x"][:, :CONTEXT], hist) and torch.equal(fr["x"][:, :CONTEXT], hist)
    assert (ff["t"][:, :CONTEXT] == 0).all() and torch.equal(ff["t"], fr["t"])
    s1 = float(ff["t"][0, CONTEXT, 0]) / 1000
    assert 0.05 - 1e-3 <= s1 <= 0.625 + 1e-3
    fake1, real1 = p._stash["roll"][0], gt[:, CONTEXT:WINDOW]
    assert torch.allclose(ff["x"][:, CONTEXT:] - fr["x"][:, CONTEXT:], (1 - s1) * (fake1 - real1), atol=1e-5), \
        "first pair: same sigma and noise, differs only in the future"
    assert p.calls["head_steps"] == 1 and p.calls["head_grad"] > 0
    g_with = p.calls["F_grad_at_clip"]
    p0, gt0 = make()
    p0.gan_discriminator_weight = 0.0
    critic(p0, gt0, 0)
    assert g_with is not None and torch.allclose(g_with, p0.calls["F_grad_at_clip"]), "F trained by flow matching only"
    want_keys = {"critic_loss", "disc_loss", "d_real", "d_fake", "disc_real_acc", "disc_fake_acc", "d_gap_lo",
                 "d_gap_hi", "disc1_loss", "d1_real", "d1_fake", "disc1_real_acc", "disc1_fake_acc", "hf_ratio_w1",
                 "motion_ratio_w1", "hf_ratio_w2", "gan_dmd_ratio", "gan_cap_frac", "adv_first", "gt_reg",
                 "gan_weight", "mem_peak_gb"}
    assert set(rep) == want_keys, set(rep) ^ want_keys
    rows = [critic(p, gt, k) for k in range(1, 4)]
    assert all(set(r) == want_keys for r in rows)
    assert all(all(v == v for v in r.values()) for r in rows), "no NaN in the report"
    p, gt = make(rollout_input="noise")
    critic(p, gt, 0)
    rf = p.calls["F"][1]
    assert rf["x"].shape[1] == WINDOW and (rf["t"][:, :CONTEXT] == 1000).all() and rf["tag"] == "w2"
    print("ok: critic = flow matching (grad) + no_grad pairs; shared sigma/noise; first pair keeps the real "
          "history; F's gradient unchanged by the discriminator; identical keys; noise-history fallback works")

    # 6) student
    p, gt = make(warmup=(5, 10))
    student(p, 2)
    assert p.calls["F"] == [] and p._last_gan_weight == 0.0 and p.optimizer.steps == 1, "warm-up: DMD only"
    student(p, 9)
    assert abs(p._last_gan_weight - 0.003 * 5 / 10) < 1e-12
    tags = [c["tag"] for c in p.calls["F"]]
    assert tags == ["trunc", "w0"] and all(c["grad"] for c in p.calls["F"]), tags
    first_calls = [c for c in p.calls["gen"] if c["tag"] == "w0"]
    assert len(first_calls) == 4 and all(not c["rand"] for c in first_calls) and p.random_exit_step is True
    assert torch.equal(p.calls["F"][1]["x"][:, :CONTEXT], gt[:, :CONTEXT])
    assert all(q.grad is None for q in p.disc_head.parameters()), "student step must clear head gradients"
    assert p._last_adv_first != 0.0
    p, gt = make(big_gan=True, g_weight=1.0)
    student(p, 3)
    assert p._last_cap_frac == 1.0 and p._last_gan_ratio > 0.5, (p._last_cap_frac, p._last_gan_ratio)
    stats = {}
    x = torch.ones(10, requires_grad=True)
    CapGrad.apply(x, 2.0, stats).mul(100.0).sum().backward()
    assert abs(float(x.grad.norm()) - 2.0) < 1e-5 and stats["scale"] < 1
    p, gt = make(big_gan=True, g_weight=1.0, cap=None)
    student(p, 3)
    assert p._last_cap_frac == 0.0
    p, gt = make(adv_schedule="alternate")
    student(p, 0)
    assert [c["tag"] for c in p.calls["F"]] == ["trunc"], "alternate: student step 0 -> rollout term only"
    student(p, 5)
    assert [c["tag"] for c in p.calls["F"]] == ["w0"], "alternate: student step 1 -> first-window term only"
    student(p, 10)
    assert [c["tag"] for c in p.calls["F"]] == ["trunc"]
    print("ok: warm-up (DMD only, then ramp), rollout + first-window terms, deployed sampler from the real "
          "history, adversarial gradient capped at ratio x DMD (off when cap=None), head gradients cleared, "
          "alternate schedule")
    print("ALL V7 LOGIC TESTS PASSED")


if __name__ == "__main__":
    main()
