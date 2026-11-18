"""CPU logic test for pipeline_v8_4 (no GPU, no real model): python3 scripts/test_v8_4_logic.py

Reuses the stand-ins of test_v7_logic.py / test_v8_3_logic.py and checks:
  0. with the depth fixed at 2, the sharpness floor off, no deep DMD scaling and the monitors off it passes every v8.3
     logic test unchanged (the copied student step is v8.3's);
  1. rollout depth: always in [k_min, k_max], the same for the same (seed, step) on every call (so all ranks and every
     resume draw the same k without communication), every value drawn over 2000 steps, the global torch RNG untouched;
     train_step sets context_windows before the parent runs; a clip too short for k_max fails on the first step;
  2. sharpness floor: zero at or above the floor, positive for a blurred student, and a gradient step restores detail;
     gradient at the output capped;
  3. teacher direction: positive when the teacher's x0 carries more of the student's own detail, negative when it is
     blurrier; the copied _kl_grad_with_history returns exactly the parent's gradient and records the direction;
  4. depth monitors: per-window ratios are 1 for rollout windows equal to the real ones and NaN beyond k;
  5. student step: the floor term runs after a final-step exit and is skipped after an early exit; deep DMD scaling
     halves the DMD gradient only for k >= deep_dmd_from_k and leaves the GAN cap unchanged;
  6. report: identical keys on every row incl. the v8.4 keys, no NaN.
"""
import contextlib
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_v7_logic as t7                                               # noqa: E402
import test_v8_3_logic as t83                                            # noqa: E402

from dwm_pano_cfdmd.pipeline import CONTEXT, STRIDE, WINDOW, PanoContextForcedDMD   # noqa: E402
from dwm_pano_cfdmd.pipeline_v7 import PanoContextForcedDMDv7            # noqa: E402
from dwm_pano_cfdmd.pipeline_v8_3 import PanoContextForcedDMDv8_3       # noqa: E402
from dwm_pano_cfdmd.pipeline_v8_4 import (PanoContextForcedDMDv8_4, draw_rollout_depth, hf_energy,   # noqa: E402
                                          teacher_direction)

B, V, C, H, W = t7.B, t7.V, t7.C, t7.H, t7.W
T = WINDOW - CONTEXT
KMAX = 10
FRAMES = CONTEXT + STRIDE * (KMAX + 1)


def defaults84(cls, kmin=2, kmax=KMAX, floor_weight=1.0, deep_scale=1.0, deep_from=99, monitors=True):
    cls.context_windows_min, cls.context_windows_max, cls.rollout_depth_seed = kmin, kmax, 0
    cls.sharp_floor_weight, cls.sharp_floor = floor_weight, 0.9
    cls.deep_dmd_scale, cls.deep_dmd_from_k = deep_scale, deep_from
    cls.v84_monitors = monitors
    return cls


def neutral(base_defaults):
    """A v8.4 class that must behave exactly like v8.3: depth 2, floor off, no deep scaling, monitors off."""
    def build(name):
        cls = type(name, (PanoContextForcedDMDv8_4,), {})
        return defaults84(cls, kmin=2, kmax=2, floor_weight=0.0, monitors=False)
    off = build("Off84")
    base_defaults(off, False, 0.0)
    return off, build


def make(weight=1.0, kmax=KMAX, floor_weight=1.0, deep_scale=1.0, deep_from=99, monitors=True, **kwargs):
    cls = t83.defaults(type("Case84", (PanoContextForcedDMDv8_4,), {}), False, weight)
    defaults84(cls, kmax=kmax, floor_weight=floor_weight, deep_scale=deep_scale, deep_from=deep_from,
               monitors=monitors)
    t7.PanoContextForcedDMDv7 = cls
    try:
        p, gt = t7.make(**kwargs)
    finally:
        t7.PanoContextForcedDMDv7 = PanoContextForcedDMDv7
    p._last_mm = {"mm_ratio": 1.0, "mm_loss": 0.0, "mm_dmd_ratio": 0.0, "mm_cap_frac": 0.0}
    p._mm_stats, p._mm_w = None, 0.0
    p._last_mm3 = {"mm3_hf_ratio": 1.0, "mm3_lf_ratio": 1.0, "mm3_loss": 0.0, "mm3_cap_frac": 0.0,
                   "mm3_dmd_ratio": 0.0, "mm3_roll_hf_ratio": 1.0, "mm3_roll_lf_ratio": 1.0, "mm3_roll_on": 0.0}
    p._v84_reset()
    return p, gt


def deep_stash(p, k, seed=3):
    """Real clip of FRAMES latent frames; rollout windows 0..k-1 equal to their real windows."""
    g = torch.Generator().manual_seed(seed)
    gt = torch.randn(B, FRAMES, V, C, H, W, generator=g)
    p.context_windows = k
    p._stash["gt"] = gt
    p._stash["roll"] = [gt[:, STRIDE * w + CONTEXT:STRIDE * w + WINDOW].clone() for w in range(k)]
    frame_id = torch.arange(FRAMES).view(1, FRAMES, 1).float().expand(B, FRAMES, V)
    p._batch = {"vae_images": torch.zeros(B, FRAMES, V, 3, H * 8, W * 8), "frame_id": frame_id, "clip_text": ["x"]}
    p._trunc_cond = None
    return gt


def blur(x, keep):
    """Keep `keep` of the high-frequency residual (pool 4), the low-pass part unchanged."""
    flat = x.flatten(0, 2)
    lf = torch.nn.functional.interpolate(torch.nn.functional.avg_pool2d(flat, 4), size=flat.shape[-2:],
                                         mode="nearest")
    return (lf + keep * (flat - lf)).unflatten(0, x.shape[:3])


def main():
    # 0) neutral v8.4 == v8.3 on every v8.3 (and v7) logic test
    off84, build = neutral(t83.defaults)
    saved_off, saved_cls, saved_make = t83.Off, t83.PanoContextForcedDMDv8_3, t7.make

    def make_reset(*args, **kwargs):                  # the stand-ins skip __init__, which builds the v8.4 state
        p, gt = saved_make(*args, **kwargs)
        if isinstance(p, PanoContextForcedDMDv8_4):
            p._v84_reset()
        return p, gt

    t83.Off, t83.PanoContextForcedDMDv8_3, t7.make = off84, build("Neutral84"), make_reset
    try:
        t83.main()
    finally:
        t83.Off, t83.PanoContextForcedDMDv8_3, t7.make = saved_off, saved_cls, saved_make
    print("ok: neutral v8.4 (depth 2, floor off, no deep scaling, monitors off) passes all v8.3 and v7 logic tests")

    # 1) rollout depth
    state = torch.get_rng_state()
    ks = [draw_rollout_depth(0, s, 2, KMAX) for s in range(2000)]
    assert torch.equal(state, torch.get_rng_state()), "drawing k must not touch the global torch RNG"
    assert min(ks) == 2 and max(ks) == KMAX and set(ks) == set(range(2, KMAX + 1)), sorted(set(ks))
    assert ks == [draw_rollout_depth(0, s, 2, KMAX) for s in range(2000)], "same (seed, step) -> same k"
    assert ks != [draw_rollout_depth(1, s, 2, KMAX) for s in range(2000)], "the seed matters"
    counts = [ks.count(k) for k in range(2, KMAX + 1)]
    assert min(counts) > 150, counts                                      # ~222 each when uniform
    assert all(draw_rollout_depth(0, s, 2, 2) == 2 for s in range(50))
    p, _ = make()
    seen = {}

    def parent_step(self, batch, global_step):
        seen[global_step] = self.context_windows
        return "parent"

    own = PanoContextForcedDMDv8_3.__dict__.get("train_step")
    PanoContextForcedDMDv8_3.train_step = parent_step
    try:
        batch = {"vae_images": torch.zeros(B, FRAMES, V, 3, 8, 8)}
        for step in (4001, 4002, 4003):
            assert p.train_step(batch, step) == "parent"
            assert seen[step] == draw_rollout_depth(0, step, 2, KMAX) == p.context_windows
        short = {"vae_images": torch.zeros(B, 51, V, 3, 8, 8)}
        try:
            p.train_step(short, 4004)
            raise AssertionError("a 51-frame clip must be rejected when k_max is 10")
        except ValueError as e:
            assert "179" in str(e), e
    finally:
        if own is None:
            del PanoContextForcedDMDv8_3.train_step
        else:
            PanoContextForcedDMDv8_3.train_step = own
    print(f"ok: depth k in [2, {KMAX}], same for the same step on every call, all values drawn "
          f"(counts {min(counts)}..{max(counts)}), torch RNG untouched, set before the parent's train_step, "
          f"short clips rejected up front")

    # 2) sharpness floor
    p, _ = make()
    real = torch.randn(B, T, V, C, H, W, generator=torch.Generator().manual_seed(7))
    x = real.clone().requires_grad_(True)
    loss, ratio, _ = p._sharp_floor(x, real, None)
    assert float(loss) == 0.0 and abs(float(ratio.mean()) - 1.0) < 1e-5
    x = blur(real, math.sqrt(0.95)).requires_grad_(True)                  # HF energy 0.95 x real: above the floor
    loss, ratio, _ = p._sharp_floor(x, real, None)
    assert float(loss) == 0.0 and 0.9 < float(ratio.mean()) < 1.0
    x = blur(real, 0.5).requires_grad_(True)                              # HF energy 0.25 x real
    loss, ratio, _ = p._sharp_floor(x, real, None)
    assert float(loss) > 0 and float(ratio.mean()) < 0.3
    g = torch.autograd.grad(loss, x)[0]
    stepped = (x - 0.05 * g / g.norm() * x.norm()).detach()
    assert float(hf_energy(stepped, 4).mean()) > float(hf_energy(x.detach(), 4).mean()), "a step restores detail"
    x = blur(real, 0.5).requires_grad_(True)
    loss_c, _, st = p._sharp_floor(x, real, 1e-6)
    (1000.0 * loss_c).backward()
    assert abs(float(x.grad.norm()) - 1e-6) < 1e-9 and st["scale"] < 1.0, "gradient at the output is capped"
    print(f"ok: sharpness floor zero at 1.0 and 0.95 x real HF energy, positive at {float(ratio.mean()):.2f}, a "
          f"gradient step restores detail, gradient capped")

    # 3) teacher direction
    x0 = torch.randn(B, WINDOW, V, C, H, W, generator=torch.Generator().manual_seed(11))
    sharper = x0.clone()
    sharper[:, CONTEXT:] = blur(x0[:, CONTEXT:], 1.5)
    blurrier = x0.clone()
    blurrier[:, CONTEXT:] = blur(x0[:, CONTEXT:], 0.5)
    proj_s, e_s = teacher_direction(sharper, x0, x0, 4)
    proj_b, e_b = teacher_direction(blurrier, x0, x0, 4)
    assert float(proj_s.mean()) > 0.4 and float(e_s.mean()) > 2.0, (float(proj_s.mean()), float(e_s.mean()))
    assert float(proj_b.mean()) < -0.4 and float(e_b.mean()) < 0.3, (float(proj_b.mean()), float(e_b.mean()))
    proj_0, e_0 = teacher_direction(x0, x0, x0, 4)
    assert abs(float(proj_0.mean())) < 1e-6 and abs(float(e_0.mean()) - 1.0) < 1e-5

    q = object.__new__(PanoContextForcedDMDv8_4)
    q.fake_score_causal = q.real_score_causal = True
    q._add_temporal_attention_mask = lambda cond, shape: cond.__setitem__("mask", tuple(shape[:3]))
    q.get_autocast_context = contextlib.nullcontext
    q.model_dtype, q.real_guidance_scale, q.mm3_pool = torch.float32, 2.0, 4
    q.fake_score_wrapper = lambda noisy, ts, **cond: ([noisy * 0.3 + 0.1], None, None)
    q.real_score_wrapper = lambda noisy, ts, **cond: ([blur(noisy, 0.7) * torch.tensor([0.2, 0.25]).view(
        2, 1, 1, 1, 1, 1) - 0.05], None, None)
    q._v84_reset()
    noisy = torch.randn(1, WINDOW, V, C, H, W, generator=torch.Generator().manual_seed(12))
    ts = torch.full((1, WINDOW, V), 500.0)
    ts[:, :CONTEXT] = 0
    sig = torch.full((1, WINDOW, V, 1, 1, 1), 0.5)
    x0d = torch.randn(1, WINDOW, V, C, H, W, generator=torch.Generator().manual_seed(13))
    mask = torch.ones(1, WINDOW, V, dtype=torch.bool)
    mask[:, :CONTEXT] = False
    want = PanoContextForcedDMD._kl_grad_with_history(q, noisy, ts, sig, x0d, {"tag": "c"}, {"tag": "u"}, mask)
    got = q._kl_grad_with_history(noisy, ts, sig, x0d, {"tag": "c"}, {"tag": "u"}, mask)
    assert torch.equal(want, got), "the copy must return the parent's gradient"
    assert q._v84_last["tdir_fresh"] == 1.0 and math.isfinite(q._v84_last["tdir_proj"])
    print(f"ok: teacher direction +{float(proj_s.mean()):.2f} for a sharper teacher x0, {float(proj_b.mean()):.2f} "
          f"for a blurrier one; the copied _kl_grad_with_history equals the parent and records it")

    # 4) depth monitors
    p, _ = make()
    deep_stash(p, 7)
    hf, mot = p._depth_monitors()
    assert len(hf) == len(mot) == KMAX
    assert all(abs(v - 1.0) < 1e-5 for v in hf[:7]) and all(abs(v - 1.0) < 1e-5 for v in mot[:7])
    assert all(math.isnan(v) for v in hf[7:]) and all(math.isnan(v) for v in mot[7:])
    p._stash["roll"][6] = blur(p._stash["roll"][6], 0.5)
    hf, _ = p._depth_monitors()
    assert hf[6] < 0.3 and abs(hf[5] - 1.0) < 1e-5, "the deepest window is measured against its own real window"
    print("ok: depth monitors = 1 for rollout windows equal to the real ones, NaN beyond k, per-window real windows")

    # 5) student step
    def student_run(force_exit, k=2, **kw):
        q, _ = make(**kw)
        if k != 2:
            deep_stash(q, k)
        q.random_exit_step = False
        orig = q._run_generator

        def run(*a, **kk):
            out = orig(*a, **kk)
            q._last_exit_idx = force_exit
            return out
        q._run_generator = run
        t7.student(q, 5)
        return q

    q = student_run(3)
    assert q._v84_last["sf_on"] == 1.0 and math.isfinite(q._v84_last["sf_ratio"])
    q1 = student_run(1)
    assert q1._v84_last["sf_on"] == 0.0, "skipped after an early exit"
    base = student_run(3, k=10, weight=0.0, floor_weight=0.0, warmup=(1000, 10))
    scaled = student_run(3, k=10, weight=0.0, floor_weight=0.0, warmup=(1000, 10), deep_scale=0.5, deep_from=6)
    shallow = student_run(3, k=4, weight=0.0, floor_weight=0.0, warmup=(1000, 10), deep_scale=0.5, deep_from=6)
    g_base, g_scaled, g_shallow = (float(r.gen_param.grad) for r in (base, scaled, shallow))
    assert abs(g_scaled / g_base - 0.5) < 1e-6, (g_base, g_scaled)
    assert abs(g_shallow / g_base - 1.0) < 1e-6, "no scaling below deep_dmd_from_k"
    gan1 = student_run(3, k=10, weight=0.0, floor_weight=0.0, big_gan=True, g_weight=1.0)
    gan2 = student_run(3, k=10, weight=0.0, floor_weight=0.0, big_gan=True, g_weight=1.0, deep_scale=0.5,
                       deep_from=6)
    assert gan1._v84_last["gan_cap"] > 0 and gan1._v84_last["gan_cap"] == gan2._v84_last["gan_cap"], \
        "the GAN cap keeps using the unscaled DMD norm"
    print(f"ok: floor term after a final-step exit only; deep DMD scaling 0.5 -> DMD gradient x"
          f"{g_scaled / g_base:.2f} at k=10, x{g_shallow / g_base:.2f} at k=4; GAN cap unchanged")

    # 6) report keys
    q, gtq = make()
    rows = [t7.critic(q, gtq, k) for k in range(2)]
    t7.student(q, 3)
    rows.append(t7.critic(q, gtq, 3))
    deep_stash(q, 10)
    rows.append(t7.critic(q, q._stash["gt"], 4))
    keys = set(rows[0])
    v84_keys = {"rollout_k", "hf_ratio_deep", "motion_ratio_deep", "sf_ratio", "sf_loss", "sf_dmd_ratio",
                "sf_cap_frac", "tdir_proj", "tdir_eratio"}
    assert all(set(r) == keys for r in rows) and v84_keys <= keys, keys ^ set(rows[-1])
    assert all(all(v == v for v in r.values()) for r in rows), "no NaN in the report"
    assert rows[-1]["rollout_k"] == 10.0 and rows[0]["rollout_k"] == 2.0
    print("ok: critic and student rows have identical keys incl. the v8.4 keys; no NaN")
    print("ALL V8.4 LOGIC TESTS PASSED")


if __name__ == "__main__":
    main()
