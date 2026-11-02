"""CPU logic test for pipeline_v8_3 (no GPU, no real model): python3 scripts/test_v8_3_logic.py

Reuses the stand-ins of test_v7_logic.py and checks:
  0. with the temporal heads and the motion match off it passes every v7 logic test unchanged;
  1. the v8.3 term: zero when the student matches the real change; the seam step counts (a jump at the first generated
     frame alone gives a loss and a gradient on that frame); blurring a student whose high-frequency detail boils
     leaves the high-frequency part of the loss unchanged while v8.2's raw-change ratio would fall; a student that is
     too static (low-pass change below the floor) gets a loss whose gradient step adds motion; gradient capped;
  2. student step: the first-window pass carries the term; the rollout-window term is added when the main pass exited
     at the final step and skipped otherwise; with the weight 0 no first-window pass runs (as v7);
  3. with temporal heads on as well, critic and student rows have identical keys incl. the mm3_* keys.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_v7_logic as t7                                               # noqa: E402

from dwm_pano_cfdmd.pipeline import CONTEXT, WINDOW                      # noqa: E402
from dwm_pano_cfdmd.pipeline_v7 import PanoContextForcedDMDv7, latent_motion   # noqa: E402
from dwm_pano_cfdmd.pipeline_v8_3 import PanoContextForcedDMDv8_3, change_stats   # noqa: E402

B, V, C, H, W = t7.B, t7.V, t7.C, t7.H, t7.W
T = WINDOW - CONTEXT


def defaults(cls, temporal, weight):
    cls._v8_temporal = temporal
    cls.temporal_adv_from, cls.temporal_adv_ramp, cls._last_temporal_weight = 0, 0, 0.0
    cls.motion_match_weight, cls.motion_match_cap = weight, 0.35
    cls.motion_match_from, cls.motion_match_ramp = 0, 0
    cls.mm3_pool, cls.mm3_w_lf, cls.mm3_w_hf, cls.mm3_lf_floor, cls.mm3_rollout = 4, 1.0, 1.0, 0.9, True
    return cls


class Off(PanoContextForcedDMDv8_3):
    pass


defaults(Off, False, 0.0)


def make(temporal=False, weight=1.0, **kwargs):
    cls = defaults(type("Case", (PanoContextForcedDMDv8_3,), {}), temporal, weight)
    t7.PanoContextForcedDMDv7 = cls
    try:
        p, gt = t7.make(**kwargs)
    finally:
        t7.PanoContextForcedDMDv7 = PanoContextForcedDMDv7
    p._last_mm = {"mm_ratio": 1.0, "mm_loss": 0.0, "mm_dmd_ratio": 0.0, "mm_cap_frac": 0.0}
    p._mm_stats, p._mm_w = None, 0.0
    p._last_mm3 = {"mm3_hf_ratio": 1.0, "mm3_lf_ratio": 1.0, "mm3_loss": 0.0, "mm3_cap_frac": 0.0,
                   "mm3_dmd_ratio": 0.0, "mm3_roll_hf_ratio": 1.0, "mm3_roll_lf_ratio": 1.0, "mm3_roll_on": 0.0}
    return p, gt


def real_clip(seed=5):
    """Real frames [B, 1 + T, ...]: slow low-pass motion plus a fixed high-frequency texture."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(B, 1, V, C, H // 4, W // 4, generator=g)
    drift = torch.randn(B, 1 + T, V, C, H // 4, W // 4, generator=g).cumsum(dim=1) * 0.05
    lf = torch.nn.functional.interpolate((base + drift).flatten(0, 2), size=(H, W), mode="nearest")
    texture = torch.randn(B, 1, V, C, H, W, generator=g) * 0.5
    return lf.unflatten(0, (B, 1 + T, V)) + texture


def main():
    # 0) everything off == v7
    t7.PanoContextForcedDMDv7 = Off
    try:
        t7.main()
    finally:
        t7.PanoContextForcedDMDv7 = PanoContextForcedDMDv7
    print("ok: v8.3 with temporal heads and motion match off passes all v7 logic tests")

    p, _ = make()
    real = real_clip()
    hist = real[:, :1]

    # 1a) matching student -> zero loss
    x = real[:, 1:].clone().requires_grad_(True)
    loss, r_lf, r_hf, _ = p._motion_match3(x, hist, real, None)
    assert float(loss) < 1e-10 and abs(float(r_lf.mean()) - 1) < 1e-5 and abs(float(r_hf.mean()) - 1) < 1e-5
    # 1b) seam: only the first generated frame jumps (by a low-pass offset) -> loss and gradient on that frame
    x = real[:, 1:].clone()
    x[:, 0] += 0.5
    x.requires_grad_(True)
    loss, r_lf, _, _ = p._motion_match3(x, hist, real, None)
    g = torch.autograd.grad(loss, x)[0]
    assert float(loss) > 0 and float(r_lf.mean()) > 1 and float(g[:, 0].abs().sum()) > 0, "seam step counts"
    assert latent_motion(torch.cat([hist, x.detach()], 1)[:, :1 + 1]) > 0
    # 1c) blur-proof: boiling high-frequency detail; scaling the detail down (blur) keeps the HF part unchanged
    boil = real[:, 1:] + torch.randn(B, T, V, C, H, W, generator=torch.Generator().manual_seed(9)) * 0.5
    lf_part = torch.nn.functional.interpolate(torch.nn.functional.avg_pool2d(boil.flatten(0, 2), 4), size=(H, W),
                                              mode="nearest").unflatten(0, (B, T, V))
    blurred = lf_part + 0.3 * (boil - lf_part)
    _, _, r_hf_boil, _ = p._motion_match3(boil, hist, real, None)
    _, _, r_hf_blur, _ = p._motion_match3(blurred, hist, real, None)
    v82_boil = latent_motion(boil) / latent_motion(real[:, 1:])
    v82_blur = latent_motion(blurred) / latent_motion(real[:, 1:])
    assert float(r_hf_boil.mean()) > 1.5, "boiling detail is seen"
    # (inside the window the ratio is scale-free; the seam step from the sharp history frame to a blurred first frame
    # is itself a jump, so blurring can only raise it)
    assert float(r_hf_blur.mean()) >= 0.85 * float(r_hf_boil.mean()), "blurring does not lower the HF ratio"
    assert float(v82_blur.mean()) < 0.5 * float(v82_boil.mean()), "while v8.2's raw-change ratio falls"
    # 1d) too static: 2% of the real change -> two-sided LF part gives a loss; a gradient step adds motion
    static = (hist + 0.02 * (real[:, 1:] - hist)).clone().requires_grad_(True)
    loss_s, r_lf_s, _, _ = p._motion_match3(static, hist, real, None)
    assert float(r_lf_s.mean()) < 0.9 and float(loss_s) > 0
    lf_s0, _ = change_stats(torch.cat([hist, static.detach()], 1), 4)
    g = torch.autograd.grad(loss_s, static)[0]
    stepped = (static - 0.2 * g / g.norm() * static.norm() * 0.05).detach()
    lf_s1, _ = change_stats(torch.cat([hist, stepped], 1), 4)
    assert float(lf_s1.mean()) > float(lf_s0.mean()), "a gradient step on a too-static student adds motion"
    # 1e) cap
    x = boil.clone().requires_grad_(True)
    loss_c, _, _, st = p._motion_match3(x, hist, real, 1e-6)
    (1000.0 * loss_c).backward()
    assert abs(float(x.grad.norm()) - 1e-6) < 1e-9 and st["scale"] < 1.0, "gradient at the output is capped"
    print(f"ok: v8.3 term zero on a match, sees the seam step, HF ratio {float(r_hf_boil.mean()):.2f} -> "
          f"{float(r_hf_blur.mean()):.2f} after blurring (v8.2 ratio {float(v82_boil.mean()):.2f} -> "
          f"{float(v82_blur.mean()):.2f}), too-static student pushed to move, gradient capped")

    # 2) student step: first window always, rollout term only after a final-step exit
    def student_run(weight, force_exit=None):
        q, _ = make(weight=weight, warmup=(1000, 10))
        if force_exit is not None:
            q.random_exit_step = False
            orig = q._run_generator

            def run(*a, **k):
                out = orig(*a, **k)
                q._last_exit_idx = force_exit
                return out
            q._run_generator = run
        t7.student(q, 5)
        first = [c for c in q.calls["gen"] if c["tag"] == "w0"]
        return q, first

    q, first = student_run(1.0, force_exit=3)
    assert len(first) == 4 and q._last_mm3["mm3_roll_on"] == 1.0, "rollout term after a final-step exit"
    assert q._last_mm3["mm3_loss"] >= 0 and q._last_mm3["mm3_lf_ratio"] != 1.0
    q1, _ = student_run(1.0, force_exit=1)
    assert q1._last_mm3["mm3_roll_on"] == 0.0, "skipped after an early exit"
    q0, first0 = student_run(0.0)
    assert first0 == [] and q0._last_mm3["mm3_roll_on"] == 0.0, "weight 0: no first-window pass, as v7"
    print(f"ok: student step carries the first-window term (lf {q._last_mm3['mm3_lf_ratio']:.2f}, "
          f"hf {q._last_mm3['mm3_hf_ratio']:.2f}) and the rollout term only after a final-step exit; weight 0 = v7")

    # 3) temporal heads on as well: identical report keys
    q, gtq = make(temporal=True, weight=1.0)
    rows = [t7.critic(q, gtq, k) for k in range(2)]
    t7.student(q, 3)
    rows.append(t7.critic(q, gtq, 3))
    keys = set(rows[0])
    assert all(set(r) == keys for r in rows) and {"mm3_hf_ratio", "mm3_lf_ratio", "mm3_loss", "mm3_roll_on",
                                                  "mm_weight", "temporal_weight", "dt_gap"} <= keys, keys
    assert all(all(v == v for v in r.values()) for r in rows)
    print("ok: with temporal heads and the v8.3 term on, critic and student steps run; identical report keys")
    print("ALL V8.3 LOGIC TESTS PASSED")


if __name__ == "__main__":
    main()
