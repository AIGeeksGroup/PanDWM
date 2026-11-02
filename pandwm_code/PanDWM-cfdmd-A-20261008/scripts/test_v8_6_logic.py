"""CPU logic test for pipeline_v8_6 (no GPU, no real model): python3 scripts/test_v8_6_logic.py

Reuses the stand-ins of test_v7_logic.py / test_v8_3_logic.py / test_v8_4_logic.py and checks:
  0. with mm3_lf_floor_rollout None it passes every v8.4 logic test unchanged (the copied student step is v8.4's);
  1. the floor itself: for a student whose every frame change is alpha x the real one (low-pass ratio alpha^2, mm3
     compares mean squared change), the term with floor 0.97 is zero for 0.97 <= alpha^2 <= 1, positive below 0.97
     where floor 0.9 gives zero, and the same as floor 0.9 above 1; a gradient step raises the change;
     mm3_lf_floor is restored after the call, also when the call raises;
     1b. with the cap binding, the capped gradient is the same for floors 0.9 and 0.97 when only the low-pass floor
         part acts and differs when the HF part is active too;
     1c. the constructor (parent constructor stubbed) adds the two report keys only when the knob is on;
  2. routing in a student step: after a final-step exit the rollout call gets the rollout floor and the first-window
     call gets None (mm3_lf_floor); after an early exit only the first-window call runs and, on the same instance,
     the two keys keep their last values; with the knob off (None) both calls get None, there are no new keys, and
     the gradients equal plain v8.4's bit for bit; with the knob on every report row has the same keys;
  2c. branch M: with motion_match_cap_rollout the rollout call is capped at that x DMD (first window unchanged),
      the capped gradient changes, and an early exit runs only the first-window term at the old cap;
  3. the constructor rejects floors outside (0, 1] and caps <= 0; the cap knob alone adds the report keys.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_v7_logic as t7                                               # noqa: E402
import test_v8_4_logic as t84                                            # noqa: E402

from dwm_pano_cfdmd.pipeline import WINDOW, CONTEXT                      # noqa: E402
from dwm_pano_cfdmd.pipeline_v8_3 import change_stats                   # noqa: E402
from dwm_pano_cfdmd.pipeline_v8_4 import PanoContextForcedDMDv8_4       # noqa: E402
from dwm_pano_cfdmd.pipeline_v8_6 import PanoContextForcedDMDv8_6       # noqa: E402

B, V, C, H, W = t7.B, t7.V, t7.C, t7.H, t7.W
T = WINDOW - CONTEXT


def with_base(cls, fn, *args, **kwargs):
    """Run fn with test_v8_4_logic's PanoContextForcedDMDv8_4 replaced by cls (its make() and main() use it)."""
    saved = t84.PanoContextForcedDMDv8_4
    t84.PanoContextForcedDMDv8_4 = cls
    try:
        return fn(*args, **kwargs)
    finally:
        t84.PanoContextForcedDMDv8_4 = saved


def make86(floor_rollout=None, **kwargs):
    p, gt = with_base(PanoContextForcedDMDv8_6, t84.make, **kwargs)
    assert isinstance(p, PanoContextForcedDMDv8_6)
    p.mm3_lf_floor_rollout = floor_rollout
    if floor_rollout is not None:                     # what __init__ adds (the stand-ins skip __init__)
        p._last_mm3.update(mm3_roll_dmd_ratio=0.0, mm3_roll_cap_frac=0.0)
    return p, gt


def motion_case(alpha, seed=21):
    """real_seq [B, 1 + T, ...] (the real last history frame + T frames, a random walk) and a student whose every
    frame change, the seam step included, is alpha x the real one, from the same history frame."""
    g = torch.Generator().manual_seed(seed)
    first = torch.randn(B, 1, V, C, H, W, generator=g)
    real = torch.cat([first, first + torch.randn(B, T, V, C, H, W, generator=g).cumsum(1)], dim=1)
    fake = first + alpha * (real[:, 1:] - first)
    return real, first.clone(), fake


def term(p, alpha, floor):
    real, hist, fake = motion_case(alpha)
    x = fake.clone().requires_grad_(True)
    loss, r_lf, _, _ = p._motion_match3(x, hist, real, None, floor=floor)
    return loss, r_lf, x, hist, real


def main():
    # 0) knob off == v8.4 on every v8.4 (and v8.3 / v7) logic test
    with_base(PanoContextForcedDMDv8_6, t84.main)
    print("ok: v8.6 with mm3_lf_floor_rollout None passes all v8.4, v8.3 and v7 logic tests")

    # 1) the floor itself (HF part off so only the low-pass band acts)
    p, _ = make86()
    p.mm3_w_hf = 0.0
    assert p.mm3_lf_floor == 0.9
    for alpha in (0.9, 0.95, 1.0, 1.05):
        loss, r_lf, *_ = term(p, alpha, None)
        assert abs(float(r_lf.mean()) - alpha * alpha) < 1e-4, (alpha, float(r_lf.mean()))
    cases = {}
    for alpha in (math.sqrt(0.93), math.sqrt(0.975), 1.0, math.sqrt(1.1)):
        lo = float(term(p, alpha, None)[0])
        hi = float(term(p, alpha, 0.97)[0])
        cases[round(alpha * alpha, 3)] = (lo, hi)
        assert p.mm3_lf_floor == 0.9, "mm3_lf_floor restored after the call"
    assert cases[0.93][0] == 0.0 and cases[0.93][1] > 0.0, cases
    assert abs(cases[0.93][1] - (math.log(0.97) - math.log(0.93)) ** 2) < 1e-6, cases
    assert cases[0.975] == (0.0, 0.0) and max(cases[1.0]) < 1e-10, cases     # 1.0 only up to float rounding
    assert cases[1.1][0] > 0.0 and abs(cases[1.1][0] - cases[1.1][1]) < 1e-12, "upper side unchanged"
    loss, r_lf, x, hist, real = term(p, math.sqrt(0.93), 0.97)
    g = torch.autograd.grad(loss, x)[0]
    stepped = (x - 0.05 * g / g.norm() * x.norm()).detach()
    lf_before = change_stats(torch.cat([hist, x.detach()], 1), 4)[0] / change_stats(real, 4)[0]
    lf_after = change_stats(torch.cat([hist, stepped], 1), 4)[0] / change_stats(real, 4)[0]
    assert float(lf_after.mean()) > float(lf_before.mean()), (float(lf_before.mean()), float(lf_after.mean()))
    try:
        p._motion_match3(torch.zeros(B, T, V, C, H, W), torch.zeros(B, 2, V, C, H, W + 1),
                         torch.zeros(B, T + 1, V, C, H, W), None, floor=0.97)
        raise AssertionError("mismatched shapes must raise")
    except RuntimeError:
        pass
    assert p.mm3_lf_floor == 0.9, "mm3_lf_floor restored after a raising call"
    print(f"ok: floor 0.97 zero for low-pass ratio 0.975 / 1.0, {cases[0.93][1]:.5f} at 0.93 (floor 0.9: 0), upper "
          f"side unchanged at 1.1, a gradient step raises the change ({float(lf_before.mean()):.4f} -> "
          f"{float(lf_after.mean()):.4f}), mm3_lf_floor restored, also after a raise")

    # 1b) capped gradient (CapGrad caps the whole term): the same for floors 0.9 and 0.97 when only the low-pass floor
    #     part acts, different when the HF part is active too (Codex review 2026-10-16)
    real, hist, fake = motion_case(0.8)                                   # low-pass ratio 0.64: below both floors
    flick = fake + 0.3 * torch.randn(fake.shape, generator=torch.Generator().manual_seed(5))   # HF flicker

    def capped_grad(inp, floor, w_hf):
        p.mm3_w_hf = w_hf
        x = inp.clone().requires_grad_(True)
        loss, r_lf, r_hf, st = p._motion_match3(x, hist, real, 1e-6, floor=floor)
        (1000.0 * loss).backward()
        assert st["scale"] < 1.0, "the cap must bind in this check"
        return x.grad.double(), float(r_lf.mean()), float(r_hf.mean())

    def rel(a, b):
        return float((a - b).norm() / a.norm())
    g_a, lf_a, _ = capped_grad(fake, None, 0.0)
    g_b, _, _ = capped_grad(fake, 0.97, 0.0)
    g_c, lf_c, hf_c = capped_grad(flick, None, 1.0)
    g_d, _, _ = capped_grad(flick, 0.97, 1.0)
    p.mm3_w_hf = 0.0
    assert lf_a < 0.9 and lf_c < 0.9 and hf_c > 1.0, (lf_a, lf_c, hf_c)
    assert rel(g_a, g_b) < 1e-5, rel(g_a, g_b)
    assert rel(g_c, g_d) > 1e-3, rel(g_c, g_d)
    print(f"ok: capped gradient identical for floors 0.9 / 0.97 with only the LF floor active (rel diff "
          f"{rel(g_a, g_b):.1e}); with HF active too (hf ratio {hf_c:.2f}, lf {lf_c:.2f}) it differs by "
          f"{rel(g_c, g_d):.1%}")

    # 1c) constructor (parent constructor stubbed): knob on -> the two report keys exist from the first row; off ->
    #     no new keys, i.e. v8.4's report
    def parent_init(self, *args, **kwargs):
        self.mm3_lf_floor, self.should_save = 0.9, False
        self._last_mm3 = {"mm3_roll_on": 0.0}
    saved_init = PanoContextForcedDMDv8_4.__init__
    PanoContextForcedDMDv8_4.__init__ = parent_init
    try:
        on = PanoContextForcedDMDv8_6(mm3_lf_floor_rollout=0.97)
        off = PanoContextForcedDMDv8_6()
    finally:
        PanoContextForcedDMDv8_4.__init__ = saved_init
    assert on.mm3_lf_floor_rollout == 0.97 and on._last_mm3["mm3_roll_cap_frac"] == 0.0 \
        and on._last_mm3["mm3_roll_dmd_ratio"] == 0.0, on._last_mm3
    assert off.mm3_lf_floor_rollout is None and set(off._last_mm3) == {"mm3_roll_on"}, off._last_mm3
    print("ok: constructor adds mm3_roll_cap_frac / mm3_roll_dmd_ratio only when the knob is on")

    # 2) routing in a student step
    def prepare(q, k=6):
        """Deep stash, a recorder on _motion_match3 and a settable forced exit; returns (calls, exit holder)."""
        t84.deep_stash(q, k)
        q._stash["roll"][k - 1] = q._stash["roll"][k - 1] * 0.8              # not equal to the real window
        q.random_exit_step = False
        calls, force = [], {"exit": 3}
        q._caps = []                                     # cap argument of every call
        orig_mm = q._motion_match3

        def mm(*a, **kw):
            calls.append(kw.get("floor"))
            q._caps.append(a[3] if len(a) > 3 else kw.get("cap"))
            return orig_mm(*a, **kw)
        q._motion_match3 = mm
        orig = q._run_generator

        def run(*a, **kk):
            out = orig(*a, **kk)
            q._last_exit_idx = force["exit"]
            return out
        q._run_generator = run
        return calls, force

    def student_run(force_exit, floor_rollout, cls=PanoContextForcedDMDv8_6, k=6, cap_rollout=None):
        if cls is PanoContextForcedDMDv8_6:
            q, _ = make86(floor_rollout)
            q.motion_match_cap_rollout = cap_rollout
        else:
            q, _ = t84.make()
            assert type(q).__mro__[1] is PanoContextForcedDMDv8_4
        calls, force = prepare(q, k)
        force["exit"] = force_exit
        t7.student(q, 5)
        return q, calls

    q, calls = student_run(3, 0.97)
    assert calls == [0.97, None] and q._last_mm3["mm3_roll_on"] == 1.0, calls
    roll_cap, roll_ratio = q._last_mm3["mm3_roll_cap_frac"], q._last_mm3["mm3_roll_dmd_ratio"]
    assert roll_cap in (0.0, 1.0) and roll_ratio > 0.0 and math.isfinite(roll_ratio), (roll_cap, roll_ratio)
    q, calls = student_run(1, 0.97)
    assert calls == [None] and q._last_mm3["mm3_roll_on"] == 0.0, calls
    # the same instance: a final-step exit refreshes the two keys, a later early exit keeps them
    q, _ = make86(0.97)
    calls, force = prepare(q)
    t7.student(q, 5)
    kept = (q._last_mm3["mm3_roll_cap_frac"], q._last_mm3["mm3_roll_dmd_ratio"])
    assert kept[1] > 0.0 and calls == [0.97, None], (kept, calls)
    force["exit"] = 1
    t7.student(q, 6)
    assert calls == [0.97, None, None] and q._last_mm3["mm3_roll_on"] == 0.0, calls
    assert (q._last_mm3["mm3_roll_cap_frac"], q._last_mm3["mm3_roll_dmd_ratio"]) == kept, "carried, not reset"
    q_off, calls = student_run(3, None)
    assert calls == [None, None], calls
    assert "mm3_roll_cap_frac" not in q_off._last_mm3, "knob off: no new report keys (v8.4's report)"
    q_84, _ = student_run(3, None, cls=PanoContextForcedDMDv8_4)
    assert torch.equal(q_off.gen_param.grad, q_84.gen_param.grad), "knob off: same gradient as v8.4"
    q_on, _ = student_run(3, 1.0)
    print(f"ok: final-step exit -> rollout call floor 0.97, first-window call default; early exit -> first window "
          f"only; knob off -> both default and the gradient equals v8.4's; rollout term cap_frac {roll_cap:.0f}, "
          f"grad/DMD {roll_ratio:.3f}; floor 1.0 grad {float(q_on.gen_param.grad):+.4e} vs off "
          f"{float(q_off.gen_param.grad):+.4e} (equal here: the stand-in's term is capped with only the LF floor acting, see 1b)")

    # 2b) report keys with the knob on: identical on every row, incl. the two v8.6 keys
    q, gtq = make86(0.97)
    rows = [t7.critic(q, gtq, k) for k in range(2)]
    t7.student(q, 3)
    rows.append(t7.critic(q, gtq, 3))
    keys = set(rows[0])
    assert {"mm3_roll_cap_frac", "mm3_roll_dmd_ratio"} <= keys and all(set(r) == keys for r in rows), keys
    assert all(all(v == v for v in r.values()) for r in rows), "no NaN in the report"
    print("ok: with the knob on, critic and student rows have identical keys incl. mm3_roll_cap_frac / "
          "mm3_roll_dmd_ratio; no NaN")

    # 2c) rollout cap (branch M): the rollout call gets motion_match_cap_rollout x DMD, the first-window call keeps
    #     motion_match_cap x DMD; with the knob off both calls get the same cap (branch L / v8.4)
    q_m, calls = student_run(3, 0.97, cap_rollout=1.0)
    c_roll, c_first = q_m._caps
    assert calls == [0.97, None] and c_roll and c_first, (calls, q_m._caps)
    assert abs(c_roll / c_first - 1.0 / 0.35) < 1e-9, (c_roll, c_first)
    q_l, calls = student_run(3, 0.97)
    assert q_l._caps[0] == q_l._caps[1], q_l._caps
    assert not torch.equal(q_m.gen_param.grad, q_l.gen_param.grad), "a larger cap changes the capped rollout term"
    q_e, calls = student_run(1, 0.97, cap_rollout=1.0)
    assert len(q_e._caps) == 1 and abs(q_e._caps[0] / c_first - 1.0) < 1e-9, "early exit: first window only, 0.35"
    print(f"ok: rollout cap {c_roll / c_first * 0.35:.2f} x DMD vs first window 0.35 x DMD with the knob on, equal "
          f"with it off; the gradient changes (grad {float(q_m.gen_param.grad):+.4e} vs {float(q_l.gen_param.grad):+.4e})")

    # 3) constructor bounds (the check runs before the parent constructor)
    for bad in (0.0, 1.2, -0.5):
        try:
            PanoContextForcedDMDv8_6(mm3_lf_floor_rollout=bad)
            raise AssertionError(f"floor {bad} must be rejected")
        except ValueError as e:
            assert "mm3_lf_floor_rollout" in str(e), e
    print("ok: mm3_lf_floor_rollout outside (0, 1] rejected")
    for bad in (0.0, -1.0):
        try:
            PanoContextForcedDMDv8_6(motion_match_cap_rollout=bad)
            raise AssertionError(f"cap {bad} must be rejected")
        except ValueError as e:
            assert "motion_match_cap_rollout" in str(e), e
    PanoContextForcedDMDv8_4.__init__ = parent_init
    try:
        cap_only = PanoContextForcedDMDv8_6(motion_match_cap_rollout=1.0)
    finally:
        PanoContextForcedDMDv8_4.__init__ = saved_init
    assert cap_only.motion_match_cap_rollout == 1.0 and "mm3_roll_cap_frac" in cap_only._last_mm3, cap_only._last_mm3
    print("ok: motion_match_cap_rollout <= 0 rejected; the cap knob alone also adds the two report keys")
    print("ALL V8.6 LOGIC TESTS PASSED")


if __name__ == "__main__":
    main()
