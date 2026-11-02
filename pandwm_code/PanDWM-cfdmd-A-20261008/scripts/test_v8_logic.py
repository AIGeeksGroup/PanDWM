"""CPU logic test for pipeline_v8 (no GPU, no real model): python3 scripts/test_v8_logic.py

Reuses the stand-ins of test_v7_logic.py and checks:
  0. v8 with the temporal heads off passes every v7 test unchanged (it is v7);
  1. the parent's head assignment yields spatial heads first, then one temporal head per block;
  2. temporal logits: shapes (one fewer frame), the change is taken between consecutive target frames of the
     same view and place; spatial logits are still each frame's own tokens;
  3. the temporal heads take part in the student's adversarial gradient;
  4. critic: temporal heads get gradients, dt_* keys on every row, no NaN; noise-history fallback runs;
  5. student: warm-up, both terms, cap and cleared head gradients as in v7;
  6. resume from v7: a v7 head file loads into the v8 heads (temporal heads keep their values, other
     mismatches still raise) and a v7 head-optimizer state loads into the v8 head optimizer.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_v7_logic as t7                                               # noqa: E402

from dwm_pano_cfdmd.pipeline import CONTEXT, WINDOW                      # noqa: E402
from dwm_pano_cfdmd.pipeline_v7 import PanoContextForcedDMDv7            # noqa: E402
from dwm_pano_cfdmd.pipeline_v8 import PanoContextForcedDMDv8, TolerantAdamW   # noqa: E402

B, V, C, H, W = t7.B, t7.V, t7.C, t7.H, t7.W
HP, WP, CH = t7.HP, t7.WP, t7.CH
V7_KEYS = {"critic_loss", "disc_loss", "d_real", "d_fake", "disc_real_acc", "disc_fake_acc", "d_gap_lo",
           "d_gap_hi", "disc1_loss", "d1_real", "d1_fake", "disc1_real_acc", "disc1_fake_acc", "hf_ratio_w1",
           "motion_ratio_w1", "hf_ratio_w2", "gan_dmd_ratio", "gan_cap_frac", "adv_first", "gt_reg", "gan_weight",
           "mem_peak_gb"}
V8_KEYS = V7_KEYS | {"dt_real_acc", "dt_fake_acc", "dt_gap", "dt1_real_acc", "dt1_fake_acc", "dt1_gap"}


class V8Off(PanoContextForcedDMDv8):
    _v8_temporal = False


class V8On(PanoContextForcedDMDv8):
    _v8_temporal = True


def make(cls, **kwargs):
    t7.PanoContextForcedDMDv7 = cls
    try:
        return t7.make(**kwargs)
    finally:
        t7.PanoContextForcedDMDv7 = PanoContextForcedDMDv7


def main():
    # 0) temporal heads off == v7
    t7.PanoContextForcedDMDv7 = V8Off
    try:
        t7.main()
    finally:
        t7.PanoContextForcedDMDv7 = PanoContextForcedDMDv7
    print("ok: v8 with disc_temporal=False passes all v7 logic tests")

    # 1) heads
    p, _ = make(V8On)
    assert list(p.disc_head.keys()) == ["1", "3", "t1", "t3"], list(p.disc_head.keys())
    names = [n for n, _ in p.disc_head.named_parameters()]
    half = len(names) // 2
    assert all(not n.startswith("t") for n in names[:half]) and all(n.startswith("t") for n in names[half:])
    assert type(p.disc_optimizer) is TolerantAdamW
    print("ok: spatial heads first, then one temporal head per block; head optimizer is TolerantAdamW")

    # 2) temporal logits: feature value 10 * frame + view, so the change between consecutive frames is 10
    h = torch.zeros(B, WINDOW, V, HP * WP, CH)
    for k in range(WINDOW):
        for v in range(V):
            h[:, k, v, :, 0] = 10.0 * k + v
    h = h.flatten(0, 2)
    p.disc_head = "reset"
    for head in p.disc_head.values():
        head.forward = (lambda x: x[..., 0])
    logits = p._token_logits({1: h, 3: h}, (B, WINDOW, V, C, H, W), drop_context=True)
    assert len(logits) == 4
    sp, tp = logits[:2], logits[2:]
    assert sp[0].shape == (B * (WINDOW - CONTEXT) * V, HP // 2, WP // 2), sp[0].shape
    assert tp[0].shape == (B * (WINDOW - CONTEXT - 1) * V, HP // 2, WP // 2), tp[0].shape
    assert all(torch.allclose(x, torch.full_like(x, 10.0)) for x in tp), "change within a view, frame to frame"
    first = sp[0].unflatten(0, (B, WINDOW - CONTEXT, V))
    assert torch.allclose(first[:, 0, 1], torch.full_like(first[:, 0, 1], 10.0 * CONTEXT + 1))
    h16 = h.unflatten(0, (B, WINDOW, V))[:, :16].flatten(0, 2)
    logits = p._token_logits({1: h16, 3: h16}, (B, 16, V, C, H, W), drop_context=False)
    assert logits[2].shape[0] == B * 15 * V and torch.allclose(logits[2], torch.full_like(logits[2], 10.0))
    print("ok: temporal logits = change of each pooled token from the previous target frame (same view)")

    # 3) temporal heads contribute to the student's adversarial gradient
    p, _ = make(V8On)
    x0 = torch.randn(B, WINDOW, V, C, H, W, requires_grad=True)

    def adv_grad():
        torch.manual_seed(1)
        adv, _ = p._adv_term(x0, lambda xs: p._rollout_inputs(xs, 3), None, {})
        return torch.autograd.grad(adv, x0)[0]

    g_on = adv_grad()
    with torch.no_grad():
        for name, head in p.disc_head.items():
            if name.startswith("t"):
                head[-1].weight.zero_()
                head[-1].bias.zero_()
    g_spatial = adv_grad()
    assert torch.isfinite(g_on).all() and not torch.allclose(g_on, g_spatial), "temporal heads give a gradient"
    print("ok: the temporal heads add to the adversarial gradient reaching the student output")

    # 4) critic
    p, gt = make(V8On)
    grads = {}

    def step_head(scale):
        for name, head in p.disc_head.items():
            grads[name] = sum(float(q.grad.abs().sum()) for q in head.parameters() if q.grad is not None)
        p.disc_optimizer.zero_grad()

    p._step_disc_head = step_head
    rows = [t7.critic(p, gt, k) for k in range(4)]
    assert all(set(r) == V8_KEYS for r in rows), set(rows[0]) ^ V8_KEYS
    assert all(all(v == v for v in r.values()) for r in rows), "no NaN in the report"
    assert all(grads[name] > 0 for name in ("1", "3", "t1", "t3")), grads
    assert all(0.0 <= rows[0][k] <= 1.0 for k in ("dt_real_acc", "dt_fake_acc", "dt1_real_acc", "dt1_fake_acc"))
    p, gt = make(V8On, rollout_input="noise")
    rep = t7.critic(p, gt, 0)
    assert set(rep) == V8_KEYS
    p, gt = make(V8On, pairs=("rollout",))
    rep = t7.critic(p, gt, 0)
    assert "dt_real_acc" in rep and "dt1_real_acc" not in rep and "disc1_loss" not in rep
    print("ok: critic trains spatial and temporal heads, dt_* keys on every row, noise-history and "
          "rollout-only configurations run")

    # 5) student
    p, gt = make(V8On, warmup=(5, 10))
    t7.student(p, 2)
    assert p.calls["F"] == [] and p._last_gan_weight == 0.0
    t7.student(p, 9)
    assert [c["tag"] for c in p.calls["F"]] == ["trunc", "w0"]
    assert all(q.grad is None for q in p.disc_head.parameters()), "student step must clear head gradients"
    p, gt = make(V8On, big_gan=True, g_weight=1.0)
    t7.student(p, 3)
    assert p._last_cap_frac == 1.0 and p._last_gan_ratio > 0.5
    print("ok: student step as in v7 (warm-up, both terms, cap, head gradients cleared)")

    # 6) resume from a v7 run
    p7, _ = t7.make()
    assert type(p7).__name__ == "PanoContextForcedDMDv7"
    saved = {k: v + 1.0 for k, v in p7.disc_head.state_dict().items()}
    p8, _ = make(V8On)
    temporal_before = {k: v.clone() for k, v in p8.disc_head.state_dict().items() if k.startswith("t")}
    p8.disc_head.load_state_dict(saved)
    now = p8.disc_head.state_dict()
    assert all(torch.equal(now[k], saved[k]) for k in saved)
    assert all(torch.equal(now[k], temporal_before[k]) for k in temporal_before)
    bad = dict(saved)
    bad["9.0.weight"] = torch.zeros(1)
    try:
        p8.disc_head.load_state_dict(bad)
        raise AssertionError("an unexpected key must still raise")
    except RuntimeError:
        pass
    p8.disc_head.load_state_dict(p8.disc_head.state_dict())             # a v8 head file loads strictly
    for q in p7.disc_head.parameters():
        q.grad = torch.randn_like(q)
    p7.disc_optimizer.step()
    s7 = p7.disc_optimizer.state_dict()
    n7 = len(list(p7.disc_head.parameters()))
    p8.disc_optimizer.load_state_dict(s7)
    params8 = list(p8.disc_head.parameters())
    for i, q in enumerate(params8):
        st = p8.disc_optimizer.state.get(q, {})
        if i < n7:
            assert torch.equal(st["exp_avg_sq"], s7["state"][i]["exp_avg_sq"]), i
        else:
            assert not st, i
    assert p8.disc_optimizer.param_groups[0]["lr"] == s7["param_groups"][0]["lr"]
    for q in params8:
        q.grad = torch.randn_like(q)
    p8.disc_optimizer.step()
    assert all(p8.disc_optimizer.state[q] for q in params8)
    p8.disc_optimizer.load_state_dict(p8.disc_optimizer.state_dict())   # a v8 state loads as usual
    print("ok: v7 head file and v7 head-optimizer state load into v8 (temporal heads start fresh)")
    print("ALL V8 LOGIC TESTS PASSED")


if __name__ == "__main__":
    main()
