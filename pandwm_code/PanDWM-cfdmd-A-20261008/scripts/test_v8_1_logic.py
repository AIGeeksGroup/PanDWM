"""CPU logic test for pipeline_v8_1 (no GPU, no real model): python3 scripts/test_v8_1_logic.py

Reuses the stand-ins of test_v7_logic.py and checks:
  0. with the temporal heads off it passes every v7 logic test unchanged;
  1. temporal heads have no input LayerNorm; spatial heads first in the parameter order (v7-tolerant resume);
  2. the temporal input keeps the size of the change: twice the change gives (about) twice the input norm, where
     v8's LayerNorm-on-the-difference gave the same; per-frame normalisation makes a pure rescale of a frame no change;
  3. warm-up weight: 0 before temporal_adv_from, linear ramp, then 1;
  4. student term with w_t = 0 equals v7's spatial-only term and the temporal heads give no gradient; at w_t = 1 it
     equals the mean over all heads (v8's form) and the temporal heads do contribute;
  5. critic trains the temporal heads during the warm-up; report keys = v8's + temporal_weight, no NaN;
  6. a v7 head file and head-optimizer state load (temporal heads start fresh).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_v7_logic as t7                                               # noqa: E402

from dwm_pano_cfdmd.pipeline import CONTEXT, WINDOW                      # noqa: E402
from dwm_pano_cfdmd.pipeline_v7 import PanoContextForcedDMDv7, gen_adv   # noqa: E402
from dwm_pano_cfdmd.pipeline_v8_1 import PanoContextForcedDMDv8_1        # noqa: E402
from dwm_pano_cfdmd.pipeline_v8 import TolerantAdamW                     # noqa: E402

B, V, C, H, W = t7.B, t7.V, t7.C, t7.H, t7.W
HP, WP, CH = t7.HP, t7.WP, t7.CH
V8_KEYS = {"critic_loss", "disc_loss", "d_real", "d_fake", "disc_real_acc", "disc_fake_acc", "d_gap_lo",
           "d_gap_hi", "disc1_loss", "d1_real", "d1_fake", "disc1_real_acc", "disc1_fake_acc", "hf_ratio_w1",
           "motion_ratio_w1", "hf_ratio_w2", "gan_dmd_ratio", "gan_cap_frac", "adv_first", "gt_reg", "gan_weight",
           "mem_peak_gb", "dt_real_acc", "dt_fake_acc", "dt_gap", "dt1_real_acc", "dt1_fake_acc", "dt1_gap"}


class Off(PanoContextForcedDMDv8_1):
    _v8_temporal = False
    temporal_adv_from, temporal_adv_ramp, _last_temporal_weight = 0, 0, 0.0


class On(PanoContextForcedDMDv8_1):
    _v8_temporal = True
    temporal_adv_from, temporal_adv_ramp, _last_temporal_weight = 0, 0, 0.0


def make(cls, **kwargs):
    t7.PanoContextForcedDMDv7 = cls
    try:
        return t7.make(**kwargs)
    finally:
        t7.PanoContextForcedDMDv7 = PanoContextForcedDMDv7


def main():
    # 0) temporal heads off == v7
    t7.PanoContextForcedDMDv7 = Off
    try:
        t7.main()
    finally:
        t7.PanoContextForcedDMDv7 = PanoContextForcedDMDv7
    print("ok: v8.1 with disc_temporal=False passes all v7 logic tests")

    # 1) heads
    p, _ = make(On)
    assert list(p.disc_head.keys()) == ["1", "3", "t1", "t3"]
    assert all(isinstance(p.disc_head[k][0], torch.nn.Linear) for k in ("t1", "t3")), "no LayerNorm on the change"
    assert all(isinstance(p.disc_head[k][0], torch.nn.LayerNorm) for k in ("1", "3"))
    names = [n for n, _ in p.disc_head.named_parameters()]
    first_t = next(i for i, n in enumerate(names) if n.startswith("t"))
    assert all(n.startswith("t") for n in names[first_t:]) and type(p.disc_optimizer) is TolerantAdamW
    print("ok: temporal heads = Linear -> SiLU -> Linear after the spatial heads")

    # 2) the change keeps its size; a pure per-frame rescale is no change
    torch.manual_seed(3)
    base = torch.randn(B, 1, V, HP * WP, CH)
    direction = torch.randn(B, 1, V, HP * WP, CH)
    for head in ("t1", "t3"):
        p.disc_head[head].forward = (lambda x: x.norm(dim=-1))

    def change_norm(step, scale=None):
        k = torch.arange(WINDOW).view(1, WINDOW, 1, 1, 1).float()
        h = base + step * k * direction
        if scale is not None:
            h = base * scale.view(1, WINDOW, 1, 1, 1)
        logits = p._token_logits({1: h.flatten(0, 2), 3: h.flatten(0, 2)}, (B, WINDOW, V, C, H, W), True)
        return logits[2].mean()

    small, large = change_norm(0.01), change_norm(0.02)
    assert 1.9 < float(large / small) < 2.1, float(large / small)
    rescaled = change_norm(0.0, scale=torch.linspace(0.5, 2.0, WINDOW))
    assert float(rescaled) < 1e-3, float(rescaled)
    print(f"ok: temporal input keeps the size of the change (x2 change -> x{float(large / small):.2f}); "
          f"rescaling a frame is not a change")

    # 3) warm-up weight
    p, _ = make(On)
    p.temporal_adv_from, p.temporal_adv_ramp = 10, 5
    want = {0: 0.0, 9: 0.0, 10: 0.2, 12: 0.6, 14: 1.0, 50: 1.0}
    for step, w in want.items():
        p._cur_step = step
        assert abs(p._temporal_weight() - w) < 1e-9, (step, p._temporal_weight())
    p.temporal_adv_from, p.temporal_adv_ramp = 0, 0
    p._cur_step = 0
    assert p._temporal_weight() == 1.0
    print("ok: temporal weight 0 before temporal_adv_from, linear ramp, then 1")

    # 4) student term
    p, _ = make(On)
    x0 = torch.randn(B, WINDOW, V, C, H, W, requires_grad=True)

    def term(step):
        p._cur_step = step
        torch.manual_seed(1)
        adv, _ = p._adv_term(x0, lambda xs: p._rollout_inputs(xs, 3), None, {})
        torch.manual_seed(1)
        noisy, ts, _, cond, drop = p._rollout_inputs([x0], 3)
        feats, _ = p._disc_features(noisy[0], ts, cond)
        logits = p._token_logits(feats, noisy[0].shape, drop)
        return adv, logits

    p.temporal_adv_from, p.temporal_adv_ramp = 100, 10
    adv0, logits = term(50)
    assert p._last_temporal_weight == 0.0
    assert torch.allclose(adv0, gen_adv(logits[:2]), atol=1e-6), "w_t = 0 -> v7's spatial-only term"
    g_warm = torch.autograd.grad(adv0, x0)[0]
    adv1, logits = term(200)
    assert p._last_temporal_weight == 1.0 and torch.allclose(adv1, gen_adv(logits), atol=1e-6), "w_t = 1 -> v8 form"
    with torch.no_grad():
        for name in ("t1", "t3"):
            p.disc_head[name][-1].weight.zero_()
            p.disc_head[name][-1].bias.zero_()
    adv_z, _ = term(50)
    g_zero = torch.autograd.grad(adv_z, x0)[0]
    assert torch.allclose(g_warm, g_zero, atol=1e-7), "temporal heads must give no gradient during the warm-up"
    p, _ = make(On)
    adv_on, _ = term(200)
    g_on = torch.autograd.grad(adv_on, x0)[0]
    with torch.no_grad():
        for name in ("t1", "t3"):
            p.disc_head[name][-1].weight.zero_()
            p.disc_head[name][-1].bias.zero_()
    adv_off, _ = term(200)
    g_off = torch.autograd.grad(adv_off, x0)[0]
    assert not torch.allclose(g_on, g_off), "after the warm-up the temporal heads add to the gradient"
    print("ok: student term = v7's while w_t = 0 (no temporal gradient), v8's form at w_t = 1")

    # 5) critic during the warm-up
    p, gt = make(On)
    p.temporal_adv_from, p.temporal_adv_ramp = 100, 10
    grads = {}

    def step_head(scale):
        for name, head in p.disc_head.items():
            grads[name] = sum(float(q.grad.abs().sum()) for q in head.parameters() if q.grad is not None)
        p.disc_optimizer.zero_grad()

    p._step_disc_head = step_head
    rows = [t7.critic(p, gt, k) for k in range(3)]
    assert all(set(r) == V8_KEYS | {"temporal_weight"} for r in rows), set(rows[0]) ^ (V8_KEYS | {"temporal_weight"})
    assert all(all(v == v for v in r.values()) for r in rows)
    assert all(grads[k] > 0 for k in ("1", "3", "t1", "t3")), grads
    t7.student(p, 50)
    assert p._last_temporal_weight == 0.0 and all(q.grad is None for q in p.disc_head.parameters())
    print("ok: temporal heads train during the warm-up; report has temporal_weight; student step clears heads")

    # 6) resume from v7
    p7, _ = t7.make()
    saved = {k: v + 1.0 for k, v in p7.disc_head.state_dict().items()}
    p8, _ = make(On)
    before = {k: v.clone() for k, v in p8.disc_head.state_dict().items() if k.startswith("t")}
    p8.disc_head.load_state_dict(saved)
    now = p8.disc_head.state_dict()
    assert all(torch.equal(now[k], saved[k]) for k in saved) and all(torch.equal(now[k], before[k]) for k in before)
    for q in p7.disc_head.parameters():
        q.grad = torch.randn_like(q)
    p7.disc_optimizer.step()
    p8.disc_optimizer.load_state_dict(p7.disc_optimizer.state_dict())
    for q in p8.disc_head.parameters():
        q.grad = torch.randn_like(q)
    p8.disc_optimizer.step()
    print("ok: v7 head file and head-optimizer state load into v8.1")
    print("ALL V8.1 LOGIC TESTS PASSED")


if __name__ == "__main__":
    main()
