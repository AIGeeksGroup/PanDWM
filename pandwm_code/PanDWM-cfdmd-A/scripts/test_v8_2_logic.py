"""CPU logic test for pipeline_v8_2 (no GPU, no real model): python3 scripts/test_v8_2_logic.py

Reuses the stand-ins of test_v7_logic.py and checks:
  0. with the temporal heads and the motion match off it passes every v7 logic test unchanged;
  1. motion-match term: positive when the student changes more than the real frames, a small step down its gradient
     reduces the change; zero loss and zero gradient when the student changes less (never pushed below real);
     gradient at the student output capped;
  2. weight schedule: 0 before motion_match_from, linear ramp, then motion_match_weight;
  3. student step: the first-window pass runs for the motion match alone (GAN still in warm-up), the term reaches the
     student parameters, report keys include mm_*; with the weight 0 no first-window pass runs (as v7);
  4. with temporal heads on as well, a critic + student step runs and every row has the same keys.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_v7_logic as t7                                               # noqa: E402

from dwm_pano_cfdmd.pipeline import CONTEXT, WINDOW                      # noqa: E402
from dwm_pano_cfdmd.pipeline_v7 import PanoContextForcedDMDv7, latent_motion   # noqa: E402
from dwm_pano_cfdmd.pipeline_v8_2 import PanoContextForcedDMDv8_2        # noqa: E402

B, V, C, H, W = t7.B, t7.V, t7.C, t7.H, t7.W


def defaults(cls, temporal, weight):
    cls._v8_temporal = temporal
    cls.temporal_adv_from, cls.temporal_adv_ramp, cls._last_temporal_weight = 0, 0, 0.0
    cls.motion_match_weight, cls.motion_match_cap = weight, 0.35
    cls.motion_match_from, cls.motion_match_ramp = 0, 0
    return cls


class Off(PanoContextForcedDMDv8_2):
    pass


defaults(Off, False, 0.0)


def make(temporal=False, weight=1.0, **kwargs):
    cls = defaults(type("Case", (PanoContextForcedDMDv8_2,), {}), temporal, weight)
    t7.PanoContextForcedDMDv7 = cls
    try:
        p, gt = t7.make(**kwargs)
    finally:
        t7.PanoContextForcedDMDv7 = PanoContextForcedDMDv7
    p._last_mm = {"mm_ratio": 1.0, "mm_loss": 0.0, "mm_dmd_ratio": 0.0, "mm_cap_frac": 0.0}
    p._mm_stats, p._mm_w = None, 0.0
    return p, gt


def main():
    # 0) everything off == v7
    t7.PanoContextForcedDMDv7 = Off
    try:
        t7.main()
    finally:
        t7.PanoContextForcedDMDv7 = PanoContextForcedDMDv7
    print("ok: v8.2 with temporal heads and motion match off passes all v7 logic tests")

    # 1) the motion-match term
    p, _ = make()
    torch.manual_seed(5)
    gt = torch.randn(B, WINDOW, V, C, H, W).cumsum(dim=1) * 0.1             # smooth real motion
    noisy = (gt + 0.05 * torch.randn_like(gt)).requires_grad_(True)          # student: extra frame-to-frame change
    loss, ratio, stats = p._motion_match(noisy, gt, None)
    assert float(ratio.min()) > 1.0 and float(loss) > 0
    g = torch.autograd.grad(loss, noisy)[0]
    stepped = (noisy - 0.5 * g / g.norm() * noisy.norm() * 0.01).detach()
    _, ratio2, _ = p._motion_match(stepped, gt, None)
    assert float(ratio2.mean()) < float(ratio.mean()), "a gradient step reduces the excess change"
    smooth = gt.clone()
    smooth[:, CONTEXT:] = gt[:, CONTEXT:].mean(dim=1, keepdim=True) + 0.5 * (gt[:, CONTEXT:] - gt[:, CONTEXT:].mean(dim=1, keepdim=True))
    smooth.requires_grad_(True)
    loss_s, ratio_s, _ = p._motion_match(smooth, gt, None)
    assert float(ratio_s.max()) < 1.0 and float(loss_s) == 0.0
    assert float(torch.autograd.grad(loss_s, smooth)[0].abs().sum()) == 0.0, "never pushed below the real change"
    stats = {}
    x = noisy.detach().clone().requires_grad_(True)
    loss_c, _, stats = p._motion_match(x, gt, 1e-6)
    (1000.0 * loss_c).backward()
    assert abs(float(x.grad.norm()) - 1e-6) < 1e-9 and stats["scale"] < 1.0, "gradient at the output is capped"
    print(f"ok: motion match penalises only the excess change (ratio {float(ratio.mean()):.2f} -> "
          f"{float(ratio2.mean()):.2f} after one step; {float(ratio_s.mean()):.2f} gives 0), gradient capped")

    # 2) schedule
    p, _ = make(weight=2.0)
    p.motion_match_from, p.motion_match_ramp = 10, 4
    for step, want in {0: 0.0, 9: 0.0, 10: 0.5, 12: 1.5, 13: 2.0, 40: 2.0}.items():
        p._cur_step = step
        assert abs(p._motion_weight() - want) < 1e-9, (step, p._motion_weight())
    print("ok: motion-match weight 0 before motion_match_from, linear ramp, then the configured weight")

    # 3) student step with the GAN still in warm-up
    def student_run(weight):
        q, _ = make(weight=weight, warmup=(1000, 10))
        t7.student(q, 5)
        first = [c for c in q.calls["gen"] if c["tag"] == "w0"]
        return q, first

    q, first = student_run(1.0)
    assert len(first) == 4 and q._last_gan_weight == 0.0, "first-window pass runs for the motion match alone"
    assert q._last_mm["mm_ratio"] != 1.0 and q.calls["F"] == [], "no discriminator forward in the GAN warm-up"
    g_on = None if q.gen_param.grad is None else q.gen_param.grad.clone()
    q0, first0 = student_run(0.0)
    assert first0 == [], "weight 0: no first-window pass, as v7"
    g_off = None if q0.gen_param.grad is None else q0.gen_param.grad.clone()
    if float(q._last_mm["mm_loss"]) > 0:
        assert g_on is not None and g_off is not None and not torch.allclose(g_on, g_off), "term reaches the student"
    print(f"ok: student step runs the first window for the motion match (mm_ratio {q._last_mm['mm_ratio']:.2f}, "
          f"mm_loss {q._last_mm['mm_loss']:.3f}); weight 0 reproduces v7")

    # 4) temporal heads on as well: critic rows keep the same keys
    q, gtq = make(temporal=True, weight=1.0)
    rows = [t7.critic(q, gtq, k) for k in range(2)]
    t7.student(q, 3)
    rows.append(t7.critic(q, gtq, 3))
    keys = set(rows[0])
    assert all(set(r) == keys for r in rows) and {"mm_ratio", "mm_loss", "mm_dmd_ratio", "mm_cap_frac", "mm_weight",
                                                  "temporal_weight", "dt_gap"} <= keys, keys
    assert all(all(v == v for v in r.values()) for r in rows)
    print("ok: with temporal heads and motion match on, critic and student steps run; identical report keys")
    print("ALL V8.2 LOGIC TESTS PASSED")


if __name__ == "__main__":
    main()
