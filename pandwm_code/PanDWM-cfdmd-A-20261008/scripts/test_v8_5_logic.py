"""CPU logic test for pipeline_v8_5 (no GPU, no real model): python3 scripts/test_v8_5_logic.py

  0. the v8.4 logic tests pass with the v8.5 class in place of v8.4 (defaults = v8.4);
  1. cfg_threshold: held, linear, held; a zero-length decay switches at decay_to;
  2. with the default threshold (1001 > every t) the v8.5 gradient equals v8.4's bit for bit; with tau 0 it equals a
     gradient computed from the conditional real score alone; with tau 500 and per-sample t 300 / 700 only the first
     sample is guided; cfg_frac and cfg_tau are recorded and reported.
"""
import contextlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_v8_4_logic as t84                                            # noqa: E402

from dwm_pano_cfdmd.pipeline import CONTEXT, WINDOW                      # noqa: E402
from dwm_pano_cfdmd.pipeline_v8_4 import PanoContextForcedDMDv8_4        # noqa: E402
from dwm_pano_cfdmd.pipeline_v8_5 import PanoContextForcedDMDv8_5, cfg_threshold   # noqa: E402

V, C, H, W = t84.V, t84.C, t84.H, t84.W


def stand_in(cls, b, tau_start=1001.0):
    q = object.__new__(cls)
    q.fake_score_causal = q.real_score_causal = True
    q._add_temporal_attention_mask = lambda cond, shape: cond.__setitem__("mask", tuple(shape[:3]))
    q.get_autocast_context = contextlib.nullcontext
    q.model_dtype, q.real_guidance_scale, q.mm3_pool = torch.float32, 2.0, 4
    q.fake_score_wrapper = lambda noisy, ts, **cond: ([noisy * 0.3 + 0.1], None, None)
    half = torch.cat([torch.full((b,), 0.2), torch.full((b,), 0.25)]).view(2 * b, 1, 1, 1, 1, 1)
    q.real_score_wrapper = lambda noisy, ts, **cond: ([t84.blur(noisy, 0.7) * half - 0.05], None, None)
    q._v84_reset()
    q._cur_step = 0
    if cls is PanoContextForcedDMDv8_5:
        q.cfg_tau_start, q.cfg_tau_end, q.cfg_decay_from, q.cfg_decay_to = tau_start, tau_start, 0, 0
        q._v85_last = {"cfg_tau": tau_start, "cfg_frac": 1.0}
    return q


def inputs(b, t_values):
    noisy = torch.randn(b, WINDOW, V, C, H, W, generator=torch.Generator().manual_seed(12))
    ts = torch.zeros(b, WINDOW, V)
    for i, t in enumerate(t_values):
        ts[i, CONTEXT:] = t
    sig = (ts[:, :, :1] / 1000.0).view(b, WINDOW, 1, 1, 1, 1).expand(b, WINDOW, V, 1, 1, 1).clone()
    sig[:, CONTEXT:] = sig[:, CONTEXT:].clamp(min=0.05)
    x0d = torch.randn(b, WINDOW, V, C, H, W, generator=torch.Generator().manual_seed(13))
    mask = torch.ones(b, WINDOW, V, dtype=torch.bool)
    mask[:, :CONTEXT] = False
    return noisy, ts, sig, x0d, mask


def main():
    # 0) v8.4 suite with v8.5 in place of v8.4
    saved = t84.PanoContextForcedDMDv8_4
    t84.PanoContextForcedDMDv8_4 = PanoContextForcedDMDv8_5
    try:
        t84.main()
    finally:
        t84.PanoContextForcedDMDv8_4 = saved
    print("ok: the v8.4 logic tests pass with the v8.5 class (defaults = v8.4)")

    # 1) threshold schedule
    assert cfg_threshold(100, 1000, 0, 200, 300) == 1000
    assert cfg_threshold(200, 1000, 0, 200, 300) == 1000
    assert abs(cfg_threshold(250, 1000, 0, 200, 300) - 500) < 1e-9
    assert cfg_threshold(300, 1000, 0, 200, 300) == 0 and cfg_threshold(9999, 1000, 0, 200, 300) == 0
    assert cfg_threshold(5, 1001, 1001, 0, 0) == 1001
    assert cfg_threshold(10, 1000, 0, 10, 10) == 1000 and cfg_threshold(11, 1000, 0, 10, 10) == 0
    print("ok: cfg_threshold held / linear / held, zero-length decay switches after decay_from")

    # 2) gradient
    b = 2
    noisy, ts, sig, x0d, mask = inputs(b, [300.0, 700.0])
    want = PanoContextForcedDMDv8_4._kl_grad_with_history(stand_in(PanoContextForcedDMDv8_4, b), noisy, ts, sig, x0d,
                                                          {"tag": "c"}, {"tag": "u"}, mask)
    q = stand_in(PanoContextForcedDMDv8_5, b)
    got = q._kl_grad_with_history(noisy, ts, sig, x0d, {"tag": "c"}, {"tag": "u"}, mask)
    assert torch.equal(want, got), "default threshold must be v8.4 exactly"
    assert q._v85_last["cfg_frac"] == 1.0

    q0 = stand_in(PanoContextForcedDMDv8_5, b, tau_start=0.0)
    g0 = q0._kl_grad_with_history(noisy, ts, sig, x0d, {"tag": "c"}, {"tag": "u"}, mask)
    q1 = stand_in(PanoContextForcedDMDv8_4, b)
    q1.real_guidance_scale = 0.0
    g_ref = PanoContextForcedDMDv8_4._kl_grad_with_history(q1, noisy, ts, sig, x0d, {"tag": "c"}, {"tag": "u"}, mask)
    assert torch.allclose(g0, g_ref, atol=1e-6), "tau 0 = no guidance (conditional real score)"
    assert q0._v85_last["cfg_frac"] == 0.0 and not torch.allclose(g0, want)

    qm = stand_in(PanoContextForcedDMDv8_5, b, tau_start=500.0)
    gm = qm._kl_grad_with_history(noisy, ts, sig, x0d, {"tag": "c"}, {"tag": "u"}, mask)
    assert torch.allclose(gm[0], want[0], atol=1e-6) and torch.allclose(gm[1], g_ref[1], atol=1e-6), \
        "t 300 < 500 guided, t 700 not"
    assert qm._v85_last["cfg_frac"] == 0.5
    print("ok: default = v8.4 bit for bit; tau 0 = conditional score only; tau 500 guides t 300 and not t 700; "
          "cfg_frac recorded")
    print("ALL_V8_5_TESTS_PASS")


if __name__ == "__main__":
    main()
