"""CPU logic test of the ablation switches on pipeline_v8_4 (2026-10-21; no GPU, no real model).

Reuses the stand-ins of PanDWM-cfdmd-A-20261008/scripts/test_v7_logic.py / test_v8_3_logic.py / test_v8_4_logic.py
(run with PYTHONPATH=<cfdmd>/src:<cfdmd>/scripts:<demo>/vendor:<demo>/src).  Each case runs one student step and one
critic step of the real v8.4 code on tiny tensors and records which parts ran:
  A control (temporal heads on, w_t = 1, motion match on, GAN on): adversarial F forwards in the student step (rollout +
    first pair), first-window pass, mm3 term, 3 spatial + 3 temporal heads, critic trains heads on both pairs.
  B use_gan False (both GAN weights 0): no adversarial F forward, no head step; the first-window pass and the mm3 term
    still run; the critic step is flow matching only (one F forward with grad).
  D disc_temporal False: everything of A except the temporal heads (3 spatial heads, temporal weight 0).
  C context_forcing_prob 0 emulated by the stash such a step leaves (gt present, no rollout: no cond0 / roll): the
    first-window pass (mm3 first-window term and first adversarial pair) and the critic's first pair do NOT run.
    This is the coupling that makes context_forcing_prob 0 more than one switch.
  C knob (2026-10-21, option (b): dwm_pano_cfdmd_abl.pipeline_v8_4r.PanoContextForcedDMDv8_4R), on FULL train steps
    (pipeline.py train_step through v8.4 / v7 / v5 / resumable, rollout loop, student step, critic step; global step 305,
    a student step, k drawn as in training):
    (i)  recorded_history False == v8.4 bit for bit: histories, conditions, every student and F input tensor, the
         rollout windows, gradients of the student and F stand-ins, report row, pipeline-generator and torch RNG states;
    (ii) recorded_history True vs the control: same k, trained-window conditions, real window and first-window
         conditions; the trained window's history = the recorded frames 16k..16k+2 (control: the student's own
         rollout); window 1 still sampled (roll[0] bit-equal to the control's), windows 2..k not; the student step still
         has the first-window pass, the mm3 first-window term and both adversarial forwards; the critic the same 5 F
         forwards (1 with grad) and 1 head step.
Prints a per-case table and ABLATION_SWITCH_TESTS PASSED.
"""
import sys
import time

import torch

import test_v7_logic as t7                                              # noqa: E402  (patches broadcast)
import test_v8_3_logic as t83                                           # noqa: E402
import test_v8_4_logic as t84                                           # noqa: E402

from dwm_pano_cfdmd.pipeline_v7 import PanoContextForcedDMDv7           # noqa: E402
from dwm_pano_cfdmd.pipeline_v8_4 import PanoContextForcedDMDv8_4      # noqa: E402
import dwm_pano_cfdmd.pipeline as P                                     # noqa: E402
from dwm_pano_cfdmd.pipeline import CONTEXT, STRIDE                     # noqa: E402
from dwm_pano_cfdmd_abl.pipeline_v8_4r import PanoContextForcedDMDv8_4R   # noqa: E402

FRAMES = CONTEXT + STRIDE * 11
GT = torch.randn(t7.B, FRAMES, t7.V, t7.C, t7.H, t7.W, generator=torch.Generator().manual_seed(21))
STEP = 305                                     # a student step (305 % 5 == 0); 306 is no checkpoint step

N_LAYERS = len(t7.LAYERS)


def build(temporal=True):
    cls = t83.defaults(type("Abl84", (PanoContextForcedDMDv8_4,), {}), temporal, 1.0)
    t84.defaults84(cls)
    t7.PanoContextForcedDMDv7 = cls
    try:
        p, gt = t7.make(warmup=(0, 1))
    finally:
        t7.PanoContextForcedDMDv7 = PanoContextForcedDMDv7
    p._last_mm = {"mm_ratio": 1.0, "mm_loss": 0.0, "mm_dmd_ratio": 0.0, "mm_cap_frac": 0.0}
    p._mm_stats, p._mm_w = None, 0.0
    p._last_mm3 = {"mm3_hf_ratio": 1.0, "mm3_lf_ratio": 1.0, "mm3_loss": -1.0, "mm3_cap_frac": 0.0,
                   "mm3_dmd_ratio": 0.0, "mm3_roll_hf_ratio": 1.0, "mm3_roll_lf_ratio": 1.0, "mm3_roll_on": 0.0}
    p._v84_reset()
    p.random_exit_step = False         # main pass exits at the final step, so the rollout-window terms run too
    return p, gt


def run(p, gt):
    t7.student(p, 5)
    s = {"F_student": len(p.calls["F"]),
         # the first-window pass is the only student call made with the first window's conditions (stand-in tag w0)
         "first_window_pass": any(c["tag"] == "w0" for c in p.calls["gen"]),
         "mm3_first_loss_set": p._last_mm3["mm3_loss"] >= 0.0,
         "sf_on": p._v84_last["sf_on"] == 1.0,
         "temporal_weight": float(getattr(p, "_last_temporal_weight", 0.0)),
         "gen_param_grad": p.gen_param.grad is not None and float(p.gen_param.grad.abs()) > 0}
    p.gen_param.grad = None
    heads_before = p.calls["head_steps"]
    p.proj.grad = None
    row = t7.critic(p, gt, 5)
    s["F_critic"] = len(p.calls["F"])
    s["F_critic_grad"] = sum(c["grad"] for c in p.calls["F"])
    s["head_steps"] = p.calls["head_steps"] - heads_before
    s["critic_keys"] = sorted(row)
    s["heads"] = sorted(p.disc_head.keys()) if p.use_gan else []
    return s


def main():
    out = {}
    p, gt = build(temporal=True)
    out["A_ctrl"] = run(p, gt)

    p, gt = build(temporal=True)
    p.gan_generator_weight = p.gan_discriminator_weight = 0.0
    p.use_gan = False
    out["B_noadv"] = run(p, gt)

    p, gt = build(temporal=False)
    out["D_notemp"] = run(p, gt)

    p, gt = build(temporal=True)
    p._stash.pop("cond0")
    p._stash.pop("roll")
    out["C_rechist_emulated"] = run(p, gt)

    for name, s in out.items():
        print(f"{name:20s} " + " ".join(f"{k}={v}" for k, v in s.items() if k != "critic_keys"), flush=True)

    a, b, d, c = out["A_ctrl"], out["B_noadv"], out["D_notemp"], out["C_rechist_emulated"]
    # A: all terms on
    assert a["F_student"] == 2 and a["first_window_pass"] and a["mm3_first_loss_set"] and a["sf_on"], a
    assert a["temporal_weight"] == 1.0 and len(a["heads"]) == 2 * N_LAYERS, a
    assert a["F_critic"] == 5 and a["F_critic_grad"] == 1 and a["head_steps"] == 1, a
    # B: no adversarial forward, no heads, no head step; first-window pass and mm3 kept; flow matching only
    assert b["F_student"] == 0 and b["head_steps"] == 0 and b["heads"] == [], b
    assert b["first_window_pass"] and b["mm3_first_loss_set"] and b["sf_on"] and b["gen_param_grad"], b
    assert b["F_critic"] == 1 and b["F_critic_grad"] == 1 and b["critic_keys"] == ["critic_loss"], b
    # D: as A but spatial heads only
    assert d["F_student"] == 2 and d["first_window_pass"] and d["mm3_first_loss_set"] and d["sf_on"], d
    assert d["temporal_weight"] == 0.0 and len(d["heads"]) == N_LAYERS and not any(h.startswith("t") for h in d["heads"])
    assert d["F_critic"] == 5 and d["head_steps"] == 1, d
    # C emulated: without the rollout stash the first-window pass and the first pair are skipped
    assert not c["first_window_pass"] and not c["mm3_first_loss_set"], c
    assert c["F_student"] == 1, c          # rollout adversarial term only, no first-pair term
    assert c["F_critic"] == 3, c           # flow matching + rollout pair (2), no first pair
    full_step_cases()
    print("ABLATION_SWITCH_TESTS PASSED", flush=True)
    return 0


def build_full(base, recorded=None):
    """Stand-in of `base` (v8.4 or v8.4R) that can run the real train_step chain on tiny tensors."""
    cls = t83.defaults(type("Full84", (base,), {}), True, 1.0)
    t84.defaults84(cls)
    t7.PanoContextForcedDMDv7 = cls
    try:
        p, _ = t7.make(warmup=(0, 1))
    finally:
        t7.PanoContextForcedDMDv7 = PanoContextForcedDMDv7
    p._last_mm = {"mm_ratio": 1.0, "mm_loss": 0.0, "mm_dmd_ratio": 0.0, "mm_cap_frac": 0.0}
    p._mm_stats, p._mm_w = None, 0.0
    p._last_mm3 = {"mm3_hf_ratio": 1.0, "mm3_lf_ratio": 1.0, "mm3_loss": -1.0, "mm3_cap_frac": 0.0,
                   "mm3_dmd_ratio": 0.0, "mm3_roll_hf_ratio": 1.0, "mm3_roll_lf_ratio": 1.0, "mm3_roll_on": 0.0}
    p._v84_reset()
    if recorded is not None:
        p.recorded_history = recorded
    p._rec_window = 0
    p.context_forcing_prob = 1.0
    p.real_score_wrapper = type("RS", (), {"eval": lambda self: None})()
    p.ema_start_step, p.step_duration, p.loss_report_list = 10 ** 9, 0.0, []
    p._pending_rng, p._checkpoint_steps, p._target_steps = None, 50, 10 ** 9
    p._checkpoint_seconds, p._last_save_time, p._last_saved = 1e9, time.monotonic(), None
    p._conditions = lambda sub, shape, cfg: {"tag": f"f{int(sub['frame_id'][0, 0, 0])}", "frames": sub["frame_id"],
                                             "shape": shape, "cfg": cfg}
    rec = {"gen": [], "critic": [], "pre_gen_calls": None}
    gs, cs = p._generator_step_cf, p._critic_step_cf

    def gen(latent_shape, cond, cond_cfg, history, target_mask):
        if rec["pre_gen_calls"] is None:
            rec["pre_gen_calls"] = [c["tag"] for c in p.calls["gen"]]
        n0, f0 = len(p.calls["gen"]), len(p.calls["F"])
        out = gs(latent_shape, cond, cond_cfg, history, target_mask)
        rec["gen"].append({"history": history.detach().clone(), "cond": cond["tag"], "cfg": cond_cfg["tag"],
                           "calls": [c["tag"] for c in p.calls["gen"][n0:]], "F": len(p.calls["F"]) - f0})
        return out

    def critic(latent_shape, cond, history, gt_window, target_mask):
        f0, h0 = len(p.calls["F"]), p.calls["head_steps"]
        out = cs(latent_shape, cond, history, gt_window, target_mask)
        new = p.calls["F"][f0:]
        rec["critic"].append({"history": history.detach().clone(), "gt_window": gt_window.detach().clone(),
                              "cond": cond["tag"], "F": len(new), "F_grad": sum(c["grad"] for c in new),
                              "heads": p.calls["head_steps"] - h0})
        return out

    p._generator_step_cf, p._critic_step_cf = gen, critic
    return p, rec


def run_full(base, recorded=None):
    p, rec = build_full(base, recorded)
    frame_id = torch.arange(FRAMES).view(1, FRAMES, 1).float().expand(t7.B, FRAMES, t7.V)
    batch = {"vae_images": torch.zeros(t7.B, FRAMES, t7.V, 3, t7.H * 8, t7.W * 8), "frame_id": frame_id,
             "clip_text": ["x"]}
    saved_reduce = torch.distributed.all_reduce
    torch.distributed.all_reduce = lambda *a, **k: None
    P.PanoContextForcedDMD._vae_encode = lambda self, images: GT[:, :images.shape[1]].clone()
    try:
        p.train_step(batch, STEP)
    finally:
        torch.distributed.all_reduce = saved_reduce
        del P.PanoContextForcedDMD._vae_encode
    st = p._stash
    return {"p": p, "rec": rec, "k": p.context_windows, "roll": [r.clone() for r in st.get("roll", [])],
            "cond0": st["cond0"]["tag"] if "cond0" in st else None,
            "F_x": [c["x"] for c in p.calls["F"]], "F_t": [c["t"] for c in p.calls["F"]],
            "gen_tags": [c["tag"] for c in p.calls["gen"]],
            "gen_grad": p.gen_param.grad.detach().clone(), "F_grad": p.proj.grad.detach().clone(),
            "report": dict(p.loss_report_list[-1]), "gen_rng": p.generator.get_state(),
            "torch_rng": torch.get_rng_state(), "mm3_first": p._last_mm3["mm3_loss"] >= 0.0}


def same_tensors(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b))


def full_step_cases():
    ctrl = run_full(PanoContextForcedDMDv8_4)
    off = run_full(PanoContextForcedDMDv8_4R, recorded=False)
    on = run_full(PanoContextForcedDMDv8_4R, recorded=True)
    k = ctrl["k"]
    start = STRIDE * k
    rec_hist = GT[:, start:start + CONTEXT]
    # (i) knob off == v8.4 bit for bit on a full step
    assert off["k"] == k
    for key in ("gen", "critic"):
        for x, y in zip(ctrl["rec"][key], off["rec"][key]):
            assert x.keys() == y.keys()
            for name in x:
                if isinstance(x[name], torch.Tensor):
                    assert torch.equal(x[name], y[name]), (key, name)
                else:
                    assert x[name] == y[name], (key, name, x[name], y[name])
    assert same_tensors(ctrl["roll"], off["roll"]) and ctrl["cond0"] == off["cond0"]
    assert same_tensors(ctrl["F_x"], off["F_x"]) and same_tensors(ctrl["F_t"], off["F_t"])
    assert ctrl["gen_tags"] == off["gen_tags"] and ctrl["rec"]["pre_gen_calls"] == off["rec"]["pre_gen_calls"]
    assert torch.equal(ctrl["gen_grad"], off["gen_grad"]) and torch.equal(ctrl["F_grad"], off["F_grad"])
    assert ctrl["report"] == off["report"]
    assert torch.equal(ctrl["gen_rng"], off["gen_rng"]) and torch.equal(ctrl["torch_rng"], off["torch_rng"])
    print(f"ok (i): knob off == v8.4 bit for bit on a full train step (k={k}; {len(ctrl['F_x'])} F inputs, "
          f"{len(ctrl['gen_tags'])} student calls, report, gradients, RNG states)", flush=True)
    # (ii) knob on: only the trained window's history changes
    cg, og = ctrl["rec"]["gen"][0], on["rec"]["gen"][0]
    cc, oc = ctrl["rec"]["critic"][0], on["rec"]["critic"][0]
    assert on["k"] == k and og["cond"] == cg["cond"] == f"f{start}" and og["cfg"] == cg["cfg"]
    assert torch.equal(og["history"], rec_hist) and torch.equal(oc["history"], rec_hist), "trained history = recorded"
    assert not torch.equal(cg["history"], rec_hist), "control trains behind its own rollout"
    assert torch.equal(oc["gt_window"], cc["gt_window"]) and oc["cond"] == cc["cond"]
    assert on["cond0"] == ctrl["cond0"] == "f0" and len(on["roll"]) == 1 and len(ctrl["roll"]) == k
    assert torch.equal(on["roll"][0], ctrl["roll"][0]), "window 1 sampled as in the control"
    n_steps = 4
    assert ctrl["rec"]["pre_gen_calls"] == [f"f{STRIDE * w}" for w in range(k) for _ in range(n_steps)]
    assert on["rec"]["pre_gen_calls"] == ["f0"] * n_steps, on["rec"]["pre_gen_calls"]
    assert "f0" in og["calls"] and "f0" in cg["calls"], "first-window pass in the student step"
    assert og["F"] == cg["F"] == 2, (og["F"], cg["F"])            # rollout + first adversarial forwards
    assert on["mm3_first"] and ctrl["mm3_first"], "mm3 first-window term"
    assert oc["F"] == cc["F"] == 5 and oc["F_grad"] == cc["F_grad"] == 1 and oc["heads"] == cc["heads"] == 1
    assert set(on["report"]) == set(ctrl["report"])
    print(f"ok (ii): knob on: trained-window history = recorded frames {start}..{start + 2} (control: own rollout), "
          f"same k / conditions / real window, window 1 sampled bit-equal to the control, windows 2..{k} not sampled, "
          f"student step: first-window pass + mm3 first-window term + 2 adversarial forwards, critic 5 F forwards "
          f"(1 with grad) + 1 head step, as the control", flush=True)


if __name__ == "__main__":
    sys.exit(main())
