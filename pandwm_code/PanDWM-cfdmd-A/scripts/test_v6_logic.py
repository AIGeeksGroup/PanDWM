"""CPU logic test for pipeline_v6 (no GPU, no real model): python scripts/test_v6_logic.py

Builds a v6 instance without __init__, replaces the heavy parts (generator rollout, the v5 student/critic
steps) with tiny stand-ins, and checks:
  1. weight 0  -> the regression pass never runs (v6 == v5);
  2. weight >0 -> the first window is rolled out with the deployed sampler (random exit off, restored after),
                  the loss uses only the 16 target latents, and its gradient is already in the student
                  parameters when the v5 student step runs (so both land in one optimizer step);
  3. every critic report carries gt_reg (the trainer averages each key over all rows).
"""
import torch

import dwm_pano_cfdmd.pipeline_v5 as v5mod
from dwm_pano_cfdmd.pipeline import CONTEXT, WINDOW
from dwm_pano_cfdmd.pipeline_v6 import PanoContextForcedDMDv6


class Scaler:
    def __init__(self):
        self.scaled = []

    def scale(self, loss):
        self.scaled.append(float(loss))
        return loss * 4.0


def make(weight):
    p = object.__new__(PanoContextForcedDMDv6)
    p.gt_regression_weight = weight
    p._last_gt_reg = 0.0
    p.random_exit_step = True
    p.training_config = {"enable_grad_scaler": True}
    p.grad_scaler = Scaler()
    p.param = torch.nn.Parameter(torch.zeros(()))
    shape = (1, WINDOW, 2, 4, 2, 2)
    gt = torch.randn(1, 51, 2, 4, 2, 2)
    p._stash = {"gt": gt, "cond0": {"c": 0}, "latent_shape": shape}
    p.calls = []

    def run_generator(latent_shape, cond, image_latents=None, reference_frame_count=0, grad_on_last=True):
        p.calls.append({"exit_random": p.random_exit_step, "ref": reference_frame_count,
                        "hist_is_gt": torch.equal(image_latents, gt[:, :CONTEXT]), "grad": grad_on_last})
        x0 = torch.zeros(latent_shape) + p.param
        x0 = x0.clone()
        x0[:, :CONTEXT] = 1e6 + p.param          # garbage in history slots must not reach the loss
        return x0

    p._run_generator = run_generator
    return p, gt


def main():
    seen = {}

    def v5_student(self, *a, **k):
        seen["grad_at_v5_step"] = None if self.param.grad is None else float(self.param.grad)
        return {"gen_loss": 0.0, "dmd_grad_norm": 0.0, "adv_gen": 0.0}

    def v5_critic(self, *a, **k):
        return {"critic_loss": 0.0, "disc_fake_acc": 0.5, "disc_loss": 1.0, "disc_real_acc": 0.5}

    v5mod.PanoContextForcedDMDv5._generator_step_cf = v5_student
    v5mod.PanoContextForcedDMDv5._critic_step_cf = v5_critic

    p, _ = make(0.0)
    out = p._generator_step_cf(None, None, None, None, None)
    assert p.calls == [] and seen["grad_at_v5_step"] is None, "weight 0 must not run the regression pass"
    assert set(out) == {"gen_loss", "dmd_grad_norm", "adv_gen"}
    assert "gt_reg" in p._critic_step_cf(None, None, None, None, None)
    print("ok: weight 0 is plain v5")

    p, gt = make(0.1)
    p._generator_step_cf(None, None, None, None, None)
    assert len(p.calls) == 1 and p.calls[0] == {"exit_random": False, "ref": CONTEXT, "hist_is_gt": True, "grad": True}
    assert p.random_exit_step is True, "random exit must be restored"
    expected_mse = float(gt[:, CONTEXT:WINDOW].square().mean())
    assert abs(p._last_gt_reg - expected_mse) < 1e-5, (p._last_gt_reg, expected_mse)
    expected_grad = 4.0 * 0.1 * float((-2.0 * gt[:, CONTEXT:WINDOW]).mean())
    assert seen["grad_at_v5_step"] is not None and abs(seen["grad_at_v5_step"] - expected_grad) < 1e-4, seen
    assert abs(p.grad_scaler.scaled[0] - 0.1 * expected_mse) < 1e-5
    rows = [p._critic_step_cf(None, None, None, None, None) for _ in range(3)]
    assert all(r["gt_reg"] == p._last_gt_reg for r in rows)
    print(f"ok: regression on target latents only (mse={p._last_gt_reg:.4f}), deployed sampler, "
          f"gradient present before the v5 step, gt_reg in every critic row")

    p, _ = make(0.1)
    p._stash = {}
    p._generator_step_cf(None, None, None, None, None)
    assert p.calls == [], "no stash (e.g. real-history branch) must skip the regression pass"
    print("ok: skipped cleanly when the first-window data is missing")
    print("ALL V6 LOGIC TESTS PASSED")


if __name__ == "__main__":
    main()
