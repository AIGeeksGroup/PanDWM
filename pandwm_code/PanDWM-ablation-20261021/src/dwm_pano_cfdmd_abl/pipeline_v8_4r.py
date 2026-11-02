"""v8.4R = v8.4 + one default-off knob for the ablation "recorded instead of self-generated history" (2026-10-21).

User decision 2026-10-21 (ABLATION_20261021.md section 1.8, option (b)): with the knob on, ONLY the history of the trained
window changes, from the student's own rollout to the recorded frames; window 1 is still sampled by the student from the
recorded history, so the first-window terms (mm3 first-window motion term, first adversarial pair on the student side
and its discriminator training) stay exactly as in the full recipe.

Why a knob: pipeline.py's switch context_forcing_prob = 0 also removes those first-window terms, because their
conditions and window-1 sample are stashed by _rollout_window (pipeline_v7.py) during the rollout it skips.

How (recorded_history=True): context_forcing_prob must stay 1.0, so pipeline.py's train_step runs its rollout loop for
w = 0 .. k-1 (and draws torch.rand for the forcing decision exactly as v8.4).  This class overrides _rollout_window:
  w = 0      the real v8.4 call (deployed 4-step sample of window 1 from the recorded history it is given, which stashes
             cond0 / latent_shape / roll[0] exactly as v8.4), but it returns the RECORDED window 0 frames;
  w >= 1     no student forward; returns the recorded frames of window w.
pipeline.py keeps the last 3 frames of the returned window as the next history, so after the loop the trained window's
history is the recorded frames 16k .. 16k+2 -- the same frames pipeline.py's context_forcing_prob = 0 branch uses.
The rollout depth k, the trained window's position, its conditions and every loss term are unchanged.  Not changed by
the knob but different as a consequence: no student forwards for windows 2..k (so the pipeline generator's noise is
not consumed for them), and the depth monitors see one rollout window only (logging).
recorded_history=False: every method defers to v8.4 unchanged (bit-identical; scripts/test_ablation_switches.py).
"""
import torch

from dwm_pano_cfdmd.pipeline import STRIDE, WINDOW
from dwm_pano_cfdmd.pipeline_v8_4 import PanoContextForcedDMDv8_4


class PanoContextForcedDMDv8_4R(PanoContextForcedDMDv8_4):
    # Class default keeps instances built without __init__ (the CPU logic tests) on the v8.4 path.
    recorded_history = False

    def __init__(self, *args, recorded_history=False, **kwargs):
        self.recorded_history = bool(recorded_history)
        self._rec_window = 0
        super().__init__(*args, **kwargs)
        if self.recorded_history and self.context_forcing_prob != 1.0:
            raise ValueError("recorded_history needs context_forcing_prob 1.0 (the rollout loop must run)")
        if self.should_save:
            print(f"V8_4R recorded_history={self.recorded_history} (on: trained-window history = recorded frames, "
                  f"window 1 still sampled from the recorded history for the first-window terms)", flush=True)

    def train_step(self, batch, global_step):
        self._rec_window = 0
        return super().train_step(batch, global_step)

    @torch.no_grad()
    def _rollout_window(self, latent_shape, cond, history):
        if not self.recorded_history:
            return super()._rollout_window(latent_shape, cond, history)
        w = self._rec_window
        self._rec_window += 1
        gt = self._stash["gt"]
        recorded = gt[:, STRIDE * w:STRIDE * w + WINDOW]
        if recorded.shape[1] != WINDOW:
            raise ValueError(f"recorded window {w} has {recorded.shape[1]} frames")
        if w == 0:
            x0 = super()._rollout_window(latent_shape, cond, history)
            self._rec_dtype = x0.dtype
        return recorded.to(getattr(self, "_rec_dtype", recorded.dtype))
