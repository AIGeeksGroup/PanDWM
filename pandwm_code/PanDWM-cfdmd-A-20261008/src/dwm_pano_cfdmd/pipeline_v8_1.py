"""v8.1 = v8 with two fixes to the temporal-difference heads (2026-10-11).

v8 (branch E, resumed from v7 step 1000) did not reduce flicker.  Its latent flicker monitor motion_ratio_w1 went from
1.083 (steps 1001-1150, same as v7) and 1.075 (1151-1250) up to 1.123 (1351-1430), and hf_ratio_w1 from 0.98 to 1.02,
while v7 had been drifting down (1.137 -> 1.101 -> 1.083 over steps 701-1150).  Two causes in v8's temporal heads:

1. Each temporal head began with LayerNorm on the change feature f_k - f_{k-1}.  That divides out the size of the
   change: the head could not tell a large change from a small one, and its gradient (LayerNorm's Jacobian removes
   the input's own direction) could not ask the student for a smaller change, only for a differently shaped one.
   v8.1 normalises each frame's pooled features on their own (LayerNorm without learned scale) and feeds the
   difference of the normalised features to Linear -> SiLU -> Linear, so the head sees how much a token changes
   relative to its own feature scale.
2. The new heads pushed the student from their first step, while near random (v7's heads had 100 heads-only steps
   and a 200-step ramp).  In v8.1 the temporal heads' share of the student's adversarial term has weight
   w_t = 0 before `temporal_adv_from`, then ramps linearly to 1 over `temporal_adv_ramp` steps; the heads train
   throughout.  Student term: -(sum of spatial head means + w_t * sum of temporal head means) /
   (n_spatial + w_t * n_temporal), which is exactly v7's while w_t = 0 and v8's form at w_t = 1.

Everything else as v8 / v7 (pairs, sigma, cap on the total adversarial gradient, schedules, v7-tolerant resume).
Report adds `temporal_weight` (w_t of the last student step).
"""
import torch
import torch.nn.functional as F

from .pipeline import CONTEXT
from .pipeline_v7 import CapGrad, make_heads
from .pipeline_v8 import TEMPORAL, HeadDict, PanoContextForcedDMDv8


def make_change_heads(layers, hidden, width=256):
    """Temporal heads without an input LayerNorm (their input is already a difference of normalised features)."""
    return {TEMPORAL + str(int(layer)): torch.nn.Sequential(torch.nn.Linear(hidden, width), torch.nn.SiLU(),
                                                           torch.nn.Linear(width, 1))
            for layer in layers}


class PanoContextForcedDMDv8_1(PanoContextForcedDMDv8):
    def _get_heads(self):
        return self._v7_heads

    def _set_heads(self, _parent_head):
        hidden = self.fake_score.config.num_attention_heads * self.fake_score.config.attention_head_dim
        heads = HeadDict(make_heads(self._v7_layers, hidden))
        if self._v8_temporal:
            for name, head in make_change_heads(self._v7_layers, hidden).items():
                heads[name] = head
        self._v7_heads = heads.to(self.device)

    disc_head = property(_get_heads, _set_heads)

    def __init__(self, *args, temporal_adv_from=0, temporal_adv_ramp=0, **kwargs):
        self.temporal_adv_from, self.temporal_adv_ramp = int(temporal_adv_from), int(temporal_adv_ramp)
        self._last_temporal_weight = 0.0
        super().__init__(*args, **kwargs)
        if self.should_save:
            print(f"V8_1 temporal head input = difference of per-frame LayerNorm features (no LayerNorm on the "
                  f"difference); temporal_adv_from={self.temporal_adv_from} ramp={self.temporal_adv_ramp}",
                  flush=True)

    def _temporal_weight(self):
        if not self._v8_temporal:
            return 0.0
        step = self._cur_step or 0
        if step < self.temporal_adv_from:
            return 0.0
        if self.temporal_adv_ramp <= 0:
            return 1.0
        return min(1.0, (step - self.temporal_adv_from + 1) / self.temporal_adv_ramp)

    def _token_logits(self, feats, shape, drop_context):
        """As v8, but a temporal head sees LayerNorm(f_k) - LayerNorm(f_{k-1}) per pooled token (size kept)."""
        b, t, v = shape[:3]
        frames = t - CONTEXT if drop_context else t
        p = self.fake_score.config.patch_size
        hp, wp = shape[-2] // p, shape[-1] // p
        pooled = {}
        for layer in self._v7_layers:
            h = feats[layer]
            if drop_context:
                h = h.unflatten(0, (b, t, v))[:, CONTEXT:].flatten(0, 2)
            n, tokens, c = h.shape
            if tokens != hp * wp:
                raise ValueError(f"block {layer}: {tokens} tokens per view, expected {hp} x {wp}")
            h = h.reshape(n, hp, wp, c).permute(0, 3, 1, 2)       # tokens are row-major (see unpatchify)
            if self.disc_pool > 1:
                h = F.avg_pool2d(h, self.disc_pool)
            pooled[layer] = h.permute(0, 2, 3, 1)                 # [(b frames v), hp', wp', c]
        heads = self.disc_head
        out = [heads[str(layer)](pooled[layer].float()).squeeze(-1) for layer in self._v7_layers]
        if self._v8_temporal:
            for layer in self._v7_layers:
                x = pooled[layer].float()
                z = F.layer_norm(x, (x.shape[-1],)).unflatten(0, (b, frames, v))
                change = (z[:, 1:] - z[:, :-1]).flatten(0, 2)
                out.append(heads[TEMPORAL + str(layer)](change).squeeze(-1))
        return out

    def _adv_term(self, x0, inputs_fn, cap, stats):
        """-mean(D) on the student's own sample with the temporal heads weighted by w_t; gradient to x0 capped."""
        x_adv = CapGrad.apply(x0, cap, stats)
        noisy, ts, _, cond, drop = inputs_fn([x_adv])
        feats, out = self._disc_features(noisy[0], ts, cond)
        logits = self._token_logits(feats, noisy[0].shape, drop)
        n = len(self._v7_layers)
        w = self._temporal_weight()
        self._last_temporal_weight = w
        total = sum(x.mean() for x in logits[:n])
        count = float(n)
        if w > 0 and len(logits) > n:
            total = total + w * sum(x.mean() for x in logits[n:])
            count += w * (len(logits) - n)
        return -total / count, 0.0 * out[0].float().sum()

    @torch.no_grad()
    def _report(self, stats):
        out = super()._report(stats)
        if self._v8_temporal:
            out["temporal_weight"] = self._last_temporal_weight
        return out
