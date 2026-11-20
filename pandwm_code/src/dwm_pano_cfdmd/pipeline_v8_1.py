"""Temporal-difference discriminator heads with scale-aware input and a delayed adversarial ramp.

Each temporal head sees the difference of per-frame normalised pooled features (LayerNorm without learned scale,
then Linear -> SiLU -> Linear), so it can tell a large change from a small one relative to the feature scale.
The temporal heads' share of the student's adversarial term has weight w_t = 0 before `temporal_adv_from`, then
ramps linearly to 1 over `temporal_adv_ramp` steps; the heads train throughout. Student term:
-(sum of spatial head means + w_t * sum of temporal head means) / (n_spatial + w_t * n_temporal).
Everything else is as in v8. The report adds `temporal_weight`.
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
        """As in v8, but a temporal head sees LayerNorm(f_k) - LayerNorm(f_{k-1}) per pooled token (size kept)."""
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
