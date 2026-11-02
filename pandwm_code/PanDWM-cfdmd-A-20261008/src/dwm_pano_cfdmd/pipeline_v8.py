"""v8 = v7 + temporal-difference discriminator heads, aimed at the flicker v7 leaves (2026-10-10).

Step-500 evaluation of v7 (eval_v7/V7_EVAL_500_20261010.md, frozen 149-clip protocol): the student's
frame-to-frame change is ~20% above the real footage while B0's is ~7% above, so distillation adds flicker; the
latent monitor motion_ratio_w1 agrees (1.1-1.2).  The v7 heads score every token of every frame on its own, so a
token that changes too much between frames looks no different from one that does not.  v8 adds, per feature
block, a second small head that scores the change of each pooled token between consecutive target frames
(feature at frame k minus frame k-1, same view and place).  Real and fake of a pair share sigma and the noise
tensor, so noise-driven differences are the same on both sides; excess change is what tells them apart.

Everything else is v7: same pairs, sigma, cap (it bounds the sum over both kinds of heads), warm-up, schedule.
The hinge and student losses average over all heads, spatial and temporal alike.

Resume from a v7 checkpoint (branching off a v7 run): the v7 heads load as saved, the temporal heads keep their
fresh initialisation, and the head optimizer takes the v7 heads' Adam state while the temporal heads start with
none (HeadDict and TolerantAdamW below; both only relax the "v7 file, more v8 heads" case).

Reported: disc_* / disc1_* / d_gap_* are the spatial heads only (same meaning as in v7; disc_loss / disc1_loss
are the full pair losses); dt_real_acc / dt_fake_acc / dt_gap (rollout pair) and dt1_real_acc / dt1_fake_acc /
dt1_gap (first pair) are the temporal heads.
"""
import torch
import torch.nn.functional as F

from .pipeline import CONTEXT
from .pipeline_v7 import PanoContextForcedDMDv7, make_heads

TEMPORAL = "t"            # key prefix of the temporal heads, e.g. "t12" next to the spatial "12"


def _rank0():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


class HeadDict(torch.nn.ModuleDict):
    """ModuleDict that also loads a v7 head file: missing temporal-head tensors keep their initial values."""

    def load_state_dict(self, state_dict, strict=True, assign=False):
        own = self.state_dict()
        missing = [k for k in own if k not in state_dict]
        unexpected = [k for k in state_dict if k not in own]
        if missing and not unexpected and all(k.startswith(TEMPORAL) for k in missing):
            merged = dict(own)
            merged.update(state_dict)
            if _rank0():
                print(f"V8_HEADS_FROM_V7: {len(state_dict)} saved tensors loaded, {len(missing)} temporal-head "
                      f"tensors start fresh", flush=True)
            return super().load_state_dict(merged, strict=True, assign=assign)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class TolerantAdamW(torch.optim.AdamW):
    """AdamW that also loads a v7 head-optimizer state (one group, fewer parameters): the saved state goes to the
    leading parameters, which are the v7 heads because the spatial heads come first in HeadDict; the temporal
    heads start without state, as at the beginning of training."""

    def load_state_dict(self, state_dict):
        saved, own = state_dict["param_groups"], self.param_groups
        if len(saved) == 1 and len(own) == 1 and len(saved[0]["params"]) < len(own[0]["params"]):
            n_saved, n_own = len(saved[0]["params"]), len(own[0]["params"])
            index = {old: new for new, old in enumerate(saved[0]["params"])}
            state = {index[k]: v for k, v in state_dict["state"].items()}
            group = dict(saved[0], params=list(range(n_own)))
            if _rank0():
                print(f"V8_HEAD_OPTIMIZER_FROM_V7: Adam state for {n_saved} of {n_own} head tensors", flush=True)
            return super().load_state_dict({"state": state, "param_groups": [group]})
        return super().load_state_dict(state_dict)


class PanoContextForcedDMDv8(PanoContextForcedDMDv7):
    # Same mechanism as v7: the parent's head assignment builds the v8 heads, so the optimizer, resume,
    # save_checkpoint and milestones use them unchanged.
    @property
    def disc_head(self):
        return self._v7_heads

    @disc_head.setter
    def disc_head(self, _parent_head):
        hidden = self.fake_score.config.num_attention_heads * self.fake_score.config.attention_head_dim
        heads = HeadDict(make_heads(self._v7_layers, hidden))
        if self._v8_temporal:
            for name, head in make_heads(self._v7_layers, hidden).items():
                heads[TEMPORAL + name] = head
        self._v7_heads = heads.to(self.device)

    # The parent builds a plain AdamW over the heads; give it the v7-tolerant load_state_dict.
    @property
    def disc_optimizer(self):
        return self._v8_disc_optimizer

    @disc_optimizer.setter
    def disc_optimizer(self, optimizer):
        if type(optimizer) is torch.optim.AdamW:
            optimizer.__class__ = TolerantAdamW
        self._v8_disc_optimizer = optimizer

    def __init__(self, *args, disc_temporal=True, **kwargs):
        self._v8_temporal = bool(disc_temporal)
        super().__init__(*args, **kwargs)
        if self.should_save:
            heads = list(self.disc_head.keys()) if self.use_gan else []
            print(f"V8 temporal_heads={self._v8_temporal} heads={heads}", flush=True)

    def _token_logits(self, feats, shape, drop_context):
        """Spatial logits [(b t v), hp / pool, wp / pool] per block (as v7), then temporal logits
        [(b (t - 1) v), hp / pool, wp / pool] per block: the head applied to the change of each pooled token
        from the previous target frame."""
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
                h = pooled[layer].unflatten(0, (b, frames, v))
                change = (h[:, 1:] - h[:, :-1]).flatten(0, 2)
                out.append(heads[TEMPORAL + str(layer)](change.float()).squeeze(-1))
        return out

    @torch.no_grad()
    def _report(self, stats):
        if not self._v8_temporal:
            return super()._report(stats)
        n = len(self._v7_layers)
        spatial, temporal = {}, {}
        for key, (loss, lr, lf, *rest) in stats.items():
            spatial[key] = (loss, lr[:n], lf[:n], *rest)
            temporal[key] = (lr[n:], lf[n:])
        out = super()._report(spatial)
        row = [float("nan")] * 6
        for i, key in enumerate(("roll", "first")):
            if key in temporal:
                lr, lf = temporal[key]
                real = torch.stack([x.float().mean() for x in lr]).mean()
                fake = torch.stack([x.float().mean() for x in lf]).mean()
                acc_r = torch.stack([(x > 0).float().mean() for x in lr]).mean()
                acc_f = torch.stack([(x < 0).float().mean() for x in lf]).mean()
                row[3 * i:3 * i + 3] = [float(acc_r), float(acc_f), float(real - fake)]
        vec = torch.tensor(row, device=self.device, dtype=torch.float32)
        if torch.distributed.is_initialized():
            rows = torch.empty(self._world * vec.numel(), device=self.device)
            torch.distributed.all_gather_into_tensor(rows, vec)
            rows = rows.view(self._world, -1)
        else:
            rows = vec.view(1, -1)

        def avg(col, default):
            v = rows[:, col]
            ok = torch.isfinite(v)
            return float(v[ok].mean()) if ok.any() else default

        if "rollout" in self.disc_pairs:
            out.update(dt_real_acc=avg(0, 0.5), dt_fake_acc=avg(1, 0.5), dt_gap=avg(2, 0.0))
        if "first" in self.disc_pairs:
            out.update(dt1_real_acc=avg(3, 0.5), dt1_fake_acc=avg(4, 0.5), dt1_gap=avg(5, 0.0))
        return out
