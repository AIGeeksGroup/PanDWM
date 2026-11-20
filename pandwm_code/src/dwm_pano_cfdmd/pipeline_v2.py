"""Context-Forced DMD with a GAN loss: keeps the fake-score output in the generator graph.

The discriminator reads an intermediate block feature of the FSDP-wrapped fake score through a forward hook.
FSDP registers its pre-backward hook on the module output, so the output must be part of the graph; in the
generator step a zero-weighted output term is added to the discriminator logit (values and gradients unchanged).
"""
from dwm_pano_cfdmd.pipeline import CONTEXT, PanoContextForcedDMD


class PanoContextForcedDMDv2(PanoContextForcedDMD):
    _touch_fake_output = False

    def _generator_step_cf(self, *args, **kwargs):
        self._touch_fake_output = True
        try:
            return super()._generator_step_cf(*args, **kwargs)
        finally:
            self._touch_fake_output = False

    def _disc_logits(self, noisy, ts, cond):
        """Same as the parent, but keeps the fake-score output in the graph when asked."""
        cond_fake = dict(cond)
        if self.fake_score_causal:
            self._add_temporal_attention_mask(cond_fake, noisy.shape)
        self._capture = True
        try:
            with self.get_autocast_context():
                out, _, _ = self.fake_score_wrapper(noisy.to(self.model_dtype), ts, **cond_fake)
        finally:
            self._capture = False
        h = self._features.pop("h")                      # ((b t v), tokens, c)
        b, t, v = noisy.shape[:3]
        h = h.float().unflatten(0, (b, t, v))[:, CONTEXT:]   # target frames only
        logit = self.disc_head(h.mean(dim=(1, 2, 3))).squeeze(-1)
        if self._touch_fake_output:
            logit = logit + 0.0 * out[0].float().sum()
        return logit
