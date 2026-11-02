"""GAN fix for Context-Forced DMD (pipeline.py is untouched).

Failure (job 12922710, step 0): ValueError "expected to be in states
[FORWARD_BACKWARD] but current state is IDLE" in FSDP's _post_backward_hook.
Cause: _disc_logits takes an intermediate block feature of the FSDP-wrapped fake
score through a forward hook and the generator loss never touches the module
OUTPUT.  FSDP registers its pre-backward hook on the module output, so that hook
never fires, the training state stays IDLE (set by post-forward) and the
post-backward hook of the root unit's parameters asserts.
Reproduced on a tiny FSDP model (job 12928685): fails without the output term,
passes with it, for root-only and block-wrapped FSDP.

Fix: in the GENERATOR step only, add 0 * (module output) to the discriminator
logit.  Values and gradients are unchanged (the extra term is exactly zero); the
output is just part of the graph so the pre-backward hook fires.  The critic step
already has the output in its graph (flow-matching loss), so it is left alone to
avoid retaining two extra full activation graphs.
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
