"""D2 refiner DiT (2026-10-21, PanDWM-d2-refiner-20261021/D2_PLAN_20261021.md section 1, Task 1).

DiTCrossviewTemporalConditionModel plus one dense render condition: a (b, t, v, C, h, w) tensor with the same
(b, t, v, h, w) as the noisy latents (C = 16 render-latent channels + 1 coverage channel) is patchified by a
zero-initialised Conv2d(C, inner_dim, patch, patch), multiplied by render_condition_scale and added to the pos_embed
output, i.e. to the token embedding of the noisy latents before the first transformer block.

* With the zero init, or with render_condition=None, the model is the base model bit for bit.
* render_condition_scale multiplies the new layer's effective step size under Adam at the base learning rate.
* The base forward is not copied: the addition happens in a forward hook on pos_embed, so the state_dict keys are
  the base keys plus render_embed.weight / render_embed.bias and a base checkpoint loads with strict=False.
"""

import torch

from dwm_pano.models.crossview_temporal_dit_pano import DiTCrossviewTemporalConditionModel


class D2RenderDiT(DiTCrossviewTemporalConditionModel):
    def __init__(self, render_condition_channels: int = 17, render_condition_scale: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        inner_dim = self.config.attention_head_dim * self.config.num_attention_heads
        patch = self.config.patch_size
        self.render_condition_channels = int(render_condition_channels)
        self.render_condition_scale = float(render_condition_scale)
        self.render_embed = torch.nn.Conv2d(self.render_condition_channels, inner_dim, kernel_size=patch,
                                            stride=patch)
        torch.nn.init.zeros_(self.render_embed.weight)
        torch.nn.init.zeros_(self.render_embed.bias)
        self._d2_pending = None
        self.pos_embed.register_forward_hook(self._d2_add_render_tokens)

    def _d2_add_render_tokens(self, module, inputs, output):
        pending = self._d2_pending
        if pending is None:
            return output
        if tuple(pending.shape) != tuple(output.shape):
            raise ValueError(f"render tokens {tuple(pending.shape)} do not match the latent tokens "
                             f"{tuple(output.shape)}")
        return output + pending.to(dtype=output.dtype)

    def render_tokens(self, render_condition):
        """(b, t, v, C, h, w) -> (b * t * v, h / p * w / p, inner_dim) scaled render tokens."""
        r = render_condition.flatten(0, 2).to(dtype=self.render_embed.weight.dtype)
        return self.render_embed(r).flatten(2).transpose(1, 2) * self.render_condition_scale

    def forward(self, sample, timestep=None, *args, render_condition=None, **kwargs):
        if render_condition is None:
            return super().forward(sample, timestep, *args, **kwargs)
        if render_condition.ndim != 6 or sample.ndim != 6:
            raise ValueError("render_condition and sample must both be (b, t, v, C, h, w)")
        if tuple(render_condition.shape[:3]) != tuple(sample.shape[:3]) or \
                tuple(render_condition.shape[-2:]) != tuple(sample.shape[-2:]) or \
                render_condition.shape[3] != self.render_condition_channels:
            raise ValueError(f"render_condition {tuple(render_condition.shape)} does not fit sample "
                             f"{tuple(sample.shape)} with {self.render_condition_channels} channels")
        self._d2_pending = self.render_tokens(render_condition)
        try:
            return super().forward(sample, timestep, *args, **kwargs)
        finally:
            self._d2_pending = None
