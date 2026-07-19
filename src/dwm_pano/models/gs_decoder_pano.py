"""Feed-forward Gaussian decoder for the 2-half pano latent (WorldSplat-style).

`MultiViewAttnProcessor` and the `GSDecoder` UNet are copied from worldsplat
`gs_decoder/gs_head.py` (cross-view attention runs over the 2 pano halves);
the per-pixel head follows `gs_decoder/pixel_decoder.py` but back-projects
along pano unit rays with a range residual on top of a metric range prior:
    range = clamp(prior_range + disparity, min=near)
    means = unit_direction * range + xyz_offset
The returned gaussians tensor [B, N, 14] = [means(3), rgb(3), opacity(1),
rotation(4), scale(3)] matches the gsplat.rasterization layout used by
debug_pano_gs_geometry.py.
"""
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_up_block

# rgb(3) + disparity(1) + opacity(1) + scale(3) + rotation(4)
# + xyz_offset(3) + segmentation logits(2, static/dynamic)
GAUSSIAN_PARAMETER_SPLIT = (3, 1, 1, 3, 4, 3, 2)


class MultiViewAttnProcessor:
    """Copied from worldsplat gs_head.py: attention across all views."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        num_views: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

        hidden_states = einops.rearrange(
            hidden_states, "(b v) t c -> b (v t) c", v=num_views)
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(
            batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(
            batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = einops.rearrange(
            hidden_states, "b (v t) c -> (b v) t c", v=num_views)
        batch_size = hidden_states.shape[0]

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class GSDecoder(nn.Module):
    """Copied from worldsplat gs_head.py: cross-view UNet, 8x upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        attention_block_count: int = 3,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[-1],
            kernel_size=3, stride=1, padding=1)
        self.gn = nn.GroupNorm(
            norm_num_groups, block_out_channels[-1], affine=True)

        def make_attention_block():
            return Attention(
                block_out_channels[-1],
                heads=1,
                dim_head=block_out_channels[-1],
                rescale_output_factor=1,
                eps=1e-6,
                norm_num_groups=norm_num_groups,
                spatial_norm_dim=None,
                residual_connection=True,
                processor=MultiViewAttnProcessor(),
                bias=True,
                upcast_softmax=True,
                _from_deprecated_attn_block=True,
                dropout=0.0)

        self.attn_block_pre = nn.ModuleList(
            [make_attention_block() for _ in range(attention_block_count)])
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn="silu",
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=False)
        self.attn_block_post = nn.ModuleList(
            [make_attention_block() for _ in range(attention_block_count)])

        self.up_blocks = nn.ModuleList([])
        reversed_channels = list(reversed(block_out_channels))
        output_channel = reversed_channels[0]
        for block_index in range(len(block_out_channels)):
            previous_output_channel = output_channel
            output_channel = reversed_channels[block_index]
            is_final_block = block_index == len(block_out_channels) - 1
            up_block = get_up_block(
                "UpDecoderBlock2D",
                num_layers=layers_per_block + 1,
                in_channels=previous_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
                resnet_time_scale_shift="group")
            self.up_blocks.append(up_block)

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, view_count = x.shape[:2]
        x = einops.rearrange(x, "b v c h w -> (b v) c h w")
        x = self.conv_in(x)
        x = self.gn(x)
        for attn_block in self.attn_block_pre:
            x = attn_block(x, num_views=view_count)
        x = self.mid_block(x)
        for attn_block in self.attn_block_post:
            x = attn_block(x, num_views=view_count)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


class PanoGaussianDecoder(nn.Module):
    """Per-pixel Gaussian prediction from pano-half latents and unit rays."""

    def __init__(
        self,
        latent_channels: int = 16,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        attention_block_count: int = 3,
        use_range_prior: bool = False,
        near: float = 0.5,
        far: float = 400.0,
        scale_base: float = 0.01,
    ):
        super().__init__()
        self.use_range_prior = use_range_prior
        self.near = near
        self.far = far
        self.scale_base = scale_base
        self.decoder = GSDecoder(
            in_channels=latent_channels + 3,
            out_channels=sum(GAUSSIAN_PARAMETER_SPLIT),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            attention_block_count=attention_block_count)

    def forward(self, latents, latent_directions, pixel_directions,
                prior_range=None):
        """latents (B,V,C,h,w); latent_directions (B,V,3,h,w);
        pixel_directions (B,V,H,W,3); prior_range (B,V,H,W) meters,
        consumed only when use_range_prior."""
        batch_size, view_count = latents.shape[:2]
        decoder_input = torch.cat([latents, latent_directions], dim=2)
        raw = self.decoder(decoder_input)
        height, width = raw.shape[-2:]
        parameters = einops.rearrange(
            raw, "(b v) c h w -> b (v h w) c", b=batch_size)
        rgb, disparity, opacity, scales, rotations, xyz_offset, segments = \
            parameters.split(GAUSSIAN_PARAMETER_SPLIT, dim=-1)

        rgb = torch.sigmoid(rgb)
        opacity = torch.sigmoid(opacity)
        scales = torch.exp(scales) * self.scale_base
        rotations = F.normalize(rotations, dim=-1)

        directions = einops.rearrange(
            pixel_directions, "b v h w c -> b (v h w) c")
        if self.use_range_prior:
            prior = einops.rearrange(prior_range, "b v h w -> b (v h w) 1")
            ranges = torch.clamp(prior + disparity, min=self.near)
        else:
            inverse_span = 1.0 / self.near - 1.0 / self.far
            ranges = 1.0 / (
                torch.sigmoid(disparity) * inverse_span + 1.0 / self.far)
        means = directions * ranges + xyz_offset

        gaussians = torch.cat(
            [means, rgb, opacity, rotations, scales], dim=-1)
        predicted_range = einops.rearrange(
            ranges[..., 0], "b (v h w) -> b v h w",
            v=view_count, h=height, w=width)
        predicted_segments = einops.rearrange(
            segments, "b (v h w) c -> b v h w c",
            v=view_count, h=height, w=width)
        return gaussians, predicted_range, predicted_segments
