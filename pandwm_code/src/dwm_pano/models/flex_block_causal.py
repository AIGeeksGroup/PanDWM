"""FlexAttention block-causal mask and attention processor for video temporal attention.

The block-causal mask construction is copied from LongLive
`wan/modules/causal_model.py:_prepare_blockwise_causal_attn_mask`, without the
`local_attn_size` sliding-window branch.

`BlockCausalFlexAttnProcessor` is a fork of diffusers' `AttnProcessor2_0`: when
`attention_mask` is a `BlockMask` it dispatches to `flex_attention`, otherwise
it behaves exactly like `AttnProcessor2_0`.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask, create_block_mask, flex_attention,
)

# Compiled for speed; the default mode compiles once and stays quiet.
_compiled_flex_attention = torch.compile(flex_attention, dynamic=False)


def make_blockwise_causal_block_mask(
    device, num_frames: int = 21, frame_seqlen: int = 1560,
    num_frame_per_block: int = 1, local_attn_size: int = -1,
) -> BlockMask:
    """
    we will divide the token sequence into the following format
    [1 latent frame] [1 latent frame] ... [1 latent frame]
    We use flexattention to construct the attention mask
    """
    total_length = num_frames * frame_seqlen

    # we do right padding to get to a multiple of 128
    padded_length = math.ceil(total_length / 128) * 128 - total_length

    ends = torch.zeros(total_length + padded_length,
                       device=device, dtype=torch.long)

    # Block-wise causal mask will attend to all elements that are before the end of the current chunk
    frame_indices = torch.arange(
        start=0,
        end=total_length,
        step=frame_seqlen * num_frame_per_block,
        device=device
    )

    for tmp in frame_indices:
        ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
            frame_seqlen * num_frame_per_block

    def attention_mask(b, h, q_idx, kv_idx):
        if local_attn_size == -1:
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
        else:
            return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

    block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                   KV_LEN=total_length + padded_length, _compile=False, device=device)

    return block_mask


class BlockCausalFlexAttnProcessor:
    """Forked from `diffusers.models.attention_processor.AttnProcessor2_0`.

    Identical except for the final attention call: when `attention_mask` is a
    `BlockMask`, it dispatches to FlexAttention with right-padding to a
    multiple of 128 (a FlexAttention requirement, as in LongLive
    `causal_model.py`).
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask=None,
        temb: Optional[torch.Tensor] = None,
        *args, **kwargs,
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
            hidden_states.shape
            if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        is_block_mask = isinstance(attention_mask, BlockMask)

        if attention_mask is not None and not is_block_mask:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

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
        key = key.view(
            batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(
            batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if is_block_mask:
            # FlexAttention path: pad the sequence dim to a multiple of 128.
            seq_len = query.shape[2]
            pad_len = math.ceil(seq_len / 128) * 128 - seq_len
            if pad_len > 0:
                pad_q = torch.zeros(
                    query.shape[0], query.shape[1], pad_len, head_dim,
                    device=query.device, dtype=query.dtype)
                query = torch.cat([query, pad_q], dim=2)
                key = torch.cat([key, pad_q], dim=2)
                value = torch.cat([value, pad_q], dim=2)
            hidden_states = _compiled_flex_attention(
                query, key, value, block_mask=attention_mask)
            if pad_len > 0:
                hidden_states = hidden_states[:, :, :seq_len, :]
        else:
            # Default path (same as AttnProcessor2_0).
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask,
                dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
