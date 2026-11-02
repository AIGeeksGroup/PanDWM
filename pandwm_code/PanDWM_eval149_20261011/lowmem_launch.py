"""Run an existing PanDWM script on a 24 GB GPU (HPC1 A30) without editing it.

Two substitutions, both applied before the target script builds the pipeline:

1. The consistency-distillation pipeline also builds a frozen teacher and an
   EMA copy and moves them to the GPU; the eval scripts free them only after
   construction.  Here they stay on CPU while they exist.
2. The frozen eval config wraps the student (and T5) in single-rank FSDP with
   ``MixedPrecision(param_dtype=float16)`` and CPU offload.  FSDP first moves
   the whole fp32 student (13.4 GB) to the GPU, which next to fp16 T5 does not
   fit 24 GB.  The single-rank wrapper is replaced by what it computes: the
   parameters cast to the mixed-precision param dtype on the GPU, and floating
   root inputs cast the same way (FSDP's ``cast_root_forward_inputs``).
   Without mixed precision the module is just moved to the device.

3. The block-causal FlexAttention mask is built on CPU and moved to the GPU
   (identical BlockMask; building it on the GPU needs a 14 GB dense temporary).

Evaluations launched this way are compared only with each other (same GPU,
same path), not with numbers produced through real FSDP on HPC2.

  python lowmem_launch.py run_eval149_videos.py <its arguments...>
"""

import runpy
import sys

import torch
from torch.utils._pytree import tree_map

import dwm_pano.pipelines.ctsd_pano as ctsd_pano
from dwm_pano.pipelines.ctsd_pano_consistency_distillation import (
    PanoCrossviewTemporalConsistencyDistillation as CD,
)


class _CastRoot(torch.nn.Module):
    def __init__(self, module, dtype):
        super().__init__()
        self.module, self.dtype = module, dtype

    def forward(self, *args, **kwargs):
        cast = lambda x: x.to(self.dtype) if torch.is_tensor(x) and x.is_floating_point() else x
        return self.module(*tree_map(cast, args), **tree_map(cast, kwargs))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class _CpuModule(torch.nn.Module):
    """Frozen encoder kept on CPU (fp32); outputs come back on the GPU in the
    dtype the GPU copy would have produced."""

    def __init__(self, module, device, out_dtype):
        super().__init__()
        self.module, self.target, self.out_dtype = module.float().cpu(), device, out_dtype

    def forward(self, *args, **kwargs):
        to_cpu = lambda x: x.cpu() if torch.is_tensor(x) else x
        back = lambda x: (x.to(self.target, self.out_dtype) if torch.is_tensor(x) and x.is_floating_point()
                          else x.to(self.target) if torch.is_tensor(x) else x)
        with torch.no_grad():
            out = self.module(*tree_map(to_cpu, args), **tree_map(to_cpu, kwargs))
        return tree_map(back, out)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def _single_rank_fsdp(module, device_id=None, mixed_precision=None, **_):
    device = torch.device("cuda", torch.cuda.current_device() if device_id is None else device_id)
    if type(module).__name__ == "T5EncoderModel":
        # 9.5 GB in fp16; on a 24 GB GPU it leaves no room for sampling.
        return _CpuModule(module, device, next(module.parameters()).dtype)
    dtype = getattr(mixed_precision, "param_dtype", None)
    if dtype is None:
        return module.to(device)
    for p in module.parameters():
        p.data = p.data.to(dtype)
    return _CastRoot(module.to(device), dtype)


import dwm_pano.models.flex_block_causal as _fbc

_make_mask = _fbc.make_blockwise_causal_block_mask


def _mask_built_on_cpu(device, num_frames=21, frame_seqlen=1560, num_frame_per_block=1,
                       local_attn_size=-1):
    """Same BlockMask as make_blockwise_causal_block_mask, with the block
    structure computed on CPU (create_block_mask(_compile=False) materialises
    the dense L x L mask: ~14 GB for 19 frames x 2304 tokens) and the mask_mod
    then rebound to a GPU copy of its index table for the attention kernel."""
    import math
    from torch.nn.attention.flex_attention import create_block_mask
    total = num_frames * frame_seqlen
    padded = math.ceil(total / 128) * 128 - total
    step = frame_seqlen * num_frame_per_block
    ends_cpu = torch.zeros(total + padded, dtype=torch.long)
    for start in range(0, total, step):
        ends_cpu[start:start + step] = start + step

    def make_mod(ends):
        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) \
                | (q_idx == kv_idx)
        return attention_mask

    mask = create_block_mask(make_mod(ends_cpu), B=None, H=None, Q_LEN=total + padded,
                             KV_LEN=total + padded, _compile=False, device="cpu").to(device)
    mask.mask_mod = make_mod(ends_cpu.to(device))
    return mask


CD._wrap_auxiliary_model = lambda self, model: model   # stays on CPU, dropped by the script
ctsd_pano.FSDP = _single_rank_fsdp
_fbc.make_blockwise_causal_block_mask = _mask_built_on_cpu

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
