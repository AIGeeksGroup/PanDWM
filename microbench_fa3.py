"""Microbenchmark FA3 vs torch SDPA on this model's spatial-attention shape.
Isolates the kernel-level speedup FA3 would give the spatial (non-causal,
fp16, head_dim=64) attention — the only path FA3 can replace here.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

import flash_attn_interface as fa3

torch.manual_seed(0)
dev = "cuda"
dt = torch.float16

# (batch=b*t*v, heads, headdim) fixed by config; sweep seqlen.
B, H, D = 38, 24, 64
SEQS = [1152, 2304, 4096]
ITERS, WARM = 50, 10


def timed(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS  # ms


def run(seq):
    # SDPA layout (B, H, S, D); FA3 layout (B, S, H, D).
    q = torch.randn(B, H, seq, D, device=dev, dtype=dt)
    k = torch.randn(B, H, seq, D, device=dev, dtype=dt)
    v = torch.randn(B, H, seq, D, device=dev, dtype=dt)
    qf, kf, vf = (x.transpose(1, 2).contiguous() for x in (q, k, v))

    def sdpa_auto():
        F.scaled_dot_product_attention(q, k, v)

    def sdpa_flash():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(q, k, v)

    def sdpa_mem():
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            F.scaled_dot_product_attention(q, k, v)

    def fa3_fwd():
        fa3.flash_attn_func(qf, kf, vf, causal=False)

    out = {"seq": seq}
    out["sdpa_auto"] = timed(sdpa_auto)
    try:
        out["sdpa_flash"] = timed(sdpa_flash)
    except Exception as ex:
        out["sdpa_flash"] = f"n/a ({type(ex).__name__})"
    try:
        out["sdpa_mem"] = timed(sdpa_mem)
    except Exception as ex:
        out["sdpa_mem"] = f"n/a ({type(ex).__name__})"
    out["fa3"] = timed(fa3_fwd)
    return out


print(f"shape: B={B} H={H} D={D} fp16 non-causal  (this model's spatial attn)")
print(f"{'seq':>6} {'sdpa_auto':>11} {'sdpa_flash':>11} {'sdpa_mem':>11} "
      f"{'FA3':>9} {'FA3 vs best-SDPA':>18}")
for seq in SEQS:
    r = run(seq)
    sd = [r[k] for k in ("sdpa_auto", "sdpa_flash", "sdpa_mem")
          if isinstance(r[k], float)]
    best = min(sd)
    sp = best / r["fa3"]
    def f(x):
        return f"{x:.3f}ms" if isinstance(x, float) else str(x)
    print(f"{seq:>6} {f(r['sdpa_auto']):>11} {f(r['sdpa_flash']):>11} "
          f"{f(r['sdpa_mem']):>11} {f(r['fa3']):>9} "
          f"{sp:>15.2f}x  (FA3 faster if >1)")

# Correctness sanity: FA3 vs SDPA reference output, seq=2304.
seq = 2304
q = torch.randn(B, H, seq, D, device=dev, dtype=dt)
k = torch.randn(B, H, seq, D, device=dev, dtype=dt)
v = torch.randn(B, H, seq, D, device=dev, dtype=dt)
ref = F.scaled_dot_product_attention(q, k, v)
out = fa3.flash_attn_func(*(x.transpose(1, 2).contiguous() for x in (q, k, v)),
                          causal=False)
out = out.transpose(1, 2)
md = (ref.float() - out.float()).abs().max().item()
print(f"\ncorrectness vs SDPA (seq=2304): max|diff|={md:.4e} "
      f"(fp16 ~1e-2 expected)")
