import warnings
warnings.filterwarnings("ignore")
# torch first (same lazy-init ordering fix as train.py).
import torch
import transformers.utils.logging as _tfm_log
_tfm_log.set_verbosity_error()
_tfm_log.disable_progress_bar()
import diffusers.utils.logging as _diff_log
_diff_log.set_verbosity_error()
_diff_log.disable_progress_bar()
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

import argparse
import datetime
import json
import os
import time
from collections import defaultdict

import dwm.common
import torch.nn.functional as F


_FA3_HITS = [0]
_FA3_MISS = [0]


def install_fa3_sdpa():
    """Route eligible SDPA calls (non-causal, hdim 64, fp16/bf16, no mask,
    no dropout) through FlashAttention-3. Falls back to real SDPA otherwise.
    Faithful: same math, so spatial/crossview attention swaps transparently;
    temporal uses flex_attention (a different call) and is never touched."""
    import flash_attn_interface as _fa3
    _orig = F.scaled_dot_product_attention

    def _shim(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
              scale=None, **kw):
        if (attn_mask is None and dropout_p == 0.0 and q.dim() == 4
                and q.shape[-1] == 64
                and q.dtype in (torch.float16, torch.bfloat16)):
            _FA3_HITS[0] += 1
            o = _fa3.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                softmax_scale=scale, causal=is_causal)
            if isinstance(o, tuple):
                o = o[0]
            return o.transpose(1, 2)
        _FA3_MISS[0] += 1
        return _orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kw)

    F.scaled_dot_product_attention = _shim
    torch.nn.functional.scaled_dot_product_attention = _shim


def is_leaf_kernel(name: str) -> bool:
    # Keep only actual device kernels; drop CPU dispatcher ops and FSDP/module
    # record_function annotations that double-count their children's CUDA time.
    n = name
    if n.startswith(("aten::", "autograd::", "nccl:", "c10d::", "cudaLaunch",
                     "Optimizer.", "ProfilerStep")):
        return False
    if ".forward" in n or ".backward" in n or n == "record_param_comms":
        return False
    return True


def bucket(name: str) -> str:
    n = name.lower()
    # FlexAttention (temporal causal) compiles to templated triton kernels
    # (triton_tem_fused*). Separate it from the spatial/crossview SDPA-flash.
    if "triton_tem" in n or "flex" in n:
        return "Attn:flex(temporal)"
    if any(k in n for k in (
            "flash", "fmha", "sdpa", "scaled_dot", "cudnn_attention", "mha")):
        return "Attn:flash(spatial)"
    if any(k in n for k in (
            "nccl", "allgather", "all_gather", "reduce_scatter",
            "reducescatter", "allreduce", "all_reduce", "broadcast")):
        return "Comm/NCCL"
    if any(k in n for k in (
            "implicit_gemm", "fprop", "winograd")) or "conv" in n:
        return "Conv(VAE)"
    if any(k in n for k in (
            "gemm", "cutlass", "ampere", "cublas", "sm90_xmma_gemm",
            "wgmma", "s16816gemm", "dgrad", "wgrad")):
        return "GEMM/Linear"
    if "triton_poi" in n or "triton_red" in n or any(k in n for k in (
            "elementwise", "vectorized", "reduce_kernel", "layer_norm",
            "softmax", "gelu", "silu", "catarray", "cat_", "index", "fill",
            "copy")):
        return "Elementwise/Norm"
    return "Other"


def self_cuda(evt) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(evt, attr, None)
        if v is not None:
            return float(v)
    return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-path", type=str, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the 3 DiT wrappers (A/B vs baseline)")
    parser.add_argument("--orig-params", action="store_true",
                        help="inject use_orig_params=True into FSDP settings "
                             "(required for dynamo+FSDP); in-memory only")
    parser.add_argument("--fa3", action="store_true",
                        help="monkeypatch F.scaled_dot_product_attention to route "
                             "non-causal hdim64 fp16/bf16 attn through FlashAttention-3 "
                             "(spatial/crossview only; temporal flex untouched)")
    parser.add_argument("--mode", choices=["train", "infer"], default="train",
                        help="train = full DMD step (gen+critic+real_score CFG); "
                             "infer = generator 4-step forward only, NO CFG, "
                             "NO real_score/critic (deployment path)")
    parser.add_argument("--warmup-steps", type=int, default=6)
    parser.add_argument("--profile-cycle", type=int, default=5,
                        help="steps to profile; dfake_gen_update_ratio so one "
                             "generator update is included")
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    torch.manual_seed(config["generator_seed"])

    if args.orig_params:
        # In-memory only — does NOT touch the shared config file or the running
        # training. Required for torch.compile + FSDP.
        for k in ("ddp_wrapper_settings", "t5_fsdp_wrapper_settings"):
            if k in config["pipeline"]["common_config"]:
                config["pipeline"]["common_config"][k]["use_orig_params"] = True

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(config["device"], local_rank)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend=config["ddp_backend"],
        timeout=datetime.timedelta(minutes=60))

    if "global_state" in config:
        for key, value in config["global_state"].items():
            dwm.common.global_state[key] = \
                dwm.common.create_instance_from_config(value)

    rank0 = torch.distributed.get_rank() == 0
    world = torch.distributed.get_world_size()

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=args.output_path, config=config,
        device=device, resume_from=None)
    if rank0:
        print(f"[profile] pipeline loaded, world_size={world}", flush=True)

    if args.fa3:
        install_fa3_sdpa()
        if rank0:
            print("[profile] FA3 SDPA shim installed", flush=True)

    if args.compile:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 128  # B and 2B (CFG) shapes
        try:
            for attr in ("model_wrapper", "fake_score_wrapper",
                         "real_score_wrapper"):
                m = getattr(pipeline, attr, None)
                if m is not None:
                    setattr(pipeline, attr, torch.compile(m))
            if rank0:
                print("[profile] torch.compile applied to 3 DiT wrappers "
                      "(mode=default)", flush=True)
        except Exception as e:
            if rank0:
                print(f"[profile] COMPILE SETUP FAILED: {type(e).__name__}: {e}",
                      flush=True)
            raise

    training_dataset = dwm.common.create_instance_from_config(
        config["training_dataset"])
    sampler = torch.utils.data.distributed.DistributedSampler(
        training_dataset, shuffle=config["data_shuffle"],
        seed=config["generator_seed"])
    loader = torch.utils.data.DataLoader(
        training_dataset,
        **dwm.common.instantiate_config(config["training_dataloader"]),
        sampler=sampler)
    sampler.set_epoch(0)
    it = iter(loader)
    batch = next(it)
    if rank0:
        print("[profile] first batch fetched", flush=True)

    ratio = pipeline.dfake_gen_update_ratio

    # The unit of work being profiled.
    if args.mode == "infer":
        # Generator 4-step forward ONLY — no CFG, no real_score/critic.
        # This is the deployment/inference path (pipeline.inference_pipeline
        # builds conditions with do_classifier_free_guidance=False).
        seq_len = pipeline.inference_config["sequence_length_per_iteration"]
        view_count = batch["vae_images"].shape[2]
        latent_shape = pipeline._get_latent_shape(batch, seq_len, view_count)

        def run_one(_step):
            with torch.no_grad():
                pipeline.inference_pipeline(
                    latent_shape, batch, "pt",
                    image_latents=None, reference_frame_count=0)
    else:
        def run_one(step):
            pipeline.train_step(batch, step)

    # warmup: cross a generator step (train) / compile both paths.
    gstep = 0
    for _ in range(args.warmup_steps):
        gstep += 1
        ws = time.time()
        run_one(gstep)
        torch.cuda.synchronize()
        if rank0:
            tag = "infer" if args.mode == "infer" else (
                ("gen+" if gstep % ratio == 0 else "") + "critic")
            print(f"[profile] warmup {gstep} ({tag}): {time.time()-ws:.1f}s",
                  flush=True)
    torch.distributed.barrier()
    if rank0:
        print(f"[profile] warmup done ({args.warmup_steps})", flush=True)

    # For train mode, align so the profiled window starts on a generator step.
    if args.mode == "train":
        while (gstep + 1) % ratio != 0:
            gstep += 1
            run_one(gstep)
    torch.cuda.synchronize()
    torch.distributed.barrier()

    t0 = time.time()
    n_gen = 0
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, with_stack=False,
    ) as prof:
        for _ in range(args.profile_cycle):
            gstep += 1
            if args.mode == "train" and gstep % ratio == 0:
                n_gen += 1
            run_one(gstep)
        torch.cuda.synchronize()
    wall = time.time() - t0

    if rank0:
        ka = prof.key_averages()
        kernels = [e for e in ka if is_leaf_kernel(e.key) and self_cuda(e) > 0]
        buckets = defaultdict(float)
        total = 0.0
        for evt in kernels:
            t = self_cuda(evt)
            buckets[bucket(evt.key)] += t
            total += t
        n = args.profile_cycle
        print("\n================ PROFILE (rank0) ================", flush=True)
        last = pipeline.loss_report_list[-1] if pipeline.loss_report_list else {}
        import math
        finite = all(
            isinstance(v, (int, float)) and math.isfinite(v)
            for v in last.values()) if last else False
        if args.mode == "infer":
            desc = f"{n} generator 4-step inference forwards (NO CFG)"
        else:
            desc = f"{n} train steps ({n_gen} generator + {n} critic)"
        print(f"profiled {desc}, "
              f"wall={wall:.1f}s  ->  {wall/n:.2f} s/unit", flush=True)
        if args.fa3:
            print(f"FA3 shim: routed {_FA3_HITS[0]} attn calls to FA3, "
                  f"{_FA3_MISS[0]} fell back to SDPA", flush=True)
        if args.mode == "train":
            print(f"compile={args.compile}  last-step loss row={last}  "
                  f"finite={finite}", flush=True)
        print(f"leaf-kernel CUDA time (sum/{n} steps): {total/1e3:.1f} ms  "
              f"-> {total/1e3/n:.0f} ms/step  "
              f"(GPU busy ~{100*(total/1e3/n)/(wall/n*1000):.0f}% of wall)",
              flush=True)
        attn = buckets["Attn:flex(temporal)"] + buckets["Attn:flash(spatial)"]
        print(f"\n>>> ATTENTION TOTAL: {100*attn/max(total,1):.1f}% of GPU "
              f"compute  (flex {100*buckets['Attn:flex(temporal)']/max(total,1):.1f}% "
              f"+ flash {100*buckets['Attn:flash(spatial)']/max(total,1):.1f}%)",
              flush=True)
        print("\n--- GPU kernel time by bucket (leaf kernels only) ---",
              flush=True)
        for k in sorted(buckets, key=buckets.get, reverse=True):
            print(f"  {k:22s} {buckets[k]/1e3:9.1f} ms   "
                  f"{100*buckets[k]/max(total,1):5.1f}%", flush=True)
        print("\n--- top 30 leaf kernels (self CUDA time) ---", flush=True)
        for evt in sorted(kernels, key=self_cuda, reverse=True)[:30]:
            st = self_cuda(evt)
            print(f"  {st/1e3:8.1f} ms  [{bucket(evt.key):20s}] {evt.key[:80]}",
                  flush=True)
        print("=================================================\n", flush=True)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
