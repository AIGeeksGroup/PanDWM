#!/usr/bin/env python3
"""Single-GPU diagnostic: does the CF-DMD+GAN discriminator use a "history shortcut"?

D = H(F_features): forward hook on fake_score.transformer_blocks[12], target tokens of
the 16 trained frames mean-pooled -> 1536-d -> head H.  GAN real input = GT history
(frames 32-34) + noised GT target (35-50); fake = generated history (two no-grad student
rollout windows) + noised generated target.  F's causal temporal blocks (after blocks
1, 5, 9) let target tokens read the clean history before block 12, so D could separate
real/fake by history source, which G cannot change (history is detached).

For each clip this script builds, from the SAME training code path:
  h_GT, x_GT : GT latents of window 32-50
  h_G        : 2 no-grad full 4-step rollout windows (PanoContextForcedDMD._rollout_window)
  x_Gk       : student sample on h_G that exits after denoising step k (k=1..4), all from
               the same initial noise (generator state restored), via
               PanoCrossviewTemporalDMD._run_generator with a forced exit index
and scores D (PanoContextForcedDMDv2._disc_logits) on matched arms with IDENTICAL
(timestep, epsilon) per clip and setting:
  a: h_GT + x_GT   b: h_G + x_G4   c: h_G + x_GT   d: h_GT + x_G4   e: h_G + x_G1
  (+ f: h_G + x_G2, g: h_G + x_G3 unless --no-mid-exits; they let us rebuild the
   training fake mixture, since training fakes come from a uniform random exit).
Contrasts (paired per clip):
  history swap (same target):  a - c  (at x_GT),  d - b  (at x_G4)
  target swap (same history):  a - d  (at h_GT),  c - b  (at h_G)
  NOTE: the task text listed (a)-(d), (c)-(b) as "history swaps"; with the arm
  definitions above those pairs share the history and differ in the target, so they are
  reported here as TARGET swaps (and vice versa).
Gradient balance on x_G4 w.r.t. the target latent only:
  ||d L_DMD / d x||  (same pseudo-loss as _generator_step_cf, teacher with CFG)
  ||d (gan_generator_weight * softplus(-D)) / d x||

Modes:
  --dry-run   CPU only: build the 1-GPU config copy, load the disc head, parse checkpoint
              headers (legacy-pickle header only / zip mmap; no tensor data is read) and
              compare key/shape layout with a meta-device model.  Safe on a login node.
  (default)   full diagnostic on one CUDA device; no torch.distributed, no FSDP.
"""
# Import torch first (see vendor_snapshot/train.py header: import-order CUDA init bug).
import torch

import argparse
import contextlib
import copy
import datetime
import hashlib
import json
import math
import os
import pickle
import socket
import sys
import time
import zipfile
from pathlib import Path

import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
DEFAULT_CONFIG = EXP_ROOT / "configs" / "cfdmd_A_gan_hpc3_8gpu.json"
DEFAULT_CKPT_DIR = "/path/to/pandwm_root/pandwm_runs/cfdmd_20261008_A/gan/milestones"

ARMS_CORE = {
    # name: (history source, target source)
    "a_hGT_xGT": ("GT", "GT"),
    "b_hG_xG4": ("G", "G4"),
    "c_hG_xGT": ("G", "GT"),
    "d_hGT_xG4": ("GT", "G4"),
    "e_hG_xG1": ("G", "G1"),
}
ARMS_MID = {
    "f_hG_xG2": ("G", "G2"),
    "g_hG_xG3": ("G", "G3"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--step", type=int, default=1585)
    p.add_argument("--generator", choices=["raw", "ema"], default="raw",
                   help="raw G produced the fakes D was trained on (default)")
    p.add_argument("--teacher-ckpt", default=None,
                   help="override real_teacher_checkpoint_path (default: from config)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--splits", default="train", help="comma list from {train,val}")
    p.add_argument("--num-clips", type=int, default=32, help="clips for the train split")
    p.add_argument("--val-clips", type=int, default=16, help="clips for the val split")
    p.add_argument("--clip-source", choices=["recent_trained", "random"],
                   default="recent_trained",
                   help="train split: recent_trained = clips the run consumed just before "
                        "--step (DistributedSampler reconstruction); random = seeded sample")
    p.add_argument("--train-world-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--score-t", default="50,200,500,800,950",
                   help="raw timesteps (before shift/clamp/lookup) for D scoring")
    p.add_argument("--score-random-draws", type=int, default=2,
                   help="extra settings drawn from the training timestep distribution")
    p.add_argument("--grad-t", default="200,500,800,950")
    p.add_argument("--grad-random-draws", type=int, default=2)
    p.add_argument("--grad-clips", type=int, default=12,
                   help="clips per split that get the gradient-balance test (-1 = all)")
    p.add_argument("--grad-loss-scale", type=float, default=65536.0,
                   help="static loss scale for the fp16 GAN backward (GradScaler stand-in)")
    p.add_argument("--no-mid-exits", action="store_true")
    p.add_argument("--no-early-exit", action="store_true",
                   help="run F to the end for D scoring (default stops after block 12, "
                        "which cannot change the block-12 features)")
    p.add_argument("--time-budget-min", type=float, default=160.0,
                   help="stop starting new clips after this many minutes; report is kept")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", "--cpu-check", dest="dry_run", action="store_true")
    p.add_argument("--dry-run-dataset", action="store_true",
                   help="dry run also instantiates the datasets (loads the 1.7 GB train pkl)")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------------
def ckpt_paths(args, config):
    d = Path(args.ckpt_dir)
    g_name = f"{args.step}.pth" if args.generator == "raw" else f"{args.step}_ema.pth"
    return {
        "generator": d / g_name,
        "critic": d / f"{args.step}_critic.pth",
        "disc_head": d / f"{args.step}_disc_head.pth",
        "teacher": Path(args.teacher_ckpt or config["pipeline"]["real_teacher_checkpoint_path"]),
    }


def make_diag_config(config, paths):
    """1-GPU copy of the training config.  Only distribution/IO keys change."""
    cfg = copy.deepcopy(config)
    cfg.pop("global_state", None)          # device_mesh [1,8]
    cfg.pop("optimizer", None)             # no G update in a diagnostic
    p = cfg["pipeline"]
    cc = p["common_config"]
    # FSDP settings reference the device mesh; torch.distributed is not initialised
    # here, so every wrap branch in the pipeline takes its single-device path.
    cc.pop("ddp_wrapper_settings", None)
    cc.pop("t5_fsdp_wrapper_settings", None)
    p.pop("metrics", None)                 # FID/FVD nets are not needed
    for k in ("checkpoint_steps", "checkpoint_seconds", "target_steps"):
        p.pop(k, None)                     # Resumable-only kwargs (we build v4)
    p["model_checkpoint_path"] = str(paths["generator"])
    p["fake_score_checkpoint_path"] = str(paths["critic"])
    p["real_teacher_checkpoint_path"] = str(paths["teacher"])
    return cfg


# ----------------------------------------------------------------------------------
# checkpoint header inspection (no tensor data read)
# ----------------------------------------------------------------------------------
_STORAGE_DTYPES = {
    "FloatStorage": "torch.float32", "HalfStorage": "torch.float16",
    "BFloat16Storage": "torch.bfloat16", "DoubleStorage": "torch.float64",
    "LongStorage": "torch.int64", "IntStorage": "torch.int32",
    "ShortStorage": "torch.int16", "CharStorage": "torch.int8",
    "ByteStorage": "torch.uint8", "BoolStorage": "torch.bool",
}


class _StorageMarker:
    def __init__(self, name):
        self.name = name


class _TensorInfo:
    def __init__(self, shape, dtype):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype


def _fake_rebuild(name):
    if name == "_rebuild_tensor_v2" or name == "_rebuild_tensor":
        def rebuild(storage, offset, size, stride, *rest):
            return _TensorInfo(size, storage[1])
        return rebuild
    if name.startswith("_rebuild_parameter"):
        return lambda data, *rest: data
    raise NotImplementedError(f"unsupported rebuild function {name}")


class _HeaderUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name.startswith("_rebuild"):
            return _fake_rebuild(name)
        if module == "torch._tensor" and name == "_rebuild_from_type_v2":
            return lambda func, new_type, args, state: func(*args)
        if module == "torch" and name.endswith("Storage"):
            return _StorageMarker(name)
        return super().find_class(module, name)

    def persistent_load(self, pid):
        typ = pid[1]
        name = typ.name if isinstance(typ, _StorageMarker) else str(typ)
        return ("__storage__", _STORAGE_DTYPES.get(name, name))


def read_ckpt_layout(path):
    """{key: (shape, dtype)} without reading tensor payloads."""
    path = str(path)
    if zipfile.is_zipfile(path):
        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        out = {k: (tuple(v.shape), str(v.dtype)) for k, v in sd.items()}
        del sd
        return out, "zip(mmap)"
    with open(path, "rb") as f:
        magic = pickle.load(f)
        if magic != 0x1950a86a20f9469cfc6c:
            raise ValueError(f"{path}: not a torch legacy checkpoint")
        pickle.load(f)   # protocol version
        pickle.load(f)   # sys info
        obj = _HeaderUnpickler(f).load()
    out = {k: (v.shape, v.dtype) for k, v in obj.items()}
    return out, "legacy-pickle(header only)"


def compare_layouts(ref, other):
    ref_k, oth_k = set(ref), set(other)
    shape_mismatch = [k for k in ref_k & oth_k if tuple(ref[k][0]) != tuple(other[k][0])]
    return {"n_ref": len(ref_k), "n_other": len(oth_k),
            "missing_in_ckpt": sorted(ref_k - oth_k)[:20],
            "n_missing_in_ckpt": len(ref_k - oth_k),
            "unexpected_in_ckpt": sorted(oth_k - ref_k)[:20],
            "n_unexpected_in_ckpt": len(oth_k - ref_k),
            "shape_mismatch": sorted(shape_mismatch)[:20],
            "n_shape_mismatch": len(shape_mismatch),
            "ok": ref_k == oth_k and not shape_mismatch}


def build_disc_head(model_cfg):
    hidden = int(model_cfg["num_attention_heads"]) * int(model_cfg["attention_head_dim"])
    # identical to pipeline.py PanoContextForcedDMD.__init__ (lines 75-79)
    return torch.nn.Sequential(
        torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, 256),
        torch.nn.SiLU(), torch.nn.Linear(256, 1))


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------
# dry run
# ----------------------------------------------------------------------------------
def dry_run(args, config, diag_cfg, paths, out_dir):
    import dwm.common
    report = {"mode": "dry_run", "host": socket.gethostname(),
              "time": datetime.datetime.now().astimezone().isoformat(),
              "args": vars(args), "paths": {k: str(v) for k, v in paths.items()}}
    report["exists"] = {k: Path(v).exists() for k, v in paths.items()}
    report["sizes"] = {k: Path(v).stat().st_size for k, v in paths.items() if Path(v).exists()}
    data_paths = {}
    for split, key in (("train", "training_dataset"), ("val", "validation_dataset")):
        b = config[key]["base_dataset"]
        for k in ("pkl_path", "rgb_dir", "bbox_dir", "hdmap_dir", "nuscenes_meta_dir"):
            data_paths[f"{split}.{k}"] = Path(b[k]).exists()
    data_paths["sd35"] = Path(config["pipeline"]["pretrained_model_name_or_path"]).exists()
    report["data_paths_exist"] = data_paths

    # imports of the real training modules (CPU only)
    from dwm_pano_cfdmd.pipeline import CONTEXT, STRIDE, WINDOW
    import dwm_pano_cfdmd.pipeline_v4 as v4
    report["imports"] = {"pipeline_v4": v4.__file__, "CONTEXT": CONTEXT,
                         "STRIDE": STRIDE, "WINDOW": WINDOW}

    # disc head: strict load into the training architecture
    head = build_disc_head(diag_cfg["pipeline"]["model"])
    sd = torch.load(paths["disc_head"], map_location="cpu")
    head.load_state_dict(sd)          # strict
    with torch.no_grad():
        logit0 = float(head(torch.zeros(1, head[0].normalized_shape[0])).item())
    report["disc_head"] = {"keys": {k: [list(v.shape), str(v.dtype)] for k, v in sd.items()},
                           "strict_load": "ok", "logit_of_zero_feature": logit0,
                           "finite": all(torch.isfinite(v).all().item() for v in sd.values())}

    # model layout on the meta device vs checkpoint headers
    try:
        with torch.device("meta"):
            model = dwm.common.create_instance_from_config(diag_cfg["pipeline"]["model"])
        ref = {k: (tuple(v.shape), str(v.dtype)) for k, v in model.state_dict().items()}
        report["meta_model"] = {"n_keys": len(ref),
                                "n_params": int(sum(p.numel() for p in model.parameters())),
                                "block12": type(model.transformer_blocks[12]).__name__,
                                "n_transformer_blocks": len(model.transformer_blocks),
                                "temporal_block_layers": list(model.temporal_block_layers)}
        del model
    except Exception as exc:  # noqa: BLE001
        ref = None
        report["meta_model"] = {"error": repr(exc)}

    layouts = {}
    for name in ("generator", "critic", "teacher"):
        if not Path(paths[name]).exists():
            layouts[name] = {"error": "missing"}
            continue
        t0 = time.time()
        try:
            lay, how = read_ckpt_layout(paths[name])
        except Exception as exc:  # noqa: BLE001
            layouts[name] = {"error": repr(exc)}
            continue
        dtypes = {}
        for _, (_, dt) in lay.items():
            dtypes[dt] = dtypes.get(dt, 0) + 1
        entry = {"format": how, "n_keys": len(lay), "dtypes": dtypes,
                 "seconds": round(time.time() - t0, 2)}
        if ref is not None:
            entry["vs_model"] = compare_layouts(ref, lay)
        layouts[name] = entry
        layouts[name]["_layout"] = lay
    # cross-checks between checkpoints (all three are the same DiT class)
    if all("_layout" in layouts.get(n, {}) for n in ("generator", "critic")):
        layouts["generator_vs_critic"] = compare_layouts(
            layouts["generator"]["_layout"], layouts["critic"]["_layout"])
    for n in list(layouts):
        if isinstance(layouts[n], dict):
            layouts[n].pop("_layout", None)
    report["checkpoints"] = layouts

    # timestep settings resolved exactly as _sample_dmd_timestep does (CPU replica of
    # the arithmetic only for display; the GPU run calls the real method)
    p = diag_cfg["pipeline"]
    shift = float(p.get("timestep_shift", 5.0))
    lo, hi = float(p.get("min_step", 20)), float(p.get("max_step", 980))
    disp = {}
    for t_raw in parse_int_list(args.score_t) + parse_int_list(args.grad_t):
        t = shift * (t_raw / 1000) / (1.0 + (shift - 1) * (t_raw / 1000)) * 1000
        disp[str(t_raw)] = round(min(max(t, lo), hi), 2)
    report["shifted_timesteps_preview"] = disp

    if args.dry_run_dataset:
        ds = build_dataset(diag_cfg, "train", context_windows=p["context_windows"])
        report["train_dataset_len"] = len(ds)
        idx = recent_trained_indices(len(ds), args.step, args.train_world_size,
                                     4, config.get("generator_seed", 0))
        report["recent_trained_first4"] = idx
        dsv = build_dataset(diag_cfg, "val", context_windows=p["context_windows"])
        report["val_dataset_len_seq51"] = len(dsv)

    report["diag_config_pipeline_overrides"] = {
        k: p[k] for k in ("model_checkpoint_path", "fake_score_checkpoint_path",
                          "real_teacher_checkpoint_path")}
    ok = (report["exists"]["generator"] and report["exists"]["critic"]
          and report["exists"]["disc_head"] and report["exists"]["teacher"]
          and all(layouts.get(n, {}).get("vs_model", {}).get("ok", False)
                  for n in ("generator", "critic", "teacher")))
    report["dry_run_ok"] = bool(ok)
    write_json(out_dir / "dry_run_report.json", report)
    with open(out_dir / "diag_config_used.json", "w") as f:
        json.dump(diag_cfg, f, indent=2)
    print(json.dumps({k: report[k] for k in ("dry_run_ok", "exists", "disc_head",
                                             "meta_model", "checkpoints",
                                             "shifted_timesteps_preview")},
                     indent=1, default=str))
    return 0 if ok else 2


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------
def build_dataset(diag_cfg, split, context_windows):
    import dwm.common
    key = "training_dataset" if split == "train" else "validation_dataset"
    ds_cfg = copy.deepcopy(diag_cfg[key])
    if split == "val":
        # The config's val set is 35-frame clips; context forcing needs
        # 3 + 16 * (context_windows + 1) = 51 frames (stride stays 20 s).
        ds_cfg["base_dataset"]["sequence_length"] = 3 + 16 * (context_windows + 1)
    return dwm.common.create_instance_from_config(ds_cfg)


def recent_trained_indices(n_dataset, step, world, n_clips, seed):
    """Dataset indices consumed by the run just before `step`.

    Rebuilds torch DistributedSampler(shuffle=True, seed=generator_seed) at epoch 0
    (fix_training_data_order=true), drop_last=False padding, rank r reads
    indices[r::world]; global step k (0-based) reads position k on every rank
    (resume offsets reproduce the same order, vendor_snapshot/train.py:216-245).
    """
    g = torch.Generator()
    g.manual_seed(seed + 0)
    perm = torch.randperm(n_dataset, generator=g).tolist()
    total = int(math.ceil(n_dataset / world)) * world
    perm += perm[: total - n_dataset]
    out, seen = [], set()
    k = step - 1
    while len(out) < n_clips and k >= 0:
        for r in range(world):
            idx = perm[r + world * k]
            if idx not in seen:
                seen.add(idx)
                out.append({"index": int(idx), "rank": r, "global_step": k})
            if len(out) >= n_clips:
                break
        k -= 1
    return out


def random_indices(n_dataset, n_clips, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return [{"index": int(i)} for i in torch.randperm(n_dataset, generator=g)[:n_clips].tolist()]


def json_safe(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
        return v.item() if v.numel() == 1 else v.tolist()
    if isinstance(v, (list, tuple)):
        return [json_safe(i) for i in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


# ----------------------------------------------------------------------------------
# GPU pieces
# ----------------------------------------------------------------------------------
class _StopAfterFeature(Exception):
    pass


def _cast_floating(x, dtype):
    """Mirror of FSDP MixedPrecision(cast_root_forward_inputs=True) input casting."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype) if torch.is_floating_point(x) and x.dtype != dtype else x
    if isinstance(x, dict):
        return {k: _cast_floating(v, dtype) for k, v in x.items()}
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return type(x)(*[_cast_floating(v, dtype) for v in x])
    if isinstance(x, (list, tuple)):
        return type(x)(_cast_floating(v, dtype) for v in x)
    return x  # e.g. flex-attention BlockMask (integer tensors only)


class Fp16RootCast(torch.nn.Module):
    """Stand-in for the training FSDP wrapper on a single GPU.

    Training: FSDP(MixedPrecision(param_dtype=float16)) -> every parameter is cast to
    fp16 for compute (buffers untouched) and floating root inputs are cast to fp16.
    Here the parameters are stored in fp16 directly and inputs are cast the same way.
    If `early_exit` is set, a hook stops the forward right after the GAN feature block
    while the pipeline is capturing features without autograd.
    """

    def __init__(self, module, dtype=torch.float16):
        super().__init__()
        self.module = module
        self.dtype = dtype
        self.early_exit = False

    def forward(self, *args, **kwargs):
        args, kwargs = _cast_floating((args, kwargs), self.dtype)
        try:
            return self.module(*args, **kwargs)
        except _StopAfterFeature:
            return (None,), None, None


def cast_params_fp16(model):
    for p in model.parameters():
        p.data = p.data.to(torch.float16)


def build_pipeline(diag_cfg, device, out_dir):
    import dwm.common
    from dwm_pano_cfdmd.pipeline_v4 import PanoContextForcedDMDv4

    class DiagPipeline(PanoContextForcedDMDv4):
        def _wrap_auxiliary_model(self, model):
            cast_params_fp16(model)        # after the fp32 checkpoint was loaded
            model.to(self.device)
            return Fp16RootCast(model)

    torch.manual_seed(diag_cfg["generator_seed"])      # as vendor_snapshot/train.py:134
    pcfg = copy.deepcopy(diag_cfg["pipeline"])
    pcfg.pop("_class_name")
    kwargs = dwm.common.instantiate_config(pcfg)
    scratch = out_dir / "_pipeline_scratch"            # pipeline writes a TB log dir here
    pipe = DiagPipeline(output_path=str(scratch), config=diag_cfg, device=device,
                        resume_from=None, **kwargs)
    # G was built by the base class in fp32 on the device; cast like FSDP param_dtype.
    cast_params_fp16(pipe.model)
    pipe.model_wrapper = Fp16RootCast(pipe.model)
    for m in (pipe.model, pipe.real_score, pipe.fake_score, pipe.disc_head):
        m.requires_grad_(False)
    pipe.optimizer = None
    pipe.critic_optimizer = None
    pipe.disc_optimizer = None
    torch.cuda.empty_cache()
    return pipe


def install_early_exit(pipe, block_index):
    wrapper = pipe.fake_score_wrapper

    def stop_hook(module, inputs, output):
        if pipe._capture and wrapper.early_exit and not torch.is_grad_enabled():
            raise _StopAfterFeature()

    # registered after the pipeline's _feature_hook, so the features are captured first
    return pipe.fake_score.transformer_blocks[block_index].register_forward_hook(stop_hook)


@contextlib.contextmanager
def forced_exit(pipe, idx):
    """Run the real _run_generator with exit index `idx` (0-based denoising step).

    _run_generator draws exit_idx with torch.randint and broadcasts it from rank 0;
    without a process group we substitute the broadcast by a fill with `idx`.
    """
    orig = torch.distributed.broadcast

    def fake_broadcast(tensor, src=0, *a, **k):
        tensor.fill_(idx)

    prev = pipe.random_exit_step
    torch.distributed.broadcast = fake_broadcast
    pipe.random_exit_step = True
    try:
        yield
    finally:
        torch.distributed.broadcast = orig
        pipe.random_exit_step = prev


def draw_timestep(pipe, shape, dtype, t_raw=None):
    """(ts, sigmas_b) from the real _sample_dmd_timestep; t_raw pins the uniform draw."""
    if t_raw is None:
        return pipe._sample_dmd_timestep(shape, dtype)
    lo, hi = pipe.min_score_timestep, pipe.num_train_timestep
    pipe.min_score_timestep, pipe.num_train_timestep = int(t_raw), int(t_raw) + 1
    try:
        return pipe._sample_dmd_timestep(shape, dtype)
    finally:
        pipe.min_score_timestep, pipe.num_train_timestep = lo, hi


@contextlib.contextmanager
def pinned_timestep(pipe, ts, sigmas_b):
    """Make _noised_with_clean_history use a given (ts, sigma) for every arm."""
    pipe._sample_dmd_timestep = lambda shape, dtype: (ts.clone(), sigmas_b.to(dtype))
    try:
        yield
    finally:
        del pipe._sample_dmd_timestep


def disc_logit(pipe, x, history, ts, sigmas_b, eps, cond):
    with pinned_timestep(pipe, ts, sigmas_b):
        noisy, ts_a, _, _ = pipe._noised_with_clean_history(x, history, noise=eps)
    return pipe._disc_logits(noisy, ts_a, cond)


def grad_balance(pipe, x_g, h_g, cond, cond_cfg, target_mask, ts, sigmas_b, eps,
                 loss_scale):
    """Gradients of the two generator-loss terms w.r.t. the target latent x_G4."""
    # --- DMD pseudo-loss, as pipeline.py _generator_step_cf lines 236-245 ---
    pipe.fake_score_wrapper.train()            # state F is in during a training G step
    x_leaf = x_g.detach().clone().requires_grad_(True)
    with torch.no_grad():
        x0_det = x_leaf.detach()
        with pinned_timestep(pipe, ts, sigmas_b):
            noisy, ts_n, sig_n, _ = pipe._noised_with_clean_history(x0_det, h_g, noise=eps)
        g = pipe._kl_grad_with_history(noisy, ts_n, sig_n, x0_det, cond, cond_cfg,
                                       target_mask)
    target = (x0_det.double() - g.double()).detach()
    mask = target_mask.to(torch.float64)
    while mask.ndim < x_leaf.ndim:
        mask = mask.unsqueeze(-1)
    diff = (x_leaf.double() - target) * mask
    loss_dmd = 0.5 * diff.square().sum() / mask.expand_as(diff).sum().clamp(min=1.0)
    grad_dmd = torch.autograd.grad(loss_dmd, x_leaf)[0].float()

    # --- GAN generator term, as _generator_step_cf lines 247-250 ---
    w = pipe.gan_generator_weight
    scale = float(loss_scale)
    grad_gan, logit, adv = None, float("nan"), float("nan")
    for _ in range(6):
        x2 = x_g.detach().clone().requires_grad_(True)
        with pinned_timestep(pipe, ts, sigmas_b):
            noisy_g, ts_g, _, _ = pipe._noised_with_clean_history(x2, h_g, noise=eps)
        lg = pipe._disc_logits(noisy_g, ts_g, cond)
        adv_t = F.softplus(-lg).mean()
        gr = torch.autograd.grad(scale * w * adv_t.float(), x2)[0].float() / scale
        logit, adv = float(lg.item()), float(adv_t.item())
        if torch.isfinite(gr).all():
            grad_gan = gr
            break
        scale /= 256.0
    pipe.fake_score_wrapper.eval()

    tgt = (slice(None), slice(3, None))
    gd, gg = grad_dmd[tgt], (grad_gan[tgt] if grad_gan is not None else None)
    rec = {"loss_dmd": float(loss_dmd.item()),
           "dmd_mean_abs_g_all_frames": float(g.float().abs().mean().item()),
           "dmd_mean_abs_g_target": float(g[tgt].float().abs().mean().item()),
           "grad_dmd_l2": float(gd.norm().item()),
           "grad_dmd_hist_l2": float(grad_dmd[:, :3].norm().item()),
           "d_fake_logit": logit, "adv_gen": adv, "gan_weight": w,
           "loss_scale_used": scale}
    if gg is not None:
        rec.update({
            "grad_gan_l2": float(gg.norm().item()),
            "grad_gan_hist_l2": float(grad_gan[:, :3].norm().item()),
            "ratio_gan_over_dmd": float(gg.norm().item() / max(gd.norm().item(), 1e-30)),
            "cosine_dmd_gan": float(F.cosine_similarity(
                gd.flatten().double(), gg.flatten().double(), dim=0).item()),
        })
    else:
        rec["grad_gan_l2"] = None
    return rec


def process_clip(pipe, batch, args, clip_seed, settings_score, settings_grad, do_grad,
                 early_exit_check):
    from dwm_pano_cfdmd.pipeline import CONTEXT, STRIDE, WINDOW
    dev = pipe.device
    torch.manual_seed(clip_seed)
    torch.cuda.manual_seed_all(clip_seed)
    pipe.generator.manual_seed(clip_seed)
    pipe.real_score_wrapper.eval()
    pipe.fake_score_wrapper.eval()
    t_start = time.time()

    view_count = batch["vae_images"].shape[2]
    frames = batch["vae_images"].shape[1]
    need = CONTEXT + STRIDE * (pipe.context_windows + 1)
    if frames < need:
        raise ValueError(f"clip has {frames} frames, needs {need}")
    latent_shape = pipe._get_latent_shape(batch, WINDOW, view_count)
    start = STRIDE * pipe.context_windows

    rec = {"timing": {}}
    with torch.no_grad():
        # pipeline.py train_step lines 201-220, with context_forcing_prob = 1
        gt = pipe._vae_encode(batch["vae_images"][:, :start + WINDOW])
        history = gt[:, :CONTEXT]
        for w in range(pipe.context_windows):
            cond_w = pipe._conditions(pipe._window_batch(batch, STRIDE * w), latent_shape, False)
            history = pipe._rollout_window(latent_shape, cond_w, history)[:, -CONTEXT:]
        h_g = history.detach()
        gt_window = gt[:, start:start + WINDOW]
        h_gt = gt_window[:, :CONTEXT]
        wbatch = pipe._window_batch(batch, start)
        cond = pipe._conditions(wbatch, latent_shape, False)
        cond_cfg = pipe._conditions(wbatch, latent_shape, True)
        target_mask = torch.ones(latent_shape[:3], dtype=torch.bool, device=dev)
        target_mask[:, :CONTEXT] = False
        rec["timing"]["history"] = round(time.time() - t_start, 1)

        # student targets on h_G, all exits from the same initial noise / noise stream
        exits = [0, 3] if args.no_mid_exits else [0, 1, 2, 3]
        gen_state = pipe.generator.get_state()
        xg = {}
        for k in exits:
            pipe.generator.set_state(gen_state)
            with forced_exit(pipe, k):
                xg[k] = pipe._run_generator(latent_shape, cond, image_latents=h_g,
                                            reference_frame_count=CONTEXT,
                                            grad_on_last=False).detach()
        rec["timing"]["targets"] = round(time.time() - t_start, 1)

        hist = {"GT": h_gt, "G": h_g}
        tgts = {"GT": gt_window, "G1": xg[0], "G4": xg[3]}
        if not args.no_mid_exits:
            tgts.update({"G2": xg[1], "G3": xg[2]})
        arms = dict(ARMS_CORE)
        if not args.no_mid_exits:
            arms.update(ARMS_MID)

        # latent statistics (target frames) for context
        tsl = (slice(None), slice(CONTEXT, None))
        rec["latent_stats"] = {
            "hist_mse_G_vs_GT": float((h_g - h_gt).float().square().mean().item()),
            "hist_std_G": float(h_g.float().std().item()),
            "hist_std_GT": float(h_gt.float().std().item()),
            **{f"target_std_{n}": float(v[tsl].float().std().item()) for n, v in tgts.items()},
            **{f"target_mse_{n}_vs_GT": float((v[tsl] - gt_window[tsl]).float().square().mean().item())
               for n, v in tgts.items() if n != "GT"},
        }

        # ---- D scoring: identical (t, eps) across arms within a setting ----
        pipe.fake_score_wrapper.early_exit = not args.no_early_exit
        scores = {}
        for s in settings_score:
            ts, sig = draw_timestep(pipe, gt_window.shape[:3], gt_window.dtype, s["t_raw"])
            eps = torch.randn_like(gt_window)
            row = {"t_model": float(ts.flatten()[0].item()),
                   "sigma": float(sig.flatten()[0].item()), "logits": {}}
            for arm, (hs, xs) in arms.items():
                row["logits"][arm] = float(disc_logit(pipe, tgts[xs], hist[hs], ts, sig, eps,
                                                      cond).item())
            if early_exit_check is not None and not early_exit_check and not args.no_early_exit:
                pipe.fake_score_wrapper.early_exit = False
                full = float(disc_logit(pipe, tgts["GT"], hist["GT"], ts, sig, eps, cond).item())
                pipe.fake_score_wrapper.early_exit = True
                early_exit_check.update({"setting": s["name"], "early": row["logits"]["a_hGT_xGT"],
                                         "full": full,
                                         "abs_diff": abs(full - row["logits"]["a_hGT_xGT"])})
            scores[s["name"]] = row
        pipe.fake_score_wrapper.early_exit = False
        rec["scores"] = scores
        rec["timing"]["scoring"] = round(time.time() - t_start, 1)

    # ---- gradient balance on x_G4 ----
    if do_grad:
        grads = {}
        for s in settings_grad:
            ts, sig = draw_timestep(pipe, gt_window.shape[:3], gt_window.dtype, s["t_raw"])
            eps = torch.randn_like(gt_window)
            r = grad_balance(pipe, xg[3], h_g, cond, cond_cfg, target_mask, ts, sig, eps,
                             args.grad_loss_scale)
            r.update({"t_model": float(ts.flatten()[0].item()),
                      "sigma": float(sig.flatten()[0].item())})
            grads[s["name"]] = r
        rec["grad"] = grads
        rec["timing"]["grad"] = round(time.time() - t_start, 1)
    rec["timing"]["total"] = round(time.time() - t_start, 1)
    return rec


# ----------------------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------------------
def _stats(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    n = len(vals)
    if n == 0:
        return {"n": 0}
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    med = sorted(vals)[n // 2] if n % 2 else 0.5 * sum(sorted(vals)[n // 2 - 1:n // 2 + 1])
    return {"n": n, "mean": m, "std": sd, "sem": sem, "median": med,
            "ci95": [m - 1.96 * sem, m + 1.96 * sem] if n > 1 else None,
            "frac_pos": sum(v > 0 for v in vals) / n}


CONTRASTS = {
    # name: (plus arm, minus arm, meaning)
    "hist_swap_at_xGT(a-c)": ("a_hGT_xGT", "c_hG_xGT", "history GT->G with GT target"),
    "hist_swap_at_xG4(d-b)": ("d_hGT_xG4", "b_hG_xG4", "history GT->G with G4 target"),
    "target_swap_at_hGT(a-d)": ("a_hGT_xGT", "d_hGT_xG4", "target GT->G4 with GT history"),
    "target_swap_at_hG(c-b)": ("c_hG_xGT", "b_hG_xG4", "target GT->G4 with G history"),
    "total_real_minus_fake(a-b)": ("a_hGT_xGT", "b_hG_xG4", "training real vs 4-step fake"),
    "G4_minus_G1_at_hG(b-e)": ("b_hG_xG4", "e_hG_xG1", "4-step vs 1-step-exit target"),
}


def aggregate(clips, setting_names, grad_names, mid_exits):
    out = {"n_clips": len(clips), "arms": {}, "contrasts": {}, "decomposition": {},
           "grad_balance": {}, "train_mixture_check": {}}
    groups = {name: [name] for name in setting_names if not name.startswith("rand")}
    rand = [n for n in setting_names if n.startswith("rand")]
    if rand:
        groups["random(train-dist)"] = rand
    arm_names = list(ARMS_CORE) + (list(ARMS_MID) if mid_exits else [])
    for gname, members in groups.items():
        # arm-level: all (clip, draw) samples
        out["arms"][gname] = {}
        for arm in arm_names:
            vals = [c["scores"][m]["logits"][arm] for c in clips for m in members
                    if m in c.get("scores", {})]
            st = _stats(vals)
            st["frac_classified_real"] = st.pop("frac_pos", None)
            out["arms"][gname][arm] = st
        if len(members) == 1 and clips and members[0] in clips[0].get("scores", {}):
            out["arms"][gname]["_t_model"] = clips[0]["scores"][members[0]]["t_model"]
            out["arms"][gname]["_sigma"] = clips[0]["scores"][members[0]]["sigma"]
        # paired contrasts: per clip, averaged over draws
        out["contrasts"][gname] = {}
        per_clip = []
        for c in clips:
            ms = [m for m in members if m in c.get("scores", {})]
            if not ms:
                continue
            per_clip.append({arm: sum(c["scores"][m]["logits"][arm] for m in ms) / len(ms)
                             for arm in arm_names})
        for cname, (pa, ma, meaning) in CONTRASTS.items():
            st = _stats([r[pa] - r[ma] for r in per_clip])
            st["meaning"] = meaning
            out["contrasts"][gname][cname] = st
        hist_main = [0.5 * ((r["a_hGT_xGT"] - r["c_hG_xGT"]) + (r["d_hGT_xG4"] - r["b_hG_xG4"]))
                     for r in per_clip]
        tgt_main = [0.5 * ((r["a_hGT_xGT"] - r["d_hGT_xG4"]) + (r["c_hG_xGT"] - r["b_hG_xG4"]))
                    for r in per_clip]
        inter = [(r["a_hGT_xGT"] - r["c_hG_xGT"]) - (r["d_hGT_xG4"] - r["b_hG_xG4"])
                 for r in per_clip]
        hm, tm = _stats(hist_main), _stats(tgt_main)
        share = None
        if hm.get("n") and tm.get("n"):
            denom = abs(hm["mean"]) + abs(tm["mean"])
            share = abs(hm["mean"]) / denom if denom > 0 else None
        out["decomposition"][gname] = {"history_main_effect": hm, "target_main_effect": tm,
                                       "interaction": _stats(inter),
                                       "history_share_of_main_effects": share}
        if mid_exits:
            # training fakes come from a uniform random exit over the 4 steps
            fake_arms = ["e_hG_xG1", "f_hG_xG2", "g_hG_xG3", "b_hG_xG4"]
            fv = [c["scores"][m]["logits"][a] for c in clips for m in members
                  if m in c.get("scores", {}) for a in fake_arms]
            rv = [c["scores"][m]["logits"]["a_hGT_xGT"] for c in clips for m in members
                  if m in c.get("scores", {})]
            if fv and rv:
                out["train_mixture_check"][gname] = {
                    "disc_real_acc": sum(v > 0 for v in rv) / len(rv),
                    "disc_fake_acc_uniform_exit_mixture": sum(v < 0 for v in fv) / len(fv),
                    "disc_loss_equiv": (sum(F.softplus(-torch.tensor(rv)).tolist()) / len(rv)
                                        + sum(F.softplus(torch.tensor(fv)).tolist()) / len(fv)),
                    "adv_gen_equiv_mixture": sum(F.softplus(-torch.tensor(fv)).tolist()) / len(fv),
                }
    ggroups = {n: [n] for n in grad_names if not n.startswith("rand")}
    grand = [n for n in grad_names if n.startswith("rand")]
    if grand:
        ggroups["random(train-dist)"] = grand
    for gname, members in ggroups.items():
        rows = [c["grad"][m] for c in clips if "grad" in c for m in members if m in c["grad"]]
        if not rows:
            continue
        ratios = [r["ratio_gan_over_dmd"] for r in rows if r.get("ratio_gan_over_dmd")]
        out["grad_balance"][gname] = {
            "grad_dmd_l2": _stats([r["grad_dmd_l2"] for r in rows]),
            "grad_gan_l2": _stats([r["grad_gan_l2"] for r in rows]),
            "ratio_gan_over_dmd": _stats(ratios),
            "ratio_geomean": (math.exp(sum(math.log(x) for x in ratios) / len(ratios))
                              if ratios else None),
            "cosine_dmd_gan": _stats([r.get("cosine_dmd_gan") for r in rows]),
            "d_fake_logit": _stats([r["d_fake_logit"] for r in rows]),
            "adv_gen": _stats([r["adv_gen"] for r in rows]),
            "dmd_mean_abs_g_all_frames": _stats([r["dmd_mean_abs_g_all_frames"] for r in rows]),
            "n_gan_grad_nonfinite": sum(r["grad_gan_l2"] is None for r in rows),
        }
    return out


def _ff(x, digits=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _fmt(st, key="mean", digits=3):
    if not st or not st.get("n"):
        return "n/a"
    v = st.get(key)
    return "n/a" if v is None else f"{v:+.{digits}f}"


def write_summary(path, report):
    L = []
    m = report["meta"]
    L.append(f"Disc history-shortcut diagnostic  step={m['args']['step']} G={m['args']['generator']}"
             f"  host={m['host']}  finished={m.get('finished')}")
    L.append(f"early-exit check: {report.get('early_exit_check')}")
    if m.get("truncated"):
        L.append("WARNING: time budget reached, fewer clips than requested.")
    for split, agg in report["splits"].items():
        L.append("")
        L.append(f"=== split {split}: {agg['n_clips']} clips ===")
        for g, arms in agg["arms"].items():
            hdr = f"-- setting {g}"
            if "_t_model" in arms:
                hdr += f"  (t_model={arms['_t_model']:.1f}, sigma={arms['_sigma']:.3f})"
            L.append(hdr)
            for arm, st in arms.items():
                if arm.startswith("_"):
                    continue
                L.append(f"   {arm:12s} logit mean {_fmt(st)} std {_fmt(st, 'std')}"
                         f"  frac_real {_ff(st.get('frac_classified_real'))}")
            for cname, st in agg["contrasts"][g].items():
                L.append(f"   {cname:28s} {_fmt(st)} +- {_fmt(st, 'sem')} (sem)"
                         f"  frac>0 {_ff(st.get('frac_pos'))}")
            d = agg["decomposition"][g]
            share = d["history_share_of_main_effects"]
            L.append(f"   history main {_fmt(d['history_main_effect'])}  target main "
                     f"{_fmt(d['target_main_effect'])}  interaction {_fmt(d['interaction'])}"
                     f"  history share {share if share is None else round(share, 3)}")
            if g in agg["train_mixture_check"]:
                L.append(f"   train-mixture check: {json.dumps(agg['train_mixture_check'][g])}")
        for g, gb in agg["grad_balance"].items():
            L.append(f"-- grad balance {g}: |dDMD/dx| median {_fmt(gb['grad_dmd_l2'], 'median', 6)}"
                     f"  |d(w*adv)/dx| median {_fmt(gb['grad_gan_l2'], 'median', 6)}"
                     f"  ratio GAN/DMD median {_fmt(gb['ratio_gan_over_dmd'], 'median', 4)}"
                     f" geomean {gb['ratio_geomean']}  cos {_fmt(gb['cosine_dmd_gan'])}"
                     f"  D(fake) {_fmt(gb['d_fake_logit'])}  nonfinite {gb['n_gan_grad_nonfinite']}")
    L.append("")
    L.append("Reading guide (heuristic, not a pre-registered test):")
    L.append(" * history shortcut supported if |history main| >> |target main| (share > ~0.67),")
    L.append("   c (real target, G history) is mostly classified fake and d (fake target,")
    L.append("   GT history) mostly real, at most timesteps;")
    L.append(" * target-driven D if target main dominates and c/d follow their targets;")
    L.append(" * a large positive interaction with c and d both classified fake means D only")
    L.append("   accepts real-history AND real-target (any generated part is flagged); then the")
    L.append("   history still gives G an un-fixable penalty on every fake;")
    L.append(" * b-e > 0 means the 1-step-exit targets are easier to reject than 4-step ones.")
    L.append(" * train-mixture check should reproduce the logged disc_*_acc ~ 1.0 at the")
    L.append("   random(train-dist) setting; if not, the 1-GPU replica differs from training.")
    Path(path).write_text("\n".join(L) + "\n")


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def parse_int_list(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


# ----------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    t_begin = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config) as f:
        config = json.load(f)
    paths = ckpt_paths(args, config)
    diag_cfg = make_diag_config(config, paths)
    if args.dry_run:
        return dry_run(args, config, diag_cfg, paths, out_dir)

    if not torch.cuda.is_available():
        print("CUDA not available; use --dry-run on CPU", file=sys.stderr)
        return 3
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True   # as scripts/train_resumable.py
    import dwm.common
    import dwm_pano_cfdmd.pipeline as cf_pipeline
    import dwm_pano_cfdmd.pipeline_v2 as cf_v2
    import dwm_pano_cfdmd.pipeline_v4 as cf_v4
    import dwm_pano.pipelines.ctsd_pano_dmd as dmd_base

    for k in ("generator", "critic", "disc_head", "teacher"):
        if not paths[k].exists():
            raise FileNotFoundError(paths[k])
    with open(out_dir / "diag_config_used.json", "w") as f:
        json.dump(diag_cfg, f, indent=2)

    meta = {"args": vars(args), "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(device),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "started": datetime.datetime.now().astimezone().isoformat(),
            "paths": {k: str(v) for k, v in paths.items()},
            "config_path": args.config, "config_sha256": sha256_file(args.config),
            "source_sha256": {Path(m.__file__).name: sha256_file(m.__file__) for m in
                              (cf_pipeline, cf_v2, cf_v4, dmd_base)},
            "deviations_from_training": [
                "single GPU, no FSDP: parameters stored in fp16 (== FSDP param_dtype cast), "
                "root inputs cast to fp16 as FSDP cast_root_forward_inputs; no 0*output term",
                "real and fake arms share (t, eps) per clip/setting (training draws them "
                "independently); fixed-t settings pin the uniform draw before the real "
                "shift/clamp/sigma lookup",
                "x_Gk exits are forced (training draws exit uniformly, broadcast from rank 0)",
                "GAN backward uses a static loss scale instead of the live GradScaler scale",
                "val clips use sequence_length 51 (config val set is 35 frames)",
            ]}
    print(f"[diag] building pipeline on {device} ...", flush=True)
    pipe = build_pipeline(diag_cfg, device, out_dir)
    head_sd = torch.load(paths["disc_head"], map_location="cpu")
    pipe.disc_head.load_state_dict(head_sd)        # strict
    pipe.disc_head.eval()
    install_early_exit(pipe, diag_cfg["pipeline"].get("gan_feature_block", 12))
    meta["build_seconds"] = round(time.time() - t_begin, 1)
    meta["mem_after_build_GB"] = round(torch.cuda.memory_allocated(device) / 2**30, 2)
    meta["denoising_step_list"] = json_safe(pipe.denoising_step_list)
    meta["denoising_sigmas"] = json_safe(pipe.denoising_sigmas)
    print(f"[diag] pipeline ready in {meta['build_seconds']} s, "
          f"{meta['mem_after_build_GB']} GB allocated", flush=True)

    score_settings = [{"name": f"t{t}", "t_raw": t} for t in parse_int_list(args.score_t)]
    score_settings += [{"name": f"rand{i}", "t_raw": None} for i in range(args.score_random_draws)]
    grad_settings = [{"name": f"t{t}", "t_raw": t} for t in parse_int_list(args.grad_t)]
    grad_settings += [{"name": f"rand{i}", "t_raw": None} for i in range(args.grad_random_draws)]

    collate = dwm.common.create_instance_from_config(
        diag_cfg["training_dataloader"]["collate_fn"])
    report = {"meta": meta, "splits": {}, "early_exit_check": {}}
    clips_all = {}
    jsonl = open(out_dir / "per_clip.jsonl", "a")
    truncated = False
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        ds = build_dataset(diag_cfg, split, diag_cfg["pipeline"]["context_windows"])
        n_req = args.num_clips if split == "train" else args.val_clips
        if split == "train" and args.clip_source == "recent_trained":
            picks = recent_trained_indices(len(ds), args.step, args.train_world_size, n_req,
                                           diag_cfg.get("generator_seed", 0))
        else:
            picks = random_indices(len(ds), n_req, args.seed + (0 if split == "train" else 1))
        print(f"[diag] split {split}: dataset {len(ds)} items, {len(picks)} clips", flush=True)
        clips_all[split] = []
        for ci, pick in enumerate(picks):
            if (time.time() - t_begin) / 60.0 > args.time_budget_min:
                truncated = True
                print("[diag] time budget reached; stopping", flush=True)
                break
            item = ds[int(pick["index"])]
            batch = collate([item])
            clip_seed = args.seed * 1000003 + (0 if split == "train" else 500000) + ci
            do_grad = args.grad_clips < 0 or ci < args.grad_clips
            check = report["early_exit_check"] if not report["early_exit_check"] else None
            rec = process_clip(pipe, batch, args, clip_seed, score_settings, grad_settings,
                               do_grad, check)
            rec.update({"split": split, "clip_i": ci, "seed": clip_seed, **pick,
                        **{k: json_safe(batch[k]) for k in
                           ("scene_token", "anchor_timestamp_us", "dataset_index")
                           if k in batch}})
            clips_all[split].append(rec)
            jsonl.write(json.dumps(rec) + "\n")
            jsonl.flush()
            a = rec["scores"]
            show = {n: {k[:1]: round(v, 2) for k, v in a[n]["logits"].items()} for n in a}
            print(f"[diag] {split} {ci + 1}/{len(picks)} idx={pick['index']} "
                  f"{rec['timing']}  logits={json.dumps(show)}", flush=True)
            if ci == 0:
                meta["peak_mem_GB_first_clip"] = round(
                    torch.cuda.max_memory_allocated(device) / 2**30, 2)
            report["splits"][split] = aggregate(
                clips_all[split], [s["name"] for s in score_settings],
                [s["name"] for s in grad_settings], not args.no_mid_exits)
            meta["truncated"] = truncated
            write_json(out_dir / "report.json", report)
            write_summary(out_dir / "summary.txt", report)
        if truncated:
            break
    jsonl.close()
    meta["truncated"] = truncated
    meta["finished"] = datetime.datetime.now().astimezone().isoformat()
    meta["wall_minutes"] = round((time.time() - t_begin) / 60.0, 1)
    meta["peak_mem_GB"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    for split in clips_all:
        report["splits"][split] = aggregate(
            clips_all[split], [s["name"] for s in score_settings],
            [s["name"] for s in grad_settings], not args.no_mid_exits)
    write_json(out_dir / "report.json", report)
    write_summary(out_dir / "summary.txt", report)
    print((out_dir / "summary.txt").read_text(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
