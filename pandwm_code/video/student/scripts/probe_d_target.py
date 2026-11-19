#!/usr/bin/env python3
"""Feature extraction for a HISTORY-BLIND discriminator probe (step-1585 CF-DMD+GAN run A).

Question: once every discriminator input gets pure-noise history slots (v5 `_disc_input`), is there enough
real-vs-fake signal in the 16 target frames alone, at which noise levels and layers, and are frozen-teacher
features (LADD-style) better than the fake score's?

Per clip (same clips / seeds / rollout code as scripts/diag_disc_shortcut.py, whose helpers are imported, so
the pipeline build, clip selection and history rollout are byte-identical):
  h_G   : 2 no-grad 4-step student rollout windows from the clip's 3 real frames (frames 32-34 of the window)
  x_G4  : student G (1585.pth raw) full 4-step sample on h_G (forced exit = last step), exactly as training
  x_GT  : GT window latents (targets = frames 35-50)
Feature extractor inputs (identical target noise eps and history noise for the real and the fake arm):
  hist=noise   : v5 `_disc_input`: target slots sigma*eps + (1-sigma)*x0 at a fixed raw t mapped through the
                 real shift/clamp/sigma lookup (`_sample_dmd_timestep`), history slots sigma0*N(0,1) with
                 timestep denoising_step_list[0]
  hist=cleanGT : `_noised_with_clean_history` with the GT history (t=0) for BOTH arms (control)
  hist=own     : fake arm with its own generated history h_G (the old training input; real arm = cleanGT real)
Feature sources (forward hooks on transformer_blocks[b], output[1] = hidden states ((b t v), tokens, c)):
  F : fake score (1585_critic.pth), causal mask as in `_disc_logits`
  T : frozen teacher (real_teacher_checkpoint_path = stage1_causal_20000.pth), conditional only (no CFG),
      causal mask as in `_kl_grad_with_history` (real_score_causal)
Stored per (source, history, arm, t_raw, block): per-(target frame, view) token MEAN and token STD (1536-d,
fp16).  The old disc head (1585_disc_head.pth) is applied to the F block-12 global mean (float32) as a
no-retraining reference.  Simple latent statistics of the noised target latents are stored as a baseline.

Results are written per clip (clips/<split>_<i>.pt, atomic) plus progress.jsonl, so partial runs survive.
The probes are fitted by scripts/probe_d_target_fit.py (CPU or GPU) on the saved features.

Modes:
  --dry-run   CPU only (login node): diag dry run (checkpoint headers only, disc head strict load, meta-device
              model) + synthetic checks of every tensor op used here.  No checkpoint tensor data is read.
  (default)   one CUDA device, no torch.distributed, no FSDP.
"""
# Import torch first (see vendor_snapshot/train.py header: import-order CUDA init bug).
import torch

import argparse
import datetime
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diag_disc_shortcut as diag  # noqa: E402  (helpers only; its main() is not run)

EXP_ROOT = HERE.parent
DEFAULT_CONFIG = EXP_ROOT / "configs" / "cfdmd_A_gan_hpc3_8gpu.json"
CONTEXT, WINDOW, STRIDE = 3, 19, 16          # == dwm_pano_cfdmd.pipeline (checked at run time)
N_TARGET = WINDOW - CONTEXT


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--ckpt-dir", default=diag.DEFAULT_CKPT_DIR)
    p.add_argument("--step", type=int, default=1585)
    p.add_argument("--generator", choices=["raw", "ema"], default="raw")
    p.add_argument("--teacher-ckpt", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-clips", type=int, default=48, help="train clips (recent_trained, as the diag)")
    p.add_argument("--val-clips", type=int, default=16, help="val clips (seeded random, as the diag)")
    p.add_argument("--train-world-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blocks", default="4,8,12,16,20,23")
    p.add_argument("--noise-t", default="20,50,100,200,350,500,800,950",
                   help="raw t (before shift/clamp/lookup) for the history-blind (noise history) inputs")
    p.add_argument("--control-t", default="50,200,500,800",
                   help="raw t for the clean-GT-history and own-history controls")
    p.add_argument("--own-sources", default="F", help="sources for the own-history control (F,T or empty)")
    p.add_argument("--no-std", action="store_true", help="store only token means")
    p.add_argument("--time-budget-min", type=float, default=140.0,
                   help="stop starting new clips after this many minutes")
    p.add_argument("--max-clips", type=int, default=-1, help="cap on processed clips (smoke tests)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def parse_list(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


# ----------------------------------------------------------------------------------
# pure tensor ops (also exercised by the CPU dry run)
# ----------------------------------------------------------------------------------
def v5_disc_input(x0, eps, hist_noise, sigmas_b, ts, sigma0, t0):
    """Exactly PanoContextForcedDMDv5._disc_input with the random draws passed in."""
    noisy = sigmas_b * eps + (1.0 - sigmas_b) * x0
    hist = sigma0 * hist_noise
    noisy = torch.cat([hist.to(noisy.dtype), noisy[:, CONTEXT:]], dim=1)
    ts = ts.clone()
    ts[:, :CONTEXT] = t0
    return noisy, ts


def clean_history_input(x0, history, eps, sigmas_b, ts):
    """Exactly PanoContextForcedDMD._noised_with_clean_history with the noise passed in."""
    noisy = sigmas_b * eps + (1.0 - sigmas_b) * x0
    noisy = torch.cat([history.to(noisy.dtype), noisy[:, CONTEXT:]], dim=1)
    ts = ts.clone()
    ts[:, :CONTEXT] = 0
    return noisy, ts


def pool_tokens(h, b, t, v):
    """((b t v), tokens, c) -> per (target frame, view) token mean / std, float32 [b, 16, v, c]."""
    h = h.detach().float().unflatten(0, (b, t, v))[:, CONTEXT:]
    return h.mean(dim=3), h.std(dim=3, unbiased=False)


def latent_stats(z):
    """Per target frame statistics of a latent [1, 19, v, c, H, W] -> [16, 5*c] (float32).

    per channel: mean, std (over views and space), mean |d/dx|, mean |d/dy|, mean |d/dt| (frame f vs f-1;
    the first target frame uses frames 1 vs 0 of the targets)."""
    x = z[0, CONTEXT:].float()                                   # [16, v, c, H, W]
    mean = x.mean(dim=(1, 3, 4))
    std = x.std(dim=(1, 3, 4), unbiased=False)
    gx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean(dim=(1, 3, 4))
    gy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=(1, 3, 4))
    dt = (x[1:] - x[:-1]).abs().mean(dim=(1, 3, 4))
    dt = torch.cat([dt[:1], dt], dim=0)
    return torch.cat([mean, std, gx, gy, dt], dim=1)


class FeatureTap:
    """Forward hooks on model.transformer_blocks[blocks]; pooled features while `active`.

    A stop hook after the deepest block raises diag._StopAfterFeature (caught by diag.Fp16RootCast) so the
    forward ends right after the last needed block (no autograd here)."""

    def __init__(self, model, blocks, view_count=2, seq_len=WINDOW, early_exit=True):
        self.blocks = sorted(blocks)
        self.active = False
        self.early_exit = early_exit
        self.shape = (1, seq_len, view_count)
        self.store = {}
        self.handles = []
        for b in self.blocks:
            self.handles.append(model.transformer_blocks[b].register_forward_hook(self._make(b)))
        self.handles.append(model.transformer_blocks[self.blocks[-1]].register_forward_hook(self._stop))

    def _make(self, index):
        def hook(module, inputs, output):
            if not self.active:
                return
            h = output[1] if isinstance(output, tuple) else output
            self.store[index] = pool_tokens(h, *self.shape)
        return hook

    def _stop(self, module, inputs, output):
        if self.active and self.early_exit and not torch.is_grad_enabled():
            raise diag._StopAfterFeature()

    def take(self):
        out, self.store = self.store, {}
        missing = [b for b in self.blocks if b not in out]
        if missing:
            raise RuntimeError(f"feature hooks did not fire for blocks {missing}")
        return out


# ----------------------------------------------------------------------------------
# CPU dry run
# ----------------------------------------------------------------------------------
def synthetic_checks(blocks):
    """Exercise every tensor op of the GPU path on small synthetic tensors (CPU)."""
    torch.manual_seed(0)
    rep = {}
    shape = (1, WINDOW, 2, 4, 6, 10)
    x0 = torch.randn(shape)
    eps = torch.randn(shape)
    hn = torch.randn(1, CONTEXT, *shape[2:])
    sig = torch.full((1, WINDOW, 2), 0.3).view(1, WINDOW, 2, 1, 1, 1)
    ts = torch.full((1, WINDOW, 2), 300.0)
    noisy, ts2 = v5_disc_input(x0, eps, hn, sig, ts, 1.0, 1000.0)
    assert noisy.shape == x0.shape and torch.equal(noisy[:, :CONTEXT], hn)
    assert torch.allclose(noisy[:, CONTEXT:], 0.3 * eps[:, CONTEXT:] + 0.7 * x0[:, CONTEXT:])
    assert (ts2[:, :CONTEXT] == 1000).all() and (ts2[:, CONTEXT:] == 300).all() and (ts == 300).all()
    hist = torch.randn(1, CONTEXT, *shape[2:])
    noisy_c, ts3 = clean_history_input(x0, hist, eps, sig, ts)
    assert torch.equal(noisy_c[:, :CONTEXT], hist) and (ts3[:, :CONTEXT] == 0).all()
    assert torch.equal(noisy_c[:, CONTEXT:], noisy[:, CONTEXT:])
    # pooling: ((b t v), tokens, c) with frame/view-specific offsets must come back in the right slots
    tok, c = 7, 5
    h = torch.zeros(WINDOW * 2, tok, c)
    for f in range(WINDOW):
        for v in range(2):
            h[f * 2 + v] = 10 * f + v + torch.linspace(-1, 1, tok).unsqueeze(1)
    m, s = pool_tokens(h, 1, WINDOW, 2)
    assert m.shape == (1, N_TARGET, 2, c)
    exp = torch.tensor([[10 * (f + CONTEXT) + v for v in range(2)] for f in range(N_TARGET)]).float()
    assert torch.allclose(m[0, :, :, 0], exp, atol=1e-5)
    assert torch.allclose(s[0, 0, 0, 0], torch.linspace(-1, 1, tok).std(unbiased=False), atol=1e-6)
    # global mean of per-(frame, view) means == mean over (frames, views, tokens) (old D pooling)
    hr = torch.randn(WINDOW * 2, tok, c)
    m2, _ = pool_tokens(hr, 1, WINDOW, 2)
    ref = hr.unflatten(0, (1, WINDOW, 2))[:, CONTEXT:].mean(dim=(1, 2, 3))
    assert torch.allclose(m2.mean(dim=(1, 2)), ref, atol=1e-6)
    ls = latent_stats(x0)
    assert ls.shape == (N_TARGET, 5 * shape[3]) and torch.isfinite(ls).all()
    rep["tensor_ops"] = "ok"

    # hooks + early exit on a tiny stand-in transformer
    class Blk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(c, c)

        def forward(self, hs, ehs):
            return ehs, self.lin(hs)

    class Tiny(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([Blk() for _ in range(n)])
            self.after = 0

        def forward(self, hs):
            ehs = None
            for blk in self.transformer_blocks:
                ehs, hs = blk(hs, ehs)
            self.after += 1
            return (hs,), None, None

    nblk = max(blocks) + 1
    tiny = Tiny(nblk)
    wrapped = diag.Fp16RootCast(tiny, dtype=torch.float32)
    tap = FeatureTap(tiny, blocks)
    tap.active = True
    with torch.no_grad():
        out = wrapped(torch.randn(WINDOW * 2, tok, c))
    tap.active = False
    feats = tap.take()
    assert out[0] == (None,) and tiny.after == 0, "early exit did not stop the forward"
    assert sorted(feats) == sorted(blocks)
    assert feats[blocks[0]][0].shape == (1, N_TARGET, 2, c)
    with torch.no_grad():
        wrapped(torch.randn(WINDOW * 2, tok, c))      # inactive tap: full forward, nothing stored
    assert tiny.after == 1 and tap.store == {}
    rep["hooks_early_exit"] = "ok"
    return rep


def run_dry(args, config, diag_cfg, paths, out_dir):
    dargs = argparse.Namespace(**vars(args))
    dargs.score_t, dargs.grad_t, dargs.dry_run_dataset = args.noise_t, args.control_t, False
    rc = diag.dry_run(dargs, config, diag_cfg, paths, out_dir)
    rep = {"diag_dry_run_rc": rc, "synthetic": synthetic_checks(parse_list(args.blocks))}
    # raw t -> model t preview (same arithmetic as _sample_dmd_timestep, display only)
    p = diag_cfg["pipeline"]
    shift, lo, hi = float(p.get("timestep_shift", 5.0)), float(p.get("min_step", 20)), float(p.get("max_step", 980))
    rep["t_preview"] = {}
    for t_raw in sorted(set(parse_list(args.noise_t) + parse_list(args.control_t))):
        t = shift * (t_raw / 1000) / (1.0 + (shift - 1) * (t_raw / 1000)) * 1000
        rep["t_preview"][t_raw] = round(min(max(t, lo), hi), 2)
    rep["n_forwards_per_clip"] = n_forwards(args)
    diag.write_json(out_dir / "probe_dry_run_report.json", rep)
    print(json.dumps(rep, indent=1))
    return rc


def n_forwards(args):
    own = [s for s in args.own_sources.split(",") if s.strip()]
    return 4 * len(parse_list(args.noise_t)) + (4 + len(own)) * len(parse_list(args.control_t))


# ----------------------------------------------------------------------------------
# GPU path
# ----------------------------------------------------------------------------------
def forward_features(pipe, src, tap, noisy, ts, cond):
    if src == "F":           # as PanoContextForcedDMDv2._disc_logits
        wrapper = pipe.fake_score_wrapper
        c = dict(cond)
        if pipe.fake_score_causal:
            pipe._add_temporal_attention_mask(c, noisy.shape)
    else:                    # as _kl_grad_with_history, conditional branch only (no CFG)
        wrapper = pipe.real_score_wrapper
        c = {k: v for k, v in cond.items() if k != "temporal_attention_mask"}
        if pipe.real_score_causal:
            pipe._add_temporal_attention_mask(c, noisy.shape)
    tap.active = True
    try:
        with torch.no_grad(), pipe.get_autocast_context():
            wrapper(noisy.to(pipe.model_dtype), ts, **c)
    finally:
        tap.active = False
    return tap.take()


def process_clip(pipe, batch, args, clip_seed, taps, blocks, settings):
    from dwm_pano_cfdmd.pipeline import CONTEXT as C_, STRIDE as S_, WINDOW as W_
    assert (C_, S_, W_) == (CONTEXT, STRIDE, WINDOW)
    dev = pipe.device
    # identical seeding / call order to diag_disc_shortcut.process_clip up to x_G4
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
        rec["timing"]["history"] = round(time.time() - t_start, 1)
        with diag.forced_exit(pipe, len(pipe.denoising_step_list) - 1):
            x_g4 = pipe._run_generator(latent_shape, cond, image_latents=h_g,
                                       reference_frame_count=CONTEXT, grad_on_last=False).detach()
        rec["timing"]["target"] = round(time.time() - t_start, 1)
        tsl = (slice(None), slice(CONTEXT, None))
        rec["latent_check"] = {
            "hist_mse_G_vs_GT": float((h_g - h_gt).float().square().mean().item()),
            "target_mse_G4_vs_GT": float((x_g4[tsl] - gt_window[tsl]).float().square().mean().item()),
            "target_std_GT": float(gt_window[tsl].float().std().item()),
            "target_std_G4": float(x_g4[tsl].float().std().item())}

        # probe noise: one target eps and one history-noise draw per clip, shared by every arm / source /
        # history type / t (paired design); a dedicated generator keeps it independent of call order.
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(clip_seed) + 7919)
        eps = torch.randn(gt_window.shape, generator=gen, device=dev, dtype=gt_window.dtype)
        hist_noise = torch.randn(h_gt.shape, generator=gen, device=dev, dtype=gt_window.dtype)
        sigma0 = pipe.denoising_sigmas[0].to(dtype=gt_window.dtype)
        t0 = float(pipe.denoising_step_list[0].item())

        x = {"real": gt_window, "fake": x_g4}
        means, stds, head_logits, lat, tinfo = {}, {}, {}, {}, {}
        for arm in ("real", "fake"):
            lat[f"{arm}|clean"] = latent_stats(x[arm]).cpu()
        n_fwd = 0
        for t_raw in settings["all_t"]:
            ts, sig = diag.draw_timestep(pipe, gt_window.shape[:3], gt_window.dtype, t_raw)
            tinfo[t_raw] = {"t_model": float(ts.flatten()[0].item()), "sigma": float(sig.flatten()[0].item())}
            jobs = []      # (key, src, noisy, ts)
            for arm in ("real", "fake"):
                noisy_t = sig * eps + (1.0 - sig) * x[arm]
                lat[f"{arm}|t{t_raw}"] = latent_stats(noisy_t).cpu()
                if t_raw in settings["noise_t"]:
                    n_in, t_in = v5_disc_input(x[arm], eps, hist_noise, sig, ts, sigma0, t0)
                    for src in ("F", "T"):
                        jobs.append((f"{src}|noise|{arm}|t{t_raw}", src, n_in, t_in))
                if t_raw in settings["control_t"]:
                    n_in, t_in = clean_history_input(x[arm], h_gt, eps, sig, ts)
                    for src in ("F", "T"):
                        jobs.append((f"{src}|cleanGT|{arm}|t{t_raw}", src, n_in, t_in))
                    if arm == "fake":
                        n_in, t_in = clean_history_input(x[arm], h_g, eps, sig, ts)
                        for src in settings["own_sources"]:
                            jobs.append((f"{src}|own|{arm}|t{t_raw}", src, n_in, t_in))
            for key, src, n_in, t_in in jobs:
                feats = forward_features(pipe, src, taps[src], n_in, t_in, cond)
                n_fwd += 1
                means[key] = torch.stack([feats[b][0][0] for b in blocks]).to(torch.float16).cpu()
                if not args.no_std:
                    stds[key] = torch.stack([feats[b][1][0] for b in blocks]).to(torch.float16).cpu()
                if src == "F" and 12 in feats:
                    g = feats[12][0].mean(dim=(1, 2))                 # float32 global mean [1, 1536]
                    head_logits[key] = float(pipe.disc_head(g).squeeze(-1).item())
                del feats
        rec["timing"]["features"] = round(time.time() - t_start, 1)
    rec["n_forwards"] = n_fwd
    rec["t_info"] = tinfo
    rec["old_head_logit"] = head_logits
    payload = {"blocks": blocks, "mean": means, "std": stds, "latstats": lat,
               "old_head_logit": head_logits, "t_info": tinfo}
    return rec, payload


def main(argv=None):
    args = parse_args(argv)
    t_begin = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config) as f:
        config = json.load(f)
    paths = diag.ckpt_paths(args, config)
    diag_cfg = diag.make_diag_config(config, paths)
    blocks = parse_list(args.blocks)
    if args.dry_run:
        return run_dry(args, config, diag_cfg, paths, out_dir)
    if not torch.cuda.is_available():
        print("CUDA not available; use --dry-run on CPU", file=sys.stderr)
        return 3
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    import dwm.common
    import dwm_pano_cfdmd.pipeline as cf_pipeline
    import dwm_pano_cfdmd.pipeline_v4 as cf_v4
    import dwm_pano_cfdmd.pipeline_v5 as cf_v5
    import dwm_pano.pipelines.ctsd_pano_dmd as dmd_base
    for k in ("generator", "critic", "disc_head", "teacher"):
        if not paths[k].exists():
            raise FileNotFoundError(paths[k])
    clip_dir = out_dir / "clips"
    clip_dir.mkdir(exist_ok=True)
    with open(out_dir / "probe_config_used.json", "w") as f:
        json.dump(diag_cfg, f, indent=2)
    settings = {"noise_t": parse_list(args.noise_t), "control_t": parse_list(args.control_t),
                "own_sources": [s.strip() for s in args.own_sources.split(",") if s.strip()]}
    settings["all_t"] = sorted(set(settings["noise_t"]) | set(settings["control_t"]))
    meta = {"args": vars(args), "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(device),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "started": datetime.datetime.now().astimezone().isoformat(),
            "paths": {k: str(v) for k, v in paths.items()}, "blocks": blocks, "settings": settings,
            "config_sha256": diag.sha256_file(args.config),
            "source_sha256": {Path(m.__file__).name: diag.sha256_file(m.__file__) for m in
                              (cf_pipeline, cf_v4, cf_v5, dmd_base, diag)} |
                             {Path(__file__).name: diag.sha256_file(__file__)},
            "notes": [
                "single GPU, no FSDP, fp16 parameters + fp16 root-input cast (diag Fp16RootCast)",
                "fake target = forced last exit (full 4-step sample); training draws a uniform exit",
                "one eps / history-noise draw per clip shared by both arms, all sources, histories and t",
                "F and T run in eval mode (training runs F in train mode; no dropout in the DiT)",
                "val clips use sequence_length 51 (config val set is 35 frames), as the diag"]}
    print(f"[probe] building pipeline on {device} ...", flush=True)
    pipe = diag.build_pipeline(diag_cfg, device, out_dir)
    pipe.disc_head.load_state_dict(torch.load(paths["disc_head"], map_location="cpu"))   # strict
    pipe.disc_head.eval()
    taps = {"F": FeatureTap(pipe.fake_score, blocks), "T": FeatureTap(pipe.real_score, blocks)}
    meta["build_seconds"] = round(time.time() - t_begin, 1)
    meta["mem_after_build_GB"] = round(torch.cuda.memory_allocated(device) / 2**30, 2)
    meta["denoising_step_list"] = diag.json_safe(pipe.denoising_step_list)
    meta["denoising_sigmas"] = diag.json_safe(pipe.denoising_sigmas)
    meta["fake_score_causal"] = bool(pipe.fake_score_causal)
    meta["real_score_causal"] = bool(pipe.real_score_causal)
    meta["n_forwards_per_clip_planned"] = n_forwards(args)
    print(f"[probe] pipeline ready in {meta['build_seconds']} s, {meta['mem_after_build_GB']} GB", flush=True)
    diag.write_json(out_dir / "meta.json", meta)

    collate = dwm.common.create_instance_from_config(diag_cfg["training_dataloader"]["collate_fn"])
    datasets, picks = {}, {}
    for split, n_req in (("train", args.num_clips), ("val", args.val_clips)):
        if n_req <= 0:
            continue
        try:
            ds = diag.build_dataset(diag_cfg, split, diag_cfg["pipeline"]["context_windows"])
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] split {split} unavailable: {exc!r}", flush=True)
            meta.setdefault("errors", []).append(f"dataset {split}: {exc!r}")
            continue
        datasets[split] = ds
        if split == "train":
            picks[split] = diag.recent_trained_indices(len(ds), args.step, args.train_world_size, n_req,
                                                       diag_cfg.get("generator_seed", 0))
        else:
            picks[split] = diag.random_indices(len(ds), n_req, args.seed + 1)
        print(f"[probe] split {split}: dataset {len(ds)} items, {len(picks[split])} clips", flush=True)
        meta[f"{split}_dataset_len"] = len(ds)
    # interleave 3 train : 1 val so a truncated run still has both splits
    order, tr, va = [], list(enumerate(picks.get("train", []))), list(enumerate(picks.get("val", [])))
    while tr or va:
        for _ in range(3):
            if tr:
                order.append(("train",) + tr.pop(0))
        if va:
            order.append(("val",) + va.pop(0))
    meta["clip_order"] = [(s, ci, int(p["index"])) for s, ci, p in order]
    diag.write_json(out_dir / "meta.json", meta)

    done = {p.stem for p in clip_dir.glob("*.pt")}
    jsonl = open(out_dir / "progress.jsonl", "a")
    n_done, truncated = 0, False
    for split, ci, pick in order:
        if args.max_clips >= 0 and n_done >= args.max_clips:
            break
        if (time.time() - t_begin) / 60.0 > args.time_budget_min:
            truncated = True
            print("[probe] time budget reached; stopping", flush=True)
            break
        name = f"{split}_{ci:03d}"
        if name in done:
            continue
        item = datasets[split][int(pick["index"])]
        batch = collate([item])
        clip_seed = args.seed * 1000003 + (0 if split == "train" else 500000) + ci    # == diag
        try:
            rec, payload = process_clip(pipe, batch, args, clip_seed, taps, blocks, settings)
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] {name} FAILED: {exc!r}", flush=True)
            meta.setdefault("errors", []).append(f"{name}: {exc!r}")
            torch.cuda.empty_cache()
            continue
        rec.update({"split": split, "clip_i": ci, "seed": clip_seed, **pick,
                    **{k: diag.json_safe(batch[k]) for k in ("scene_token", "anchor_timestamp_us",
                                                             "dataset_index") if k in batch}})
        payload["meta"] = rec
        tmp = clip_dir / f".{name}.pt.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, clip_dir / f"{name}.pt")
        jsonl.write(json.dumps(rec) + "\n")
        jsonl.flush()
        n_done += 1
        if n_done == 1:
            meta["peak_mem_GB_first_clip"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
        hl = rec["old_head_logit"]
        show = {t: round(hl.get(f"F|noise|real|t{t}", float("nan")) - hl.get(f"F|noise|fake|t{t}", float("nan")), 2)
                for t in settings["noise_t"]}
        print(f"[probe] {n_done}/{len(order)} {name} idx={pick['index']} {rec['timing']} fwd={rec['n_forwards']}"
              f" mseG4={rec['latent_check']['target_mse_G4_vs_GT']:.5f}"
              f" oldhead(real-fake,noise-hist)={json.dumps(show)}", flush=True)
        meta["n_clips_done"] = n_done
        meta["elapsed_min"] = round((time.time() - t_begin) / 60.0, 1)
        diag.write_json(out_dir / "meta.json", meta)
        del payload, batch, item
        torch.cuda.empty_cache()
    jsonl.close()
    meta["truncated"] = truncated
    meta["finished"] = datetime.datetime.now().astimezone().isoformat()
    meta["wall_minutes"] = round((time.time() - t_begin) / 60.0, 1)
    meta["peak_mem_GB"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    diag.write_json(out_dir / "meta.json", meta)
    print(f"[probe] extraction done: {n_done} clips in {meta['wall_minutes']} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
