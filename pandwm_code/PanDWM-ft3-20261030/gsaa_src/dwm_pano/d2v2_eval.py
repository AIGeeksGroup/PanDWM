"""Lateral-shift FID / FVD / sharpness of the D2 v2 refiner (2026-10-31, PanDWM-d2-refiner-20261021/D2V2_PLAN_20261031.md).

eval_gs_shift.py's protocol (32 generated frames 3..34; FID-torchmetrics on the stitched pano, FVD-I3D-sector-6-16
on frames 0..15; +d and -d pooled with the real frames added once per side; sharpness = luminance gradient energy of
the output over the real frame's, eval_gs_shift.grad_energy), on the d2v2_export.py packs and the refined latents
of d2_refine.py (<refined>/clipNNN/d2_dy<d>.pt), per shift d:

  raw_<d>   the export's pixels (eval_gs_shift's 4D output; must reproduce the GS-Dev FT3_G3b numbers)
  d2f_<d>   the refined latents decoded with the SD3.5 VAE and stitched like the video model's outputs (full)
  d2m_<d>   d2f inside the export's fixmask (holes + magnified content), dilated by --dilate and feathered over
            --feather pano pixels, raw elsewhere (covered, unmagnified pixels keep the render exactly)

States after every clip (--state), merged with --merge, as in eval_gs_shift.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import einops
import torch
import torch.nn.functional as F

import dwm.common
from dwm_pano.eval_gs_shift import build_metrics, grad_energy, load_into, save_state
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.export_gs_batch149 import FRAMES
from dwm_pano.pano_stitch import stitch_pano_halves

CONTEXT = 3
METRICS = ("fid", "fvd_w1")
KINDS = ("raw", "d2f", "d2m")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path, help="GS review config (VAE, dataset)")
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--inputs", required=True, type=Path, help="d2v2_export.py out-root")
    p.add_argument("--refined", default=None, type=Path, help="d2_refine.py out-root (None: raw only)")
    p.add_argument("--clips", default="")
    p.add_argument("--shifts", default="0,1,2,4", help="unsigned; d > 0 pools +d and -d")
    p.add_argument("--fps-stride", default="10,20")
    p.add_argument("--dilate", type=int, default=6)
    p.add_argument("--feather", type=int, default=6)
    p.add_argument("--i3d", required=True)
    p.add_argument("--state", required=True, type=Path)
    p.add_argument("--merge", nargs="*", default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--expected-clips", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def soft_mask(mask, dilate, feather):
    """(T, H, W) {0, 1} -> (T, H, W) weights: dilated by `dilate` px, box-feathered over `feather` px."""
    m = mask.float()[:, None]
    if dilate > 0:
        m = F.max_pool2d(m, 2 * dilate + 1, stride=1, padding=dilate)
    if feather > 0:
        m = F.avg_pool2d(m, 2 * feather + 1, stride=1, padding=feather, count_include_pad=False)
    return m[:, 0].clamp(0, 1)


@torch.no_grad()
def decode_latents(pipe, latents):
    """(T, 2, 16, h, w) latents -> (T, 3, H, 4114) stitched pano in [0, 1] (as d2_refine.decode)."""
    vae = pipe.vae
    shift = vae.config.shift_factor or 0
    flat = latents.flatten(0, 1).to(pipe.device, dtype=vae.dtype)
    images = torch.cat([vae.decode(part / vae.config.scaling_factor + shift, return_dict=False)[0]
                        for part in flat.split(8)])                     # the GS pipeline has no common_config
    images = ((images.float() + 1.0) / 2.0).clamp(0, 1).unflatten(0, (latents.shape[0], latents.shape[1]))
    return stitch_pano_halves(images[None])[0, :, 0].clamp(0, 1)


def report(args, arms, metrics, extra, done):
    out = {"protocol": "eval_gs_shift lateral-shift protocol on d2v2_export packs; d2 arms decoded with the SD3.5 VAE",
           "metric_names": {"fid": "FID-torchmetrics-stitched-pano, 32 generated frames",
                            "fvd_w1": "FVD-I3D-sector-6-16, generated frames 0-15 (window 1)"},
           "clips": len(done), "inputs": str(args.inputs), "refined": str(args.refined), "arms": {},
           "dilate": args.dilate, "feather": args.feather}
    for arm in arms:
        out["arms"][arm] = {n: float(metrics[arm][n].compute()) for n in METRICS}
        sh = [v for v in extra["sharp"].get(arm, []) if v == v]
        out["arms"][arm]["sharpness"] = sum(sh) / len(sh) if sh else None
        cov = extra["coverage"].get(arm, [])
        out["arms"][arm]["coverage"] = sum(cov) / len(cov) if cov else None
    print(json.dumps(out, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" else "cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    shifts = [float(s) for s in args.shifts.split(",") if s.strip()]
    kinds = KINDS if args.refined is not None else ("raw",)
    arms = [f"{k}_{d:g}" for k in kinds for d in shifts]
    metrics = build_metrics(arms, args.i3d, device)
    if args.merge is not None:
        done, extra = [], {"sharp": {a: [] for a in arms}, "coverage": {a: [] for a in arms}}
        for path in args.merge:
            part = torch.load(path, map_location="cpu", weights_only=False)
            if part["arms"] != arms:
                raise ValueError(f"{path}: different arm set")
            load_into(metrics, part, device)
            done += part["done"]
            for key in ("sharp", "coverage"):
                for a in arms:
                    extra[key][a] += part["extra"][key].get(a, [])
        for arm in arms:
            for m in metrics[arm].values():
                if not hasattr(m, "orig_dtype"):
                    m.orig_dtype = torch.float32
        report(args, arms, metrics, extra, done)
        return
    done, extra = [], {"sharp": {a: [] for a in arms}, "coverage": {a: [] for a in arms}}
    if args.state.is_file():
        state = torch.load(args.state, map_location="cpu", weights_only=False)
        if state["arms"] != arms:
            raise ValueError("state belongs to a different arm set")
        load_into(metrics, state, device)
        done, extra = state["done"], state["extra"]
    started = ensure_single_rank_process_group(device)
    try:
        config = load_config(args.config)
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(args.state.parent),
                                                      config=config, device=device, resume_from=None)
        pipe.vae.eval()
        install_strict_pano_hardseam_stub()
        fps, stride = (float(x) for x in args.fps_stride.split(","))
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl,
                                   fps_stride_tuples=[[int(fps), stride]])
        cfg["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        by_key = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in
                  enumerate(zip(base.clips, base.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring
        clips = [int(x) for x in args.clips.split(",") if x.strip()]
        for index in clips:
            if index in done:
                continue
            t0 = time.monotonic()
            packs = {}
            for d in shifts:
                for sign in ((1.0, -1.0) if d > 0 else (1.0,)):
                    dy = sign * d
                    packs[dy] = torch.load(args.inputs / f"clip{index:03d}" / f"render_dy{dy:g}.pt",
                                           map_location="cpu", weights_only=True)
            meta = next(iter(packs.values()))["meta"]
            ds_index = by_key[(meta["scene_token"], int(meta["anchor_timestamp_us"]))]
            batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [ds_index]),
                                                          batch_size=1, shuffle=False, num_workers=0,
                                                          collate_fn=CollateFnIgnoring(keys=[]))))
            if [int(x) for x in batch["frame_timestamps_us"][0].tolist()] != meta["frame_timestamps_us"]:
                raise RuntimeError(f"clip {index}: export frames differ from the dataset clip")
            real = stitch_pano_halves(batch["vae_images"][:, CONTEXT:FRAMES].to(device)).clamp(0, 1)
            pipe._prepare_nuscenes6_supervision(batch)                     # the clip rig's stitch_valid, as eval_gs_shift
            valid = pipe.stitch_valid.to(device)
            for d in shifts:
                fakes = {k: [] for k in kinds}
                for sign in ((1.0, -1.0) if d > 0 else (1.0,)):
                    dy = sign * d
                    pack = packs[dy]
                    raw = pack["pixels"][CONTEXT:FRAMES].to(device).float() / 255.0              # (32, 3, H, W)
                    outs = {"raw": raw}
                    if args.refined is not None:
                        ref = torch.load(args.refined / f"clip{index:03d}" / f"d2_dy{dy:g}.pt", map_location="cpu",
                                         weights_only=True)
                        if ref["meta"]["frame_timestamps_us"] != meta["frame_timestamps_us"]:
                            raise RuntimeError(f"clip {index} dy {dy:g}: refined frames differ")
                        full = decode_latents(pipe, ref["latents"][CONTEXT:FRAMES].float())
                        if full.shape[-2:] != raw.shape[-2:]:
                            full = F.interpolate(full, size=raw.shape[-2:], mode="bilinear", align_corners=False)
                        w = soft_mask(pack["fixmask"][CONTEXT:FRAMES].to(device), args.dilate, args.feather)[:, None]
                        outs["d2f"] = full
                        outs["d2m"] = w * full + (1.0 - w) * raw
                    for k in kinds:
                        fakes[k].append(outs[k])
                        arm = f"{k}_{d:g}"
                        for t in range(raw.shape[0]):
                            e = grad_energy(real[0, t, 0], valid)
                            extra["sharp"][arm].append(grad_energy(outs[k][t], valid) / max(e, 1e-8))
                        extra["coverage"][arm].append(float(pack["meta"]["coverage_mean"]))
                for k in kinds:
                    arm = f"{k}_{d:g}"
                    for frames in fakes[k]:
                        fake = frames[None, :, None]                                     # (1, 32, 1, 3, H, W)
                        metrics[arm]["fid"].update(real.flatten(0, 2), real=True)
                        metrics[arm]["fid"].update(fake.flatten(0, 2), real=False)
                        metrics[arm]["fvd_w1"].update(einops.rearrange(real[:, :16], "b t v c h w -> (b v) t c h w"),
                                                      real=True)
                        metrics[arm]["fvd_w1"].update(einops.rearrange(fake[:, :16], "b t v c h w -> (b v) t c h w"),
                                                      real=False)
            done.append(index)
            save_state(args.state, arms, done, metrics, {}, extra)
            print(f"[d2v2-eval] clip {index} done ({time.monotonic() - t0:.0f}s)", flush=True)
        if args.expected_clips and len(done) == args.expected_clips and args.output is not None:
            report(args, arms, metrics, extra, done)
        else:
            print(f"SHARD_DONE {len(done)} clips", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
