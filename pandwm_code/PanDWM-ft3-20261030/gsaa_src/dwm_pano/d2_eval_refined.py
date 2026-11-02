"""Lateral-shift FID / FVD of the D2 refiner outputs (2026-10-21, PanDWM-d2-refiner-20261021/D2_PLAN_20261021.md).

The eval_gs_shift.py protocol (FreeVS / WorldSplat / DiST-4D novel trajectory, unpaired against the real frames of
the original trajectory; FID-torchmetrics-stitched-pano over the 32 generated frames, FVD-I3D-sector-6-16 over
generated frames 0-15; +d and -d pooled, the real frames added once per side), but every arm is a set of LATENT
files decoded with the SD3.5 VAE and stitched with stitch_pano_halves (PANDWM_PANO_BLEND_W, 128 in the sbatch)
like the video model's own outputs:

    --arm in=<review_inputs dir>:render     <dir>/clipNNN/render_dy<d>.pt["render"]   (GS render, VAE round trip)
    --arm d2=<refined dir>:latents          <dir>/clipNNN/d2_dy<d>.pt["latents"]      (refiner output)
    --arm Bf=<refined dir>:latents:fuse     the refiner output with its low frequencies (16 x 16 mean, bilinear
                                            back up) replaced by those of the "in" frames (section 43.3: the
                                            refiner's detail with the GS render's colours); needs an "in" arm

so the refined arm and its own input are measured through the same decode and stitch.  If an arm is named
"in", every other arm also gets coarse_psnr (section 43.2): PSNR against the "in" frames after a 16 x 16
average pool (colour / layout agreement with the GS render, blind to the fine artefacts a refiner should fix).  Arm names are
"<name>_<d>".  States are saved after every clip (--state) and merged with --merge, as in eval_gs_shift.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import einops
import torch

import dwm.common
from dwm_pano.eval_gs_shift import build_metrics, load_into, save_state
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.export_gs_batch149 import FRAMES
from dwm_pano.pano_stitch import stitch_pano_halves

CONTEXT = 3
METRICS = ("fid", "fvd_w1")
PREFIX = {"render": "render", "latents": "d2"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--clips-root", required=True, type=Path)
    p.add_argument("--i3d", required=True)
    p.add_argument("--arm", action="append", default=[], help="name=<dir>:<render|latents>")
    p.add_argument("--shifts", default="2", help="pooled |dy| values in metres")
    p.add_argument("--clips", default="", help="comma list or start:end")
    p.add_argument("--state", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--merge", nargs="*", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def parse_clips(text):
    if ":" in text:
        a, b = (int(x) for x in text.split(":"))
        return list(range(a, b))
    return [int(x) for x in text.split(",") if x.strip()]


def parse_arms(specs):
    arms, fused = {}, set()
    for spec in specs:
        name, rest = spec.split("=", 1)
        if rest.endswith(":fuse"):
            rest = rest[:-len(":fuse")]
            fused.add(name)
        root, key = rest.rsplit(":", 1)
        if key not in PREFIX:
            raise ValueError(f"arm {name}: key must be one of {sorted(PREFIX)}")
        arms[name] = (Path(root), key)
    if not arms:
        raise ValueError("no --arm given")
    if fused and "in" not in arms:
        raise ValueError("fused arms need an 'in' arm")
    return arms, fused


def lowpass(x, factor=16):
    """(..., 3, H, W) -> the same shape, 16 x 16 mean bilinearly upsampled back."""
    flat = x.flatten(0, -4)
    low = torch.nn.functional.avg_pool2d(flat, factor, ceil_mode=True)
    low = torch.nn.functional.interpolate(low, size=flat.shape[-2:], mode="bilinear", align_corners=False)
    return low.view_as(x)


def coarse_psnr(a, b):
    """(..., 3, H, W) in [0, 1] -> PSNR after a 16 x 16 average pool."""
    pa = torch.nn.functional.avg_pool2d(a.flatten(0, -4), 16)
    pb = torch.nn.functional.avg_pool2d(b.flatten(0, -4), 16)
    return float(-10 * torch.log10((pa - pb).square().mean().clamp_min(1e-12)))


def report(args, arm_names, metrics, done, sources, coarse=None):
    out = {"protocol": "lateral-shift novel trajectory, 149 review clips, real reference = original trajectory, "
                       "unpaired; latent arms decoded with the SD3.5 VAE and stitch_pano_halves",
           "metric_names": {"fid": "FID-torchmetrics-stitched-pano, 32 generated frames",
                            "fvd_w1": "FVD-I3D-sector-6-16, generated frames 0-15 (window 1)"},
           "clips": len(done), "sources": sources, "arms": {}}
    for arm in arm_names:
        out["arms"][arm] = {n: float(metrics[arm][n].compute()) for n in METRICS}
        values = (coarse or {}).get(arm) or []
        if values:
            out["arms"][arm]["coarse_psnr"] = sum(values) / len(values)
    print(json.dumps(out, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")


@torch.no_grad()
def decode_frames(pipe, latents):
    """(T, 2, C, h, w) latents -> (1, T, 1, 3, H, 4114) stitched panos in [0, 1]."""
    halves = torch.cat([pipe.decode_half_latents(latents[i:i + 4].to(pipe.device)) for i in
                        range(0, latents.shape[0], 4)]).float()                  # (T, 2, 3, H, W)
    return stitch_pano_halves(halves[None]).clamp(0, 1)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cpu") if args.device == "cpu" else torch.device("cuda", 0)
    if device.type == "cuda":
        torch.cuda.set_device(0)
    arms, fused = parse_arms(args.arm)
    shifts = [float(s) for s in args.shifts.split(",") if s.strip()]
    arm_names = [f"{name}_{d:g}" for name in arms for d in shifts]
    sources = {name: [str(root), key] + (["fuse"] if name in fused else []) for name, (root, key) in arms.items()}
    metrics = build_metrics(arm_names, args.i3d, device)
    if args.merge is not None:
        done, coarse = [], {}
        for path in args.merge:
            part = torch.load(path, map_location="cpu", weights_only=False)
            if part["arms"] != arm_names:
                raise ValueError(f"{path}: different arm set")
            if part["extra"].get("sources") != sources:
                raise ValueError(f"{path}: made from other arm sources {part['extra'].get('sources')}")
            load_into(metrics, part, device)
            done += part["done"]
            for arm, values in part["extra"].get("coarse", {}).items():
                coarse.setdefault(arm, []).extend(values)
        for arm in arm_names:
            for m in metrics[arm].values():
                if not hasattr(m, "orig_dtype"):
                    m.orig_dtype = torch.float32
        if len(set(done)) != len(done):
            raise ValueError("a clip appears in two merged states")
        report(args, arm_names, metrics, sorted(done), sources, coarse)
        return
    done, coarse = [], {}
    if args.state.is_file():
        state = torch.load(args.state, map_location="cpu", weights_only=False)
        if state["arms"] != arm_names or state["extra"].get("sources") != sources:
            raise ValueError("state belongs to a different arm set")
        load_into(metrics, state, device)
        done = state["done"]
        coarse = state["extra"].get("coarse", {})
        for arm in arm_names:                    # compute() needs it even when no clip is added after resuming
            for m in metrics[arm].values():
                if not hasattr(m, "orig_dtype"):
                    m.orig_dtype = torch.float32
        print(f"[d2-eval] resumed with {len(done)} clips", flush=True)
    started = ensure_single_rank_process_group(device)
    try:
        config = load_config(args.config)
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(args.state.parent),
                                                      config=config, device=device, resume_from=None)
        pipe.vae.eval()
        install_strict_pano_hardseam_stub()
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl, fps_stride_tuples=[[10, 20]])
        cfg["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        by_key = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in
                  enumerate(zip(base.clips, base.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring
        for index in parse_clips(args.clips):
            if index in done:
                continue
            t0 = time.monotonic()
            summary = json.loads((args.clips_root / f"clip{index:03d}" / "summary.json").read_text())
            ds_index = by_key[(summary["scene_token"], int(summary["anchor_timestamp_us"]))]
            batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [ds_index]),
                                                          batch_size=1, shuffle=False, num_workers=0,
                                                          collate_fn=CollateFnIgnoring(keys=[]))))
            frame_ts = [int(x) for x in batch["frame_timestamps_us"][0].tolist()]
            real = stitch_pano_halves(batch["vae_images"][:, CONTEXT:FRAMES].to(device)).clamp(0, 1)
            order = sorted(arms, key=lambda n: n != "in")
            for d in shifts:
                for sign in ((1.0, -1.0) if d > 0 else (1.0,)):
                    reference = None
                    for name in order:
                        root, key = arms[name]
                        arm = f"{name}_{d:g}"
                        path = root / f"clip{index:03d}" / f"{PREFIX[key]}_dy{sign * d:g}.pt"
                        pack = torch.load(path, map_location="cpu", weights_only=True)
                        if pack["meta"]["frame_timestamps_us"] != frame_ts:
                            raise RuntimeError(f"{path}: frame timestamps differ from the review clip")
                        lat = pack[key][CONTEXT:FRAMES].float()
                        fake = decode_frames(pipe, lat)                                     # (1, 32, 1, 3, H, W)
                        if name in fused:
                            fake = (fake + lowpass(reference) - lowpass(fake)).clamp(0, 1)
                        if name == "in":
                            reference = fake
                        elif reference is not None:
                            coarse.setdefault(arm, []).append(coarse_psnr(fake, reference))
                        metrics[arm]["fid"].update(real.flatten(0, 2), real=True)
                        metrics[arm]["fid"].update(fake.flatten(0, 2), real=False)
                        metrics[arm]["fvd_w1"].update(einops.rearrange(real[:, :16], "b t v c h w -> (b v) t c h w"),
                                                      real=True)
                        metrics[arm]["fvd_w1"].update(einops.rearrange(fake[:, :16], "b t v c h w -> (b v) t c h w"),
                                                      real=False)
            done.append(index)
            save_state(args.state, arm_names, done, metrics, {}, {"sources": sources, "coarse": coarse})
            print(f"[d2-eval] clip {index} done ({time.monotonic() - t0:.0f}s)", flush=True)
        print(f"D2_EVAL_SHARD_DONE {len(done)} clips", flush=True)
        if args.output is not None:
            report(args, arm_names, metrics, sorted(done), sources, coarse)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
