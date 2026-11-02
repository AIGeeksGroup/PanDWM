"""Lateral-shift FID / FVD of the GS renders of the 149 review clips (2026-10-19, report section 40, item A).

The FreeVS / WorldSplat / DiST-4D novel-trajectory protocol on our panorama: every frame's Gaussians (the v2 review
scene, or a v3 variant of gs_scene_v3.SceneOptions) are rendered at the six real nuScenes cameras with the ego
shifted sideways by dy and stitched with the pipeline's own hard seam into the 4114 x 256 V=1 pano the video
metrics use.  Holes (alpha < 0.5) are filled with export_gs_v2_clips.pushpull_fill, as in the v2 review videos.
Unpaired metrics against the real frames of the original trajectory, with the frozen video instruments of
compute_eval149_metrics.py: FID-torchmetrics-stitched-pano over the 32 generated frames and FVD-I3D-sector-6-16 over
generated frames 0-15 (window 1).  Arms "<mode>_<d>": mode 3d = frame t only, 4d = + nearby-frame static fill;
d > 0 pools the +d and -d trajectories (both renders count as fakes, the real frames are added once per side).
Also per arm: coverage = mean fraction of stitch-valid pixels with alpha > 0.5 before the fill.

Metric states are saved after every clip (--state) so a run can be split over array tasks and merged (--merge).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import einops
import torch

import dwm.common
from dwm.utils.preview import save_tensor_to_video
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.export_gs_batch149 import FRAMES
from dwm_pano.export_gs_v2_clips import pushpull_fill
from dwm_pano.gs_scene_v3 import SceneBuilder, SceneOptions, composite
from dwm_pano.pano_stitch import stitch_pano_halves

CONTEXT = 3
METRICS = ("fid", "fvd_w1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--clips-root", required=True, type=Path)
    p.add_argument("--i3d", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=149)
    p.add_argument("--shifts", default="0,1,2,4", help="pooled |dy| values in metres")
    p.add_argument("--modes", default="3d,4d")
    p.add_argument("--state", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--merge", nargs="*", default=None)
    p.add_argument("--device", default="cuda", help="cpu is enough for --merge (states only)")
    p.add_argument("--time-budget-s", type=float, default=0.0)
    p.add_argument("--sample-clips", default="", help="clip indices whose renders are saved as videos")
    p.add_argument("--sample-dir", type=Path, default=None)
    # scene options (defaults = the v2 review scene)
    p.add_argument("--edge-rel-h", type=float, default=0.1)
    p.add_argument("--edge-rel-v", type=float, default=0.3)
    p.add_argument("--window", default="-20,-15,-10,-5,5,10")
    p.add_argument("--ownership", action="store_true")
    p.add_argument("--own-include-current", action="store_true")
    p.add_argument("--da3-relabel", action="store_true")
    p.add_argument("--sky-layer", action="store_true")
    p.add_argument("--da3-src", default="")
    p.add_argument("--da3-anyview", default="")
    p.add_argument("--da3-metric", default="")
    return p.parse_args()


def build_metrics(arms, i3d, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from dwm_pano.metrics.fvd_pano import PanoPatchFrechetVideoDistance
    return {arm: {"fid": FrechetInceptionDistance(normalize=True).to(device),
                  "fvd_w1": PanoPatchFrechetVideoDistance(inception_3d_checkpoint_path=i3d,
                                                          sequence_count=16).to(device)} for arm in arms}


def save_state(path, arms, done, metrics, coverage, extra):
    tmp = path.with_suffix(".tmp")
    torch.save({"arms": arms, "done": done, "coverage": coverage, "extra": extra,
                "metrics": {a: {n: {k: v.detach().cpu() for k, v in m.metric_state.items()} for n, m in ms.items()}
                            for a, ms in metrics.items()}}, tmp)
    tmp.replace(path)


def load_into(metrics, state, device):
    for arm, ms in metrics.items():
        for name, m in ms.items():
            for key, value in state["metrics"][arm][name].items():
                current = getattr(m, key)
                setattr(m, key, current + value.to(device) if isinstance(current, torch.Tensor) else value)


def report(args, arms, metrics, coverage, extra, done):
    out = {"protocol": "lateral-shift novel trajectory (FreeVS / WorldSplat / DiST-4D), 149 review clips, "
                       "real reference = original trajectory, unpaired",
           "metric_names": {"fid": "FID-torchmetrics-stitched-pano, 32 generated frames",
                            "fvd_w1": "FVD-I3D-sector-6-16, generated frames 0-15 (window 1)"},
           "clips": len(done), "checkpoint": str(args.checkpoint), "arms": {}}
    for arm in arms:
        cov = coverage.get(arm, [])
        out["arms"][arm] = {n: float(metrics[arm][n].compute()) for n in METRICS}
        out["arms"][arm]["coverage"] = sum(cov) / len(cov) if cov else None
    out["scene"] = extra.get("scene")
    out["scene_stats"] = extra.get("scene_stats")
    print(json.dumps(out, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")


@torch.no_grad()
def main():
    args = parse_args()
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
    shifts = [float(s) for s in args.shifts.split(",") if s.strip()]
    modes = [m for m in args.modes.split(",") if m]
    arms = [f"{m}_{s:g}" for m in modes for s in shifts]
    metrics = build_metrics(arms, args.i3d, device)
    opts = SceneOptions(edge_rel_h=args.edge_rel_h, edge_rel_v=args.edge_rel_v,
                        window=tuple(int(x) for x in args.window.split(",") if x.strip()),
                        ownership=args.ownership, own_include_current=args.own_include_current,
                        da3_relabel=args.da3_relabel, sky_layer=args.sky_layer)
    scene_desc = {k: getattr(opts, k) for k in ("edge_rel_h", "edge_rel_v", "window", "ownership",
                                                 "own_include_current", "da3_relabel", "sky_layer")}
    if args.merge is not None:
        coverage, done, scene_stats = {a: [] for a in arms}, [], []
        for path in args.merge:
            part = torch.load(path, map_location="cpu", weights_only=False)
            if part["arms"] != arms:
                raise ValueError(f"{path}: different arm set")
            load_into(metrics, part, device)
            done += part["done"]
            for a in arms:
                coverage[a] += part["coverage"].get(a, [])
            scene_stats += part["extra"].get("scene_stats", [])
        for arm in arms:
            for m in metrics[arm].values():
                if not hasattr(m, "orig_dtype"):
                    m.orig_dtype = torch.float32
        merged = {}
        for row in scene_stats:
            for k, v in row.items():
                if v is not None:
                    merged.setdefault(k, []).append(v)
        report(args, arms, metrics, coverage, {"scene": part["extra"].get("scene"),
                                              "scene_stats": {k: sum(v) / len(v) for k, v in merged.items()}},
               done)
        return

    done, coverage, scene_stats = [], {a: [] for a in arms}, []
    if args.state.is_file():
        state = torch.load(args.state, map_location="cpu", weights_only=False)
        if state["arms"] != arms:
            raise ValueError("state belongs to a different arm set")
        load_into(metrics, state, device)
        done, coverage = state["done"], state["coverage"]
        scene_stats = state["extra"].get("scene_stats", [])
        print(f"[gs-shift] resumed with {len(done)} clips", flush=True)
    da3 = None
    if args.da3_relabel or args.sky_layer:
        from dwm_pano_4d.da3_depth import Da3Ranger
        da3 = Da3Ranger(args.da3_src, args.da3_anyview, args.da3_metric, device)
    samples = {int(x) for x in args.sample_clips.split(",") if x.strip()}
    started = ensure_single_rank_process_group(device)
    try:
        config = load_config(args.config)
        torch.manual_seed(int(config["generator_seed"]))
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(args.state.parent),
                                                      config=config, device=device, resume_from=None)
        pipe.vae.eval()
        pipe.decoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
        pipe.decoder.eval()
        pipe.decoder_wrapper.eval()
        pipe.rasterize_mode = "antialiased"
        pipe.active_rasterize_eps2d = 0.1
        install_strict_pano_hardseam_stub()
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl, fps_stride_tuples=[[10, 20]])
        cfg["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        by_key = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in
                  enumerate(zip(base.clips, base.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring
        job_start, last = time.monotonic(), 0.0
        for index in range(args.start, args.end):
            if index in done:
                continue
            src = args.clips_root / f"clip{index:03d}"
            if args.time_budget_s and last and time.monotonic() - job_start + 1.3 * last > args.time_budget_s:
                print(f"[gs-shift] budget reached before clip {index}", flush=True)
                break
            t0 = time.monotonic()
            summary = json.loads((src / "summary.json").read_text())
            ds_index = by_key[(summary["scene_token"], int(summary["anchor_timestamp_us"]))]
            batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [ds_index]),
                                                          batch_size=1, shuffle=False, num_workers=0,
                                                          collate_fn=CollateFnIgnoring(keys=[]))))
            latents = torch.load(src / "latents.pt", map_location="cpu", weights_only=True)["latents"]
            if latents.shape[0] != FRAMES:
                raise ValueError(f"clip {index}: {latents.shape[0]} latent frames")
            scene = SceneBuilder(pipe, batch, base.clips[ds_index], latents, opts, device, da3=da3)
            cam2ego = pipe.nuscenes_camera_to_ego.float()
            valid = pipe.stitch_valid
            real = stitch_pano_halves(batch["vae_images"][:, CONTEXT:FRAMES].to(device)).clamp(0, 1)
            fakes = {arm: {sign: [] for sign in ((1.0, -1.0) if float(arm.split("_")[1]) > 0 else (1.0,))}
                     for arm in arms}
            sample_rows = []
            for t in range(CONTEXT, FRAMES):
                current = scene.decode(t)
                fill = scene.fill(t) if "4d" in modes else None
                sky_canvas = None
                if opts.sky_layer:
                    sky = (scene.da3_maps(t)["sky"].float() >= 0.3).float()
                    sky_canvas = pipe._unpadded_stitch_canvas(sky[None, :, None])[0, 0]
                row = []
                for arm in arms:
                    mode, dist = arm.split("_")
                    for sign in fakes[arm]:
                        shifted = cam2ego.clone()
                        shifted[:, 1, 3] += sign * float(dist)
                        rgb, _, alpha, _ = pipe._render_nuscenes6_stitched(current, target_c2w=shifted)
                        if mode == "4d" and fill is not None and fill.shape[0]:
                            rgb_f, _, alpha_f, _ = pipe._render_nuscenes6_stitched(fill, target_c2w=shifted)
                            rgb, alpha = composite(rgb, alpha, rgb_f, alpha_f)
                        if sky_canvas is not None:
                            rgb, alpha = composite(rgb, alpha, scene.canvas(t)[None] * sky_canvas[None, None],
                                                   sky_canvas[None])
                        alpha = torch.where(valid[None], alpha, torch.zeros_like(alpha))
                        coverage[arm].append(float((alpha[0][valid] > 0.5).float().mean()))
                        out = pushpull_fill(rgb[0].clamp(0, 1), alpha[0]).clamp(0, 1)
                        fakes[arm][sign].append(out)
                        if index in samples and ((mode == "4d" and float(dist) in (0.0, 2.0)) or
                                                 (mode == "3d" and float(dist) == 2.0 and sign > 0)):
                            row.append(out)
                if index in samples:
                    sample_rows.append(torch.cat([real[0, t - CONTEXT, 0]] + row, dim=-2).cpu())
            for arm in arms:
                for sign, frames in fakes[arm].items():
                    fake = torch.stack(frames)[None, :, None]                       # (1, 32, 1, 3, H, W)
                    metrics[arm]["fid"].update(real.flatten(0, 2), real=True)
                    metrics[arm]["fid"].update(fake.flatten(0, 2), real=False)
                    metrics[arm]["fvd_w1"].update(einops.rearrange(real[:, :16], "b t v c h w -> (b v) t c h w"),
                                                  real=True)
                    metrics[arm]["fvd_w1"].update(einops.rearrange(fake[:, :16], "b t v c h w -> (b v) t c h w"),
                                                  real=False)
            if index in samples and args.sample_dir is not None:
                args.sample_dir.mkdir(parents=True, exist_ok=True)
                frames = [torch.nn.functional.interpolate(f[None], size=(f.shape[-2] // 4 * 2, f.shape[-1] // 4 * 2),
                                                          mode="area")[0] for f in sample_rows]   # even for x264
                save_tensor_to_video(str(args.sample_dir / f"clip{index:03d}_shift.mp4"), "libx264",
                                     float(batch["fps"][0]), frames)
            done.append(index)
            scene_stats.append(scene.summary())
            save_state(args.state, arms, done, metrics, coverage, {"scene": scene_desc, "scene_stats": scene_stats})
            last = time.monotonic() - t0
            print(f"[gs-shift] clip {index} done ({last:.0f}s) {json.dumps(scene.summary())}", flush=True)
        if len(done) == 149 and args.output is not None:
            report(args, arms, metrics, coverage, {"scene": scene_desc, "scene_stats": None}, done)
        else:
            print(f"SHARD_DONE {len(done)} clips", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
