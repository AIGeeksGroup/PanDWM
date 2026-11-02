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

2026-10-23 (Table 5, generated Test-S scenes): --fps-stride 10,5 enumerates the Test-S clips (tests213 manifest),
--expected-clips sets the clip count of the final report, --box-reading / --camera-time-poses select the fixed 4D
scene of section 44, and --depth-pack adds AbsRel / RMSE / delta1 of the 0 m 4D render against the recorded LiDAR
(recon/gen_depth_pack.py): the six real cameras of each pack frame are rendered at --depth-res, the expected depth is
read at every LiDAR point inside the band with alpha >= 0.5 (nearest pixel), per image all / static points.  The
defaults keep the section 40/41 behaviour.
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
from dwm_pano.gs_scene_v3 import SceneBuilder, SceneOptions, background_row_fill, composite, composite_depth
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
    p.add_argument("--da3-mode", choices=["ray", "scale", "global", "blend", "lowfreq"], default="ray",
                   help="scale = C1b (section 40.3); global / blend = round 2b; lowfreq = GS_CLIPDEPTH_PLAN section 7")
    p.add_argument("--lf-filter", choices=["gauss", "guided"], default="gauss")
    p.add_argument("--lf-sigma-px", type=float, default=32.0)
    p.add_argument("--lf-radius-px", type=int, default=32)
    p.add_argument("--lf-eps", type=float, default=0.01)
    p.add_argument("--fill-mode", choices=["pushpull", "bgrow"], default="pushpull", help="bgrow = B3 (section 40.4)")
    p.add_argument("--drop-bottom-rows", type=int, default=0, help="section 41.6: C2's band-edge line")
    p.add_argument("--da3-src", default="")
    p.add_argument("--da3-anyview", default="")
    p.add_argument("--da3-metric", default="")
    p.add_argument("--fps-stride", default="10,20", help="dataset fps,stride (Test-S: 10,5)")
    p.add_argument("--expected-clips", type=int, default=149)
    p.add_argument("--clip-list", default="", help="explicit clip indices (overrides --start/--end)")
    p.add_argument("--box-reading", choices=["legacy", "mmdet3d_v0x"], default="legacy")
    p.add_argument("--camera-time-poses", action="store_true", help="register frames with camera-time ego poses")
    p.add_argument("--depth-pack", type=Path, default=None)
    p.add_argument("--depth-res", default="224x400")
    p.add_argument("--da3-window", choices=["frame", "clip"], default="frame",
                   help="clip = one DA3 pass per clip (GS_CLIPDEPTH_PLAN_20261026.md V1)")
    p.add_argument("--da3-chunk", type=int, default=0, help="clip mode: chunk length in frames, 0 = one pass")
    p.add_argument("--da3-fuse", action="store_true", help="clip mode: static depth fusion (V2)")
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
    out = {"protocol": f"lateral-shift novel trajectory (FreeVS / WorldSplat / DiST-4D), {len(done)} clips, "
                       "real reference = original trajectory, unpaired",
           "metric_names": {"fid": "FID-torchmetrics-stitched-pano, 32 generated frames",
                            "fvd_w1": "FVD-I3D-sector-6-16, generated frames 0-15 (window 1)"},
           "clips": len(done), "checkpoint": str(args.checkpoint), "arms": {}}
    for arm in arms:
        cov = coverage.get(arm, [])
        out["arms"][arm] = {n: float(metrics[arm][n].compute()) for n in METRICS}
        out["arms"][arm]["coverage"] = sum(cov) / len(cov) if cov else None
    rows = extra.get("depth_rows") or []
    if rows:
        out["depth"] = {}
        for tag in ("all", "static"):
            vals = [r[tag] for r in rows if r[tag] is not None]
            out["depth"][tag] = {k: sum(v[k] for v in vals) / len(vals) for k in
                                 ("abs_rel", "rmse", "delta1", "abs_rel_med")} if vals else None
            if vals:
                out["depth"][tag].update(images=len(vals), points=int(sum(v["points"] for v in vals)))
        out["depth"]["protocol"] = "0 m 4D render at the six real cameras vs recorded LiDAR (gen_depth_pack.py)"
    out["scene"] = extra.get("scene")
    out["scene_stats"] = extra.get("scene_stats")
    print(json.dumps(out, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")


_BANDS = {}


def depth_frame(args, pipe, current, fill, info, pack, fi, test_index, t, device):
    """Per-camera depth rows of the 0 m render (current + 4D fill) of frame t against the pack's LiDAR points."""
    import numpy as np
    from dwm_pano.calibrated_stitch import prepare_gsplat_intrinsics
    from dwm_pano.recon.band import band_geometry
    from dwm_pano.recon.score import depth_metrics
    from dwm_pano.recon.targets import CAM_ORDER, pose
    h, w = (int(v) for v in args.depth_res.split("x"))
    sx, sy = w / 1600.0, h / 900.0
    offsets = np.concatenate([[0], np.cumsum(pack["counts"].reshape(-1))])
    s2e = np.stack([pose(info["cams"][n]["sensor2ego_translation"], info["cams"][n]["sensor2ego_rotation"])
                    for n in CAM_ORDER])
    K = pack["K"][fi].astype(np.float64)
    key = (np.round(s2e, 5).tobytes(), np.round(K, 3).tobytes(), h, w)
    if key not in _BANDS:
        _BANDS[key] = band_geometry(s2e, K, [[1600, 900]] * 6, (h, w))["valid"]
    band = _BANDS[key]
    c2w = torch.as_tensor(pack["c2w"][fi], device=device, dtype=torch.float32)
    Kt = torch.stack([prepare_gsplat_intrinsics(torch.as_tensor(k, dtype=torch.float64), (900, 1600), (h, w))
                      for k in K]).float().to(device)
    rgb, depth, alpha = pipe.render_targets(current, target_c2w=c2w, target_intrinsics=Kt, width=w, height=h)
    if fill is not None and fill.shape[0]:
        rgb_f, depth_f, alpha_f = pipe.render_targets(fill, target_c2w=c2w, target_intrinsics=Kt, width=w, height=h)
        depth = composite_depth(depth, alpha, depth_f, alpha_f)
        rgb, alpha = composite(rgb, alpha, rgb_f, alpha_f)
    depth, alpha = depth.float().cpu().numpy(), alpha.float().cpu().numpy()
    rows = []
    for c in range(6):
        q = fi * 6 + c
        sl = slice(int(offsets[q]), int(offsets[q + 1]))
        uv = pack["uv16"][sl].astype(np.float64) / 16.0 - 0.5
        z = pack["z"][sl].astype(np.float64)
        static = pack["static"][sl]
        ui = np.round((uv[:, 0] + 0.5) * sx - 0.5).astype(np.int64)
        vi = np.round((uv[:, 1] + 0.5) * sy - 0.5).astype(np.int64)
        ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        ui, vi, z, static = ui[ok], vi[ok], z[ok], static[ok]
        keep = band[c, vi, ui] & (alpha[c, vi, ui] >= 0.5) & (depth[c, vi, ui] > 1e-3)
        d_hat, z, static = depth[c, vi, ui][keep].astype(np.float64), z[keep], static[keep]
        rows.append({"test_index": int(test_index), "frame": int(t), "camera": CAM_ORDER[c],
                     "all": depth_metrics(d_hat, z), "static": depth_metrics(d_hat[static], z[static])})
    return rows


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
                        da3_relabel=args.da3_relabel, sky_layer=args.sky_layer, da3_mode=args.da3_mode,
                        drop_bottom_rows=args.drop_bottom_rows, box_reading=args.box_reading,
                        da3_window=args.da3_window, da3_chunk=args.da3_chunk, da3_fuse=args.da3_fuse,
                        lf_filter=args.lf_filter, lf_sigma_px=args.lf_sigma_px, lf_radius_px=args.lf_radius_px,
                        lf_eps=args.lf_eps)
    scene_desc = {k: getattr(opts, k) for k in ("edge_rel_h", "edge_rel_v", "window", "ownership",
                                                 "own_include_current", "da3_relabel", "sky_layer", "da3_mode",
                                                 "drop_bottom_rows", "box_reading")}
    if args.da3_window != "frame":                  # 2026-10-26: recorded only when used (old reports unchanged)
        scene_desc.update({k: getattr(opts, k) for k in ("da3_window", "da3_chunk", "da3_fuse", "fuse_offsets",
                                                          "fuse_tol_rel")})
    if args.da3_mode == "lowfreq":
        scene_desc.update({k: getattr(opts, k) for k in ("lf_filter", "lf_sigma_px", "lf_radius_px", "lf_eps")})
    scene_desc["fill_mode"] = args.fill_mode
    scene_desc["camera_time_poses"] = args.camera_time_poses
    if args.merge is not None:
        coverage, done, scene_stats, depth_rows = {a: [] for a in arms}, [], [], []
        for path in args.merge:
            part = torch.load(path, map_location="cpu", weights_only=False)
            if part["arms"] != arms:
                raise ValueError(f"{path}: different arm set")
            load_into(metrics, part, device)
            done += part["done"]
            for a in arms:
                coverage[a] += part["coverage"].get(a, [])
            scene_stats += part["extra"].get("scene_stats", [])
            depth_rows += part["extra"].get("depth_rows", [])
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
                                              "scene_stats": {k: sum(v) / len(v) for k, v in merged.items()},
                                              "depth_rows": depth_rows}, done)
        return

    done, coverage, scene_stats, depth_rows = [], {a: [] for a in arms}, [], []
    if args.state.is_file():
        state = torch.load(args.state, map_location="cpu", weights_only=False)
        if state["arms"] != arms:
            raise ValueError("state belongs to a different arm set")
        load_into(metrics, state, device)
        done, coverage = state["done"], state["coverage"]
        scene_stats = state["extra"].get("scene_stats", [])
        depth_rows = state["extra"].get("depth_rows", [])
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
        fps_stride = [float(x) if "." in x else int(x) for x in args.fps_stride.split(",")]
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl, fps_stride_tuples=[fps_stride])
        cfg["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        by_key = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in
                  enumerate(zip(base.clips, base.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring
        job_start, last = time.monotonic(), 0.0
        indices = [int(x) for x in args.clip_list.split(",") if x.strip()] if args.clip_list else \
            range(args.start, args.end)
        for index in indices:
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
            infos = base.clips[ds_index]
            if args.camera_time_poses:
                from dwm_pano.recon.render_ours import camera_time_infos
                infos = camera_time_infos(infos)
            scene = SceneBuilder(pipe, batch, infos, latents, opts, device, da3=da3)
            cam2ego = pipe.nuscenes_camera_to_ego.float()
            valid = pipe.stitch_valid
            real = stitch_pano_halves(batch["vae_images"][:, CONTEXT:FRAMES].to(device)).clamp(0, 1)
            fakes = {arm: {sign: [] for sign in ((1.0, -1.0) if float(arm.split("_")[1]) > 0 else (1.0,))}
                     for arm in arms}
            sample_rows = []
            pack = None
            if args.depth_pack is not None:
                import numpy as np
                pack = np.load(args.depth_pack / f"clip{int(summary['test_index']):03d}.npz")
                pack_frames = pack["frames"].tolist()
            for t in range(CONTEXT, FRAMES):
                current = scene.decode(t)
                fill = scene.fill(t) if "4d" in modes else None
                if pack is not None and t in pack_frames:
                    depth_rows += depth_frame(args, pipe, current, fill, infos[t], pack, pack_frames.index(t),
                                              summary["test_index"], t, device)
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
                        rgb, depth, alpha, _ = pipe._render_nuscenes6_stitched(current, target_c2w=shifted)
                        if mode == "4d" and fill is not None and fill.shape[0]:
                            rgb_f, depth_f, alpha_f, _ = pipe._render_nuscenes6_stitched(fill, target_c2w=shifted)
                            depth = composite_depth(depth, alpha, depth_f, alpha_f)
                            rgb, alpha = composite(rgb, alpha, rgb_f, alpha_f)
                        if sky_canvas is not None:
                            rgb, alpha = composite(rgb, alpha, scene.canvas(t)[None] * sky_canvas[None, None],
                                                   sky_canvas[None])
                        alpha = torch.where(valid[None], alpha, torch.zeros_like(alpha))
                        coverage[arm].append(float((alpha[0][valid] > 0.5).float().mean()))
                        rgb0, alpha0 = rgb[0].clamp(0, 1), alpha[0]
                        if args.fill_mode == "bgrow":                                    # B3
                            rgb0, alpha0 = background_row_fill(rgb0, alpha0, depth[0])
                        out = pushpull_fill(rgb0, alpha0).clamp(0, 1)
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
            save_state(args.state, arms, done, metrics, coverage, {"scene": scene_desc, "scene_stats": scene_stats,
                                                                  "depth_rows": depth_rows})
            last = time.monotonic() - t0
            print(f"[gs-shift] clip {index} done ({last:.0f}s) {json.dumps(scene.summary())}", flush=True)
        if len(done) == args.expected_clips and args.output is not None:
            report(args, arms, metrics, coverage, {"scene": scene_desc, "scene_stats": None,
                                                   "depth_rows": depth_rows}, done)
        else:
            print(f"SHARD_DONE {len(done)} clips", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
