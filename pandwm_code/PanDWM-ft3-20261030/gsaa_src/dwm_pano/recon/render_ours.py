"""Our 3D / 4D Gaussian renders of the tab:gs_recon targets (2026-10-22, RECON_PLAN_20261022.md).

Scene: the v4 settings (C2 decoder step 3000, DA3 scale relabel, the 2 bottom decoder rows dropped) on the RECORDED
panoramas of the Test-S clip, VAE posterior-mean latents.  Only clip frames 0..20 enter (frame 20 = input), so no
target frame and no later frame can leak; the scene canvases (DA3 input) are the recorded panoramas themselves.
  3d     frame 20's Gaussians; DA3 sees frame 20 only (da3_gap 0).
  4d     + the static Gaussians of the history frames 0, 5, 10, 15 (gs_scene_v3.static_world, moved into frame 20's
         ego); DA3 windows of 4 frames back (gap 4, clipped at frame 20).  Frames are registered with their
         camera-time ego poses (targets.camera_time_pose; the boxes follow through an adjusted lidar2ego) and the
         dynamic boxes are read as mmdet3d v0.x (box_reading "mmdet3d_v0x"), section 44.
  4d_v4  the 4d scene exactly as v4 built it: the pkl's LiDAR-time ego poses for the registration and the DA3 windows,
         the legacy box reading.
A history frame that IS the target frame is left out for that target (targets.json in_history).  Every target is
rendered at the six cameras of targets.json (c2w in frame 20's camera-time ego, K scaled to the output size with
calibrated_stitch.prepare_gsplat_intrinsics) through pipe.render_targets (gsplat RGB+ED, antialiased, eps2d 0.1);
the 4d modes composite current over fill (gs_scene_v3.composite / composite_depth).

    <out-root>/<mode>/<H>x<W>/sample_<NNN>.pt  {"targets": [...], "rgb": (n, 6, 3, H, W) uint8,
                                                "alpha": (n, 6, H, W) uint8, "depth": (n, 6, H, W) fp16, "meta": {}}
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from pyquaternion import Quaternion

import dwm.common
from dwm_pano.d2_render import encode_halves_mean
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.export_gs_4d_demo import transform_gaussians
from dwm_pano.gs_scene_v3 import SceneBuilder, SceneOptions, composite, composite_depth
from dwm_pano.recon.targets import camera_time_infos, camera_time_pose, pose  # noqa: F401 (re-export)

INPUT_FRAME = 20
HISTORY = (0, 5, 10, 15)
MODES = {"3d": {"da3_gap": 0, "history": False, "poses": "pkl", "box_reading": "legacy"},
         "4d": {"da3_gap": 4, "history": True, "poses": "camera", "box_reading": "mmdet3d_v0x"},
         "4d_v4": {"da3_gap": 4, "history": True, "poses": "pkl", "box_reading": "legacy"}}




def parse_res(text):
    return [tuple(int(v) for v in r.split("x")) for r in text.split(",") if r.strip()]


def parse_samples(text, count):
    if not text:
        return list(range(count))
    if ":" in text:
        a, b = (int(x) for x in text.split(":"))
        return list(range(a, b))
    return [int(x) for x in text.split(",") if x.strip()]


def render(pipe, g, c2w, K, hw):
    h, w = hw
    if g.shape[0] == 0:
        z = torch.zeros(6, h, w, device=c2w.device)
        return torch.zeros(6, 3, h, w, device=c2w.device), z.clone(), z
    rgb, depth, alpha = pipe.render_targets(g, target_c2w=c2w, target_intrinsics=K, width=w, height=h)
    return rgb, depth, alpha


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--resolutions", default="224x400,352x640")
    p.add_argument("--modes", default="3d,4d")
    p.add_argument("--samples", default="", help="positions in targets.json: list or a:b")
    p.add_argument("--da3-src", required=True)
    p.add_argument("--da3-anyview", required=True)
    p.add_argument("--da3-metric", required=True)
    p.add_argument("--scene-overrides", default="", help="JSON of SceneOptions fields applied to every mode")
    args = p.parse_args()
    overrides = json.loads(args.scene_overrides) if args.scene_overrides else {}
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    data = json.loads(args.targets.read_text())
    res = parse_res(args.resolutions)
    modes = [m for m in args.modes.split(",") if m]
    if any(m not in MODES for m in modes):
        raise ValueError(f"modes {modes}, known {list(MODES)}")
    from dwm_pano_4d.da3_depth import Da3Ranger
    from dwm_pano.calibrated_stitch import prepare_gsplat_intrinsics
    da3 = Da3Ranger(args.da3_src, args.da3_anyview, args.da3_metric, device)
    started = ensure_single_rank_process_group(device)
    try:
        config = load_config(args.config)
        torch.manual_seed(int(config["generator_seed"]))
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(args.out_root),
                                                      config=config, device=device, resume_from=None)
        pipe.vae.eval()
        pipe.decoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
        pipe.decoder.eval()
        pipe.decoder_wrapper.eval()
        pipe.rasterize_mode = "antialiased"
        pipe.active_rasterize_eps2d = 0.1
        install_strict_pano_hardseam_stub()
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl,
                                   fps_stride_tuples=[[10, 5.0]])
        cfg["base_dataset"]["sequence_length"] = 35
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        by_key = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in
                  enumerate(zip(base.clips, base.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring
        for pos in parse_samples(args.samples, data["count"]):
            s = data["samples"][pos]
            outs = {(m, hw): args.out_root / m / f"{hw[0]}x{hw[1]}" / f"sample_{s['test_index']:03d}.pt"
                    for m in modes for hw in res}
            if all(o.is_file() for o in outs.values()):
                continue
            t0 = time.monotonic()
            index = by_key[(s["scene_token"], int(s["anchor_timestamp_us"]))]
            batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [index]), batch_size=1,
                                                          shuffle=False, num_workers=0,
                                                          collate_fn=CollateFnIgnoring(keys=[]))))
            clip = base.clips[index]
            if int(clip[INPUT_FRAME]["timestamp"]) != int(s["input"]["timestamp"]):
                raise RuntimeError(f"sample {pos}: input frame differs from targets.json")
            halves = batch["vae_images"][0, :INPUT_FRAME + 1].to(device).float()   # frames 0..20 only
            latents = encode_halves_mean(pipe, halves).cpu()
            names = list(s["targets"])
            cams = {n: s["targets"][n]["cams"] for n in names}
            for mode in modes:
                spec = MODES[mode]
                opts = SceneOptions(da3_relabel=True, da3_mode="scale", drop_bottom_rows=2, da3_gap=spec["da3_gap"],
                                    window=(-20, -15, -10, -5), box_reading=spec["box_reading"])
                if overrides:
                    opts = dataclasses.replace(opts, **overrides)
                infos = clip[:INPUT_FRAME + 1]
                if spec["poses"] == "camera":
                    infos = camera_time_infos(infos)
                scene = SceneBuilder(pipe, batch, infos, latents, opts, device, da3=da3)
                for t in range(INPUT_FRAME + 1):
                    scene._canvas[t] = pipe._unpadded_stitch_canvas(halves[t:t + 1])[0].clamp(0, 1)
                current = scene.decode(INPUT_FRAME)
                static = {f: scene.static_world(f) for f in HISTORY} if spec["history"] else {}
                to_t = torch.linalg.inv(torch.as_tensor(scene.world[INPUT_FRAME], device=device, dtype=torch.float32))
                for hw in res:
                    rgbs, alphas, depths = [], [], []
                    for n in names:
                        c2w = torch.as_tensor(np.asarray([c["c2w"] for c in cams[n]]), device=device,
                                              dtype=torch.float32)
                        K = torch.stack([prepare_gsplat_intrinsics(torch.as_tensor(c["K"], dtype=torch.float64),
                                                                   (900, 1600), hw) for c in cams[n]]).float().to(device)
                        rgb, depth, alpha = render(pipe, current, c2w, K, hw)
                        if spec["history"]:
                            hist_ts = s["history_timestamps"]
                            frames = [f for f, ts in zip(HISTORY, hist_ts)
                                      if int(ts) != int(s["targets"][n]["timestamp"])]
                            parts = [static[f] for f in frames if static[f].shape[0]]
                            if parts:
                                fill = transform_gaussians(torch.cat(parts).float(), to_t)
                                rgb_f, depth_f, alpha_f = render(pipe, fill, c2w, K, hw)
                                depth = composite_depth(depth, alpha, depth_f, alpha_f)
                                rgb, alpha = composite(rgb, alpha, rgb_f, alpha_f)
                        rgbs.append((rgb.clamp(0, 1) * 255).round().byte().cpu())
                        alphas.append((alpha.clamp(0, 1) * 255).round().byte().cpu())
                        depths.append(depth.half().cpu())
                    out = outs[(mode, hw)]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix(".tmp")
                    torch.save({"targets": names, "rgb": torch.stack(rgbs), "alpha": torch.stack(alphas),
                                "depth": torch.stack(depths),
                                "meta": {"method": f"ours_{mode}", "test_index": s["test_index"],
                                         "checkpoint": str(args.checkpoint), "resolution": list(hw),
                                         "scene": {"da3_relabel": True, "da3_mode": "scale", "drop_bottom_rows": 2,
                                                   "da3_gap": opts.da3_gap, "poses": spec["poses"],
                                                   "box_reading": spec["box_reading"],
                                                   "history": list(HISTORY) if spec["history"] else [],
                                                   "overrides": overrides},
                                         "scene_stats": scene.summary()}}, tmp)
                    tmp.replace(out)
            print(f"[recon-ours] sample {pos} (test {s['test_index']}) {names} ({time.monotonic() - t0:.0f}s)",
                  flush=True)
        print("RECON_OURS_DONE", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
