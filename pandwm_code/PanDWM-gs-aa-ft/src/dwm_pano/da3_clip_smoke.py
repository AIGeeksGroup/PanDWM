"""Smoke test and temporal diagnostic for clip-level DA3 (2026-10-26, GS_CLIPDEPTH_PLAN_20261026.md sections 1, 5).

For a few generated Dev clips (eval_gs_shift's clip layout) this computes the DA3 crop depth of every frame three ways:
  V0  the current per-frame windows (t - 4, t, t + 4), each frame with its own ground scale (gs_scene_v3 "frame");
  V1  one clip-level pass (or --chunk N), one ground scale (SceneBuilder._clip_da3_crops);
  V2  V1 + static depth fusion (da3_fusion.fuse_static_depth, the registered offsets / tolerance);
and reports per clip: wall time and peak memory of V1 / V2, the temporal diagnostic (median |z_{t-1 -> t} - z_t| / z_t
of static pixels, same static masks for all three), the V1 / V0 depth ratio, and the fusion stats.  The first clip also
checks that the edited Da3Ranger returns bit-identical frame-mode ranges to the backup module (--old-da3) and saves a
picture of the front crop at frames 10 / 11 (RGB, V0, V1, V2 inverse depth).  No decoder, no rendering.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

import dwm.common
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.export_gs_batch149 import FRAMES
from dwm_pano.gs_scene_v3 import SceneBuilder, SceneOptions


def load_old_ranger(path):
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("da3_depth_old", str(path))         # the backup has no .py suffix
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("da3_depth_old", loader))
    loader.exec_module(mod)
    return mod.Da3Ranger


def inv_depth_img(z):
    v = 1.0 / z.clamp(1.0, 200.0)
    v = (v - v.min()) / (v.max() - v.min() + 1e-9)
    return (torch.stack([v, v ** 0.5, 1 - v], dim=0) * 255).byte()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--clips-root", required=True, type=Path)
    p.add_argument("--clips", default="0,1")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--fps-stride", default="10,20")
    p.add_argument("--box-reading", default="mmdet3d_v0x")
    p.add_argument("--camera-time-poses", action="store_true")
    p.add_argument("--da3-src", required=True)
    p.add_argument("--da3-anyview", required=True)
    p.add_argument("--da3-metric", required=True)
    p.add_argument("--old-da3", default="")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    from dwm_pano_4d.da3_depth import Da3Ranger
    from dwm_pano.da3_fusion import ClipGeometry, fuse_static_depth, temporal_consistency
    da3 = Da3Ranger(args.da3_src, args.da3_anyview, args.da3_metric, device)
    started = ensure_single_rank_process_group(device)
    rows = []
    try:
        config = load_config(args.config)
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(args.out), config=config,
                                                      device=device, resume_from=None)
        pipe.vae.eval()
        install_strict_pano_hardseam_stub()
        fps_stride = [float(x) if "." in x else int(x) for x in args.fps_stride.split(",")]
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl, fps_stride_tuples=[fps_stride])
        cfg["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        by_key = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in
                  enumerate(zip(base.clips, base.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring
        for n_clip, index in enumerate(int(x) for x in args.clips.split(",")):
            src = args.clips_root / f"clip{index:03d}"
            summary = json.loads((src / "summary.json").read_text())
            ds_index = by_key[(summary["scene_token"], int(summary["anchor_timestamp_us"]))]
            batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [ds_index]), batch_size=1,
                                                          shuffle=False, num_workers=0,
                                                          collate_fn=CollateFnIgnoring(keys=[]))))
            latents = torch.load(src / "latents.pt", map_location="cpu", weights_only=True)["latents"]
            infos = base.clips[ds_index]
            if args.camera_time_poses:
                from dwm_pano.recon.targets import camera_time_infos
                infos = camera_time_infos(infos)
            opts = SceneOptions(da3_relabel=True, da3_mode="scale", drop_bottom_rows=2, box_reading=args.box_reading,
                                da3_window="clip", da3_chunk=args.chunk, da3_fuse=True)
            scene = SceneBuilder(pipe, batch, infos, latents, opts, device, da3=da3)
            frames = scene.frames
            row = {"clip": index, "frames": frames}
            # V1
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.monotonic()
            depth, sky, K, chunks = scene._clip_da3_crops()
            torch.cuda.synchronize()
            row.update(v1_s=time.monotonic() - t0, v1_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                       chunks=chunks)
            h, w = depth.shape[-2:]
            rig = da3.rig(pipe)
            scale = da3.ground_scale(rig, K.repeat(frames, 1, 1), depth, sky < 0.3, frames)
            v1 = (depth * scale).view(frames, 6, h, w)
            sky4 = sky.view(frames, 6, h, w)
            # V2
            t0 = time.monotonic()
            geo = ClipGeometry(v1, sky4, K, da3.owned_masks(pipe), rig["cam2ego"].to(device),
                               torch.as_tensor(scene.world, device=device),
                               [scene.frame_boxes(s) for s in range(frames)], opts.far_cut, opts.box_margin)
            v2, fstats = fuse_static_depth(geo, opts.fuse_offsets, opts.fuse_tol_rel)
            torch.cuda.synchronize()
            row.update(v2_s=time.monotonic() - t0, v1_scale=scale, **fstats)
            # V0: the frame-mode windows of gs_scene_v3.da3_maps, crop depth of the window's frame t
            t0 = time.monotonic()
            v0 = torch.empty_like(v1)
            canvases = {f: scene.canvas(f) for f in range(frames)}
            v0_scales = []
            check = None
            for t in range(frames):
                win = sorted({max(0, t - 4), t, min(frames - 1, t + 4)})
                e2w = torch.as_tensor(scene.world[win], dtype=torch.float64)
                res = da3.frame_ranges(pipe, torch.stack([canvases[f] for f in win]), e2w)
                i = win.index(t)
                v0[t] = res["depth_crop"][i * 6:(i + 1) * 6] * res["scale_ground"]
                v0_scales.append(res["scale_ground"])
                if n_clip == 0 and t == 10 and args.old_da3:
                  try:
                    old = load_old_ranger(args.old_da3)(args.da3_src, args.da3_anyview, args.da3_metric, device)
                    res_old = old.frame_ranges(pipe, torch.stack([canvases[f] for f in win]), e2w)
                    res_new2 = da3.frame_ranges(pipe, torch.stack([canvases[f] for f in win]), e2w)
                    check = {k: {"max_abs_old_vs_new": float((res_old[k] - res[k]).abs().max()),
                                 "max_abs_new_vs_new": float((res_new2[k] - res[k]).abs().max())}
                             for k in ("range_ground", "range_pose", "sky")}
                    check["scale_ground_old_new"] = [res_old["scale_ground"], res["scale_ground"]]
                    del old
                  except Exception as exc:                                   # diagnostic only, keep going
                    check = {"error": repr(exc)}
            row.update(v0_s=time.monotonic() - t0, v0_scale_cv=float(np.std(v0_scales) / np.mean(v0_scales)))
            if check is not None:
                row["frame_mode_identity"] = check
            for name, d in (("v0", v0), ("v1", v1), ("v2", v2)):
                tc = temporal_consistency(geo, d)
                row.update({f"{name}_{k}": v for k, v in tc.items()})
            st = torch.stack(geo.mask)
            row["v1_over_v0_median"] = float((v1[st] / v0[st].clamp_min(1e-3)).median())
            row["v2_over_v1_mean_abs_rel"] = float(((v2 - v1).abs() / v1.clamp_min(1e-3))[st].mean())
            # the decoder-grid ranges SceneBuilder will use, through the real code path
            t0 = time.monotonic()
            scene._clip_da3()
            row["scene_clip_da3_s"] = time.monotonic() - t0
            r10 = scene.da3_maps(10)["range"]
            row["scene_range10_median"] = float(r10[r10 > 0].median())
            row["scene_stats"] = scene.summary()
            if n_clip == 0:
                from PIL import Image
                crops = da3.crop_images(pipe, torch.stack([canvases[10], canvases[11]]))
                tiles = []
                for j, t in enumerate((10, 11)):
                    rgb = (crops[j * 6] * 255).byte()
                    tiles.append(torch.cat([rgb] + [inv_depth_img(d[t, 0]) for d in (v0, v1, v2)], dim=1))
                img = torch.cat(tiles, dim=2).permute(1, 2, 0).cpu().numpy()
                Image.fromarray(img).save(args.out / f"clip{index:03d}_front_f10_f11.png")
            print("SMOKE_ROW " + json.dumps(row), flush=True)
            rows.append(row)
            del scene, geo, v0, v1, v2, depth, sky
            torch.cuda.empty_cache()
    finally:
        if started:
            torch.distributed.destroy_process_group()
    (args.out / f"smoke_chunk{args.chunk}.json").write_text(json.dumps(rows, indent=1) + "\n")
    print("SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
