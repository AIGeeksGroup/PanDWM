"""Stage A of GS_ANCHOR_ZEROSHOT_PLAN_20261026.md: cache the clip-level DA3 metric crop depth of the generated Dev clips.

For each clip (eval_gs_shift's layout and dataset), SceneBuilder._clip_da3_crops (one DA3 pass over all 35 decoded
frames, GS_CLIPDEPTH_PLAN V1) times the clip's ground scale gives the metric z-depth of the six pinhole crops of every
frame; it is resized (nearest) to the external models' input size and saved with DA3's sky mask (>= 0.3):
  <out>/clip<index:03d>.npz   depth float16 (35, 6, h, w), sky uint8 (35, 6, h, w), scale float
external_gs.py reads it to put the external Gaussians at metric scale (InfiniSplat pins another DA3 revision, so the two
cannot share a process).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dwm.common
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.export_gs_batch149 import FRAMES
from dwm_pano.gs_scene_v3 import SceneBuilder, SceneOptions


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--clips-root", required=True, type=Path)
    p.add_argument("--clip-list", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--hw", default="252x504")
    p.add_argument("--fps-stride", default="10,20")
    p.add_argument("--camera-time-poses", action="store_true")
    p.add_argument("--da3-src", required=True)
    p.add_argument("--da3-anyview", required=True)
    p.add_argument("--da3-metric", required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    h, w = (int(v) for v in args.hw.split("x"))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    from dwm_pano_4d.da3_depth import Da3Ranger
    da3 = Da3Ranger(args.da3_src, args.da3_anyview, args.da3_metric, device)
    started = ensure_single_rank_process_group(device)
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
        for index in (int(x) for x in args.clip_list.split(",") if x.strip()):
            dst = args.out / f"clip{index:03d}.npz"
            if dst.is_file():
                continue
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
            scene = SceneBuilder(pipe, batch, infos, latents, SceneOptions(da3_relabel=True, da3_window="clip"),
                                 device, da3=da3)
            depth, sky, K, _ = scene._clip_da3_crops()
            frames = scene.frames
            scale = da3.ground_scale(da3.rig(pipe), K.repeat(frames, 1, 1), depth, sky < 0.3, frames)
            d = F.interpolate((depth * scale)[:, None].float(), size=(h, w), mode="nearest")[:, 0]
            s = F.interpolate((sky >= 0.3).float()[:, None], size=(h, w), mode="nearest")[:, 0]
            tmp = dst.with_suffix(".tmp.npz")
            np.savez(tmp, depth=d.view(frames, 6, h, w).cpu().numpy().astype(np.float16),
                     sky=s.view(frames, 6, h, w).cpu().numpy().astype(np.uint8), scale=np.float32(scale))
            tmp.replace(dst)
            print(f"[clip-da3-cache] clip {index} scale {scale:.3f}", flush=True)
            del scene, depth, sky
            torch.cuda.empty_cache()
    finally:
        if started:
            torch.distributed.destroy_process_group()
    print("CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
