"""One-frame scale diagnostic for external_gs.py (2026-10-28): per-camera median of our clip-DA3 metric z over the
method's z, and its spread, for InfiniSplat (provided / panorama mode) and the DA3-GIANT Gaussian head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import dwm.common
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.export_gs_batch149 import FRAMES
from dwm_pano.external_gs import ExternalGaussians
from dwm_pano.gs_scene_v3 import SceneBuilder, SceneOptions
from dwm_pano.recon.targets import CAM_ORDER


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--clips-root", required=True, type=Path)
    p.add_argument("--clip", type=int, default=0)
    p.add_argument("--frames", default="10,20")
    p.add_argument("--method", required=True)
    p.add_argument("--is-mode", default="provided")
    p.add_argument("--weights", required=True)
    p.add_argument("--da3-src", default="")
    p.add_argument("--cache", required=True)
    args = p.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    ext = ExternalGaussians(args.method, args.weights, device, args.cache, 504, args.da3_src, args.is_mode)
    started = ensure_single_rank_process_group(device)
    try:
        config = load_config(args.config)
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path="/tmp", config=config,
                                                      device=device, resume_from=None)
        pipe.vae.eval()
        install_strict_pano_hardseam_stub()
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl, fps_stride_tuples=[[10, 20]])
        cfg["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        by_key = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in
                  enumerate(zip(base.clips, base.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring
        src = args.clips_root / f"clip{args.clip:03d}"
        summary = json.loads((src / "summary.json").read_text())
        ds_index = by_key[(summary["scene_token"], int(summary["anchor_timestamp_us"]))]
        batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [ds_index]), batch_size=1,
                                                      collate_fn=CollateFnIgnoring(keys=[]))))
        latents = torch.load(src / "latents.pt", map_location="cpu", weights_only=True)["latents"]
        from dwm_pano.recon.targets import camera_time_infos
        scene = SceneBuilder(pipe, batch, camera_time_infos(base.clips[ds_index]), latents, SceneOptions(), device)
        for t in (int(x) for x in args.frames.split(",")):
            ext(scene, t, args.clip)
            d = ext.last
            row = {"method": args.method, "is_mode": args.is_mode, "frame": t, "scale": d["scale"],
                   **{k: round(v, 4) for k, v in ext.stats[-1].items() if isinstance(v, float)}}
            for i, name in enumerate(CAM_ORDER):
                v = d["valid"][i]
                r = (d["z_ours"][i][v] / d["z_m"][i][v]) if bool(v.any()) else torch.zeros(1)
                row[name] = {"median": round(float(r.median()), 3), "iqr_rel": round(float(
                    (r.quantile(0.75) - r.quantile(0.25)) / r.median()), 3), "pixels": int(v.sum())}
            print("DIAG " + json.dumps(row), flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
