"""Reference frames for Difix on the GS review videos (2026-10-20, Difix line of GS_GAP_ANALYSIS_20261020.md).

For every review camera of export_gs_v3_clips.py (near rig views chase / sweep / pan, the 3 m compare camera, the far
chase 4 m behind / 2.5 m up and the +/-2 m sweep), the clip's own generated pano frame (VAE decode of latents.pt) is
sampled along that camera's pixel rays from the rig centre (the ideal band mapping of the pano canvas).  Pixels whose
ray leaves the +/-16.7 deg band are black.  These are the "same direction, original position" reference images that
difix_ref conditions on; they are never shown in the review package.

  <out-root>/clipNNN/ref_views.mp4    3 panels (chase | sweep | pan), 960 x 256 each   (for 3d_views, 4d_clean_views)
  <out-root>/clipNNN/ref_compare.mp4  1 panel, 960 x 368                               (for both 4d_clean_compare panels)
  <out-root>/clipNNN/ref_far.mp4      2 panels (far chase | far sweep), 1280 x 640 each (for 3d_far_views)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dwm.common
from dwm.utils.preview import save_tensor_to_video
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.export_gs_4d_clean import COMPARE_C2W, intrinsics
from dwm_pano.export_gs_batch149 import FRAMES
from dwm_pano.export_gs_batch149 import cameras as far_cameras
from dwm_pano.export_gs_clean_views import VIEWS
from dwm_pano.export_gs_long_demo import pinhole
from dwm_pano.export_gs_v3_clips import ray_directions
from dwm_pano.gs_geometry import PANO_LAT_HALF_RAD
from dwm_pano.gs_scene_v3 import SceneBuilder, SceneOptions


def sample_band(canvas, dirs):
    """canvas (3, H, Wc) in [0, 1], dirs (V, h, w, 3) -> (V, 3, h, w); black outside the pano band."""
    px, py, pz = -dirs[..., 1], dirs[..., 0], dirs[..., 2]
    azimuth = torch.atan2(px, py)
    elevation = torch.asin(pz.clamp(-1.0, 1.0))
    height, width = canvas.shape[-2:]
    gx = (azimuth + math.pi) / (2.0 * math.pi) * 2.0 - 1.0
    gy = ((PANO_LAT_HALF_RAD - elevation) / (2.0 * PANO_LAT_HALF_RAD)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    rgb = F.grid_sample(canvas[None].expand(dirs.shape[0], -1, -1, -1), grid, mode="bilinear",
                        padding_mode="border", align_corners=False)
    inside = (elevation.abs() <= PANO_LAT_HALF_RAD)[:, None]
    return torch.where(inside, rgb, torch.zeros_like(rgb))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--clips-root", required=True, type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=149)
    p.add_argument("--clips", default="", help="comma list overriding start / end")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--hfov-deg", type=float, default=70.0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    started = ensure_single_rank_process_group(device)
    try:
        config = load_config(args.config)
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(args.out_root),
                                                      config=config, device=device, resume_from=None)
        pipe.vae.eval()
        pipe.decoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
        pipe.decoder.eval()
        install_strict_pano_hardseam_stub()
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl, fps_stride_tuples=[[10, 20]])
        cfg["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        by_key = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in
                  enumerate(zip(base.clips, base.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring
        K_cmp, h_cmp = intrinsics(args.width, args.hfov_deg, 14.0, 16.0)
        K_view, h_view = intrinsics(args.width, args.hfov_deg, 14.0, 7.0)
        Kc = torch.tensor(K_cmp[None], device=device, dtype=torch.float32)
        Kv = torch.tensor(np.stack([K_view] * 3), device=device, dtype=torch.float32)
        Kf = torch.tensor(np.stack([pinhole(1280, 640, 90.0)] * 2), device=device, dtype=torch.float32)
        clips = [int(x) for x in args.clips.split(",") if x.strip()] or list(range(args.start, args.end))
        for index in clips:
            src = args.clips_root / f"clip{index:03d}"
            dst = args.out_root / f"clip{index:03d}"
            if (dst / "refs.json").is_file():
                continue
            t0 = time.monotonic()
            summary = json.loads((src / "summary.json").read_text())
            ds_index = by_key[(summary["scene_token"], int(summary["anchor_timestamp_us"]))]
            batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [ds_index]),
                                                          batch_size=1, shuffle=False, num_workers=0,
                                                          collate_fn=CollateFnIgnoring(keys=[]))))
            latents = torch.load(src / "latents.pt", map_location="cpu", weights_only=True)["latents"]
            scene = SceneBuilder(pipe, batch, base.clips[ds_index], latents, SceneOptions(), device)
            rows = {"ref_views.mp4": [], "ref_compare.mp4": [], "ref_far.mp4": []}
            for t in range(FRAMES):
                canvas = scene.canvas(t).float()
                for name, c2w_list, K, width, height in (
                        ("ref_views.mp4", [VIEWS[v][1](t) for v in ("chase", "sweep", "pan")], Kv, args.width, h_view),
                        ("ref_compare.mp4", [COMPARE_C2W], Kc, args.width, h_cmp),
                        ("ref_far.mp4", list(far_cameras(t)), Kf, 1280, 640)):
                    c2w = torch.tensor(np.stack(c2w_list), device=device, dtype=torch.float32)
                    panels = sample_band(canvas, ray_directions(c2w, K, width, height))
                    rows[name].append(torch.cat(list(panels.clamp(0, 1).cpu()), dim=-1))
            dst.mkdir(parents=True, exist_ok=True)
            for name, frames in rows.items():
                tmp = dst / (name[:-4] + ".tmp.mp4")
                save_tensor_to_video(str(tmp), "libx264", float(batch["fps"][0]), frames)
                tmp.replace(dst / name)
            (dst / "refs.json").write_text(json.dumps({"clip_index": index, "frames": FRAMES,
                                                       "latents": str(src / "latents.pt")}) + "\n")
            print(f"[difix-refs] clip {index} done ({time.monotonic() - t0:.0f}s)", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
