"""GS renders of the review clips at a laterally shifted rig, encoded into video-model latents (2026-10-20, D1 of
PanDWM-gs4d-20261002/GS_GAP_ANALYSIS_20261020.md, step 1 of 2; step 2 = PanDWM_eval149 d1_sdedit.py on HPC3).

For every frame of a clip the gs_scene_v3 scene (4D: frame t over the static fill of nearby frames) is rendered at the
six nuScenes cameras with the ego shifted sideways by dy and stitched with the pipeline's own hard seam into the
4114 x 256 canvas, exactly as eval_gs_shift.py renders its arms; holes (alpha < 0.5) are filled with pushpull_fill.
The canvas is cut back into the two 2304-wide halves (_unpadded_stitch_canvas is left half | right half without its
first 494 overlap columns, so left = canvas[..., :2304] and right = canvas[..., 1810:]) and encoded with the same
encode_half_latents the GS decoder uses (SD3.5 VAE, posterior mean, the video model's scaling).

  <out-root>/clipNNN/render_dy<d>.pt   {"latents": (35, 2, 16, 32, 288) fp16, "dy": d, "scene": {...}}
  <out-root>/clipNNN/render_dy<d>.mp4  the rendered canvases (for side-by-side review)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import dwm.common
from dwm.utils.preview import save_tensor_to_video
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.export_gs_batch149 import FRAMES
from dwm_pano.export_gs_v2_clips import pushpull_fill
from dwm_pano.gs_scene_v3 import SceneBuilder, SceneOptions, composite

LEFT_W = 2304


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--clips-root", required=True, type=Path)
    p.add_argument("--clips", required=True, help="comma list")
    p.add_argument("--shifts", default="0,2", help="signed lateral offsets in metres")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--da3-relabel", action="store_true")
    p.add_argument("--da3-mode", choices=["ray", "scale"], default="scale")
    p.add_argument("--drop-bottom-rows", type=int, default=0)
    p.add_argument("--da3-src", default="")
    p.add_argument("--da3-anyview", default="")
    p.add_argument("--da3-metric", default="")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    opts = SceneOptions(da3_relabel=args.da3_relabel, da3_mode=args.da3_mode, drop_bottom_rows=args.drop_bottom_rows)
    da3 = None
    if args.da3_relabel:
        from dwm_pano_4d.da3_depth import Da3Ranger
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
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl, fps_stride_tuples=[[10, 20]])
        cfg["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        by_key = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in
                  enumerate(zip(base.clips, base.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring
        shifts = [float(s) for s in args.shifts.split(",") if s.strip()]
        scene_desc = {k: getattr(opts, k) for k in ("edge_rel_h", "edge_rel_v", "window", "da3_relabel", "da3_mode",
                                                     "drop_bottom_rows", "static_range", "max_scale")}
        for index in [int(x) for x in args.clips.split(",") if x.strip()]:
            src = args.clips_root / f"clip{index:03d}"
            dst = args.out_root / f"clip{index:03d}"
            todo = [d for d in shifts if not (dst / f"render_dy{d:g}.pt").is_file()]
            if not todo:
                continue
            t0 = time.monotonic()
            summary = json.loads((src / "summary.json").read_text())
            ds_index = by_key[(summary["scene_token"], int(summary["anchor_timestamp_us"]))]
            batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [ds_index]),
                                                          batch_size=1, shuffle=False, num_workers=0,
                                                          collate_fn=CollateFnIgnoring(keys=[]))))
            latents = torch.load(src / "latents.pt", map_location="cpu", weights_only=True)["latents"]
            scene = SceneBuilder(pipe, batch, base.clips[ds_index], latents, opts, device, da3=da3)
            cam2ego = pipe.nuscenes_camera_to_ego.float()
            valid = pipe.stitch_valid
            out_lat = {d: [] for d in todo}
            out_rgb = {d: [] for d in todo}
            for t in range(FRAMES):
                current = scene.decode(t)
                fill = scene.fill(t)
                for d in todo:
                    shifted = cam2ego.clone()
                    shifted[:, 1, 3] += d
                    rgb, _, alpha, _ = pipe._render_nuscenes6_stitched(current, target_c2w=shifted)
                    if fill.shape[0]:
                        rgb_f, _, alpha_f, _ = pipe._render_nuscenes6_stitched(fill, target_c2w=shifted)
                        rgb, alpha = composite(rgb, alpha, rgb_f, alpha_f)
                    alpha = torch.where(valid[None], alpha, torch.zeros_like(alpha))
                    canvas = pushpull_fill(rgb[0].clamp(0, 1), alpha[0]).clamp(0, 1)          # (3, 256, 4114)
                    width = canvas.shape[-1]
                    halves = torch.stack([canvas[..., :LEFT_W], canvas[..., width - LEFT_W:]])  # (2, 3, 256, 2304)
                    lat = pipe.encode_half_latents(halves[None].to(pipe.vae.dtype))           # (1, 2, 16, 32, 288)
                    out_lat[d].append(lat[0].half().cpu())
                    out_rgb[d].append(canvas.cpu())
            dst.mkdir(parents=True, exist_ok=True)
            for d in todo:
                torch.save({"latents": torch.stack(out_lat[d]), "dy": d, "scene": scene_desc,
                            "checkpoint": str(args.checkpoint), "clip_index": index},
                           dst / f"render_dy{d:g}.pt")
                frames = [torch.nn.functional.interpolate(f[None], size=(128, 2056), mode="area")[0]
                          for f in out_rgb[d]]
                save_tensor_to_video(str(dst / f"render_dy{d:g}.mp4"), "libx264", float(batch["fps"][0]), frames)
            print(f"[shift-lat] clip {index} shifts {todo} done ({time.monotonic() - t0:.0f}s)", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
