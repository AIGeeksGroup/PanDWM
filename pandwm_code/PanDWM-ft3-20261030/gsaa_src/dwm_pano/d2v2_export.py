"""D2 v2 refiner inputs for generated clips (2026-10-31, PanDWM-d2-refiner-20261021/D2V2_PLAN_20261031.md).

As d2_export_shift.py, but with the scene the round-trip pairs and the GS-Dev G3b numbers use (d2v2_make_pairs:
G3b decoder, V2 clip DA3 + fusion, in-plane offsets, camera-time poses, v0.x boxes) and, besides the refiner's render
condition, the pixels the evaluation composites with:

    <out-root>/clipNNN/render_dy<d>.pt
        {"render": (35, 2, 16, 32, 288) fp16, "mask": (35, 2, 32, 288) uint8     (D2 condition, d2_render format)
         "pixels": (35, 3, 256, 4114) uint8   the push-pull filled render, exactly eval_gs_shift's 4D output
         "fixmask": (35, 256, 4114) uint8     1 where the render is a hole (alpha < 0.5) or its content is magnified
                                              > mag_thr x against its source view (in band only)
         "meta"}

Frames 3..34 of "pixels" are the frames eval_gs_shift scores; frames 0..2 are rendered too so the refiner's two
windows have their whole input.  Clips: the eval149 generations (clips_root/clipNNN/latents.pt, summary.json).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import dwm.common
from dwm_pano.d2_render import encode_condition
from dwm_pano.d2v2_make_pairs import render_frame, scene_options
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.export_gs_batch149 import FRAMES
from dwm_pano.gs_scene_v3 import SceneBuilder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--clips-root", required=True, type=Path)
    p.add_argument("--clips", required=True, help="comma list or start:end")
    p.add_argument("--shifts", default="0,1,-1,2,-2,4,-4")
    p.add_argument("--fps-stride", default="10,20", help="dataset fps,stride (Test-S: 10,5)")
    p.add_argument("--mag-thr", type=float, default=2.0)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--time-budget-s", type=float, default=0.0)
    p.add_argument("--da3-src", required=True)
    p.add_argument("--da3-anyview", required=True)
    p.add_argument("--da3-metric", required=True)
    return p.parse_args()


def parse_clips(text):
    if ":" in text:
        a, b = (int(x) for x in text.split(":"))
        return list(range(a, b))
    return [int(x) for x in text.split(",") if x.strip()]


@torch.no_grad()
def main():
    args = parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    opts = scene_options()
    shifts = [float(s) for s in args.shifts.split(",") if s.strip()]
    if any(float(f"{d:g}") != d for d in shifts) or len({f"{d:g}" for d in shifts}) != len(shifts):
        raise ValueError("shifts must have distinct, exactly representable %g names")
    fps, stride = (float(x) for x in args.fps_stride.split(","))
    from dwm_pano_4d.da3_depth import Da3Ranger
    from dwm_pano.recon.render_ours import camera_time_infos
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
        install_strict_pano_hardseam_stub()
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl,
                                   fps_stride_tuples=[[int(fps), stride]])
        cfg["base_dataset"]["sequence_length"] = FRAMES
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        by_key = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in
                  enumerate(zip(base.clips, base.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring
        job_start, last = time.monotonic(), 0.0
        for index in parse_clips(args.clips):
            src = args.clips_root / f"clip{index:03d}"
            dst = args.out_root / f"clip{index:03d}"
            todo = [d for d in shifts if not (dst / f"render_dy{d:g}.pt").is_file()]
            if not todo:
                continue
            if args.time_budget_s and last and time.monotonic() - job_start + 1.3 * last > args.time_budget_s:
                print(f"[d2v2-export] budget reached before clip {index}", flush=True)
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
            scene = SceneBuilder(pipe, batch, camera_time_infos(base.clips[ds_index]), latents, opts, device,
                                 da3=da3)
            out = {d: {k: [] for k in ("render", "mask", "pixels", "fixmask", "coverage", "magnified")}
                   for d in todo}
            for t in range(FRAMES):
                for d in todo:
                    rgb, alpha, mag_map = render_frame(pipe, scene, t, d, with_mag=True)
                    lat, mask, canvas = encode_condition(pipe, rgb, alpha)
                    valid = pipe.stitch_valid
                    hole = (alpha < 0.5) & valid
                    magnified = (mag_map > args.mag_thr) & ~hole & valid
                    o = out[d]
                    o["render"].append(lat.cpu())
                    o["mask"].append(mask.cpu())
                    o["pixels"].append((canvas * 255.0).round().clamp(0, 255).to(torch.uint8).cpu())
                    o["fixmask"].append((hole | magnified).to(torch.uint8).cpu())
                    o["coverage"].append(float((alpha[valid] >= 0.5).float().mean()))
                    o["magnified"].append(float(magnified[valid].float().mean()))
            dst.mkdir(parents=True, exist_ok=True)
            for d in todo:
                o = out[d]
                meta = {"clip_index": index, "dy": d, "scene_token": summary["scene_token"],
                        "anchor_timestamp_us": int(summary["anchor_timestamp_us"]),
                        "frame_timestamps_us": [int(x) for x in batch["frame_timestamps_us"][0].tolist()],
                        "mode": "shift_v2", "coverage": o["coverage"], "coverage_mean": sum(o["coverage"]) / FRAMES,
                        "magnified_mean": sum(o["magnified"]) / FRAMES, "mag_thr": args.mag_thr,
                        "checkpoint": str(args.checkpoint), "scene": {k: getattr(opts, k) for k in (
                            "da3_window", "da3_fuse", "plane_offset_px", "box_reading", "drop_bottom_rows")},
                        "scene_stats": scene.summary(), "latents": str(src / "latents.pt")}
                path = dst / f"render_dy{d:g}.pt"
                tmp = path.with_suffix(".tmp")
                torch.save({"render": torch.stack(o["render"]), "mask": torch.stack(o["mask"]),
                            "pixels": torch.stack(o["pixels"]), "fixmask": torch.stack(o["fixmask"]), "meta": meta},
                           tmp)
                tmp.replace(path)
            last = time.monotonic() - t0
            print(f"[d2v2-export] clip {index} shifts {todo} cov "
                  f"{[round(sum(out[d]['coverage']) / FRAMES, 3) for d in todo]} ({last:.0f}s)", flush=True)
        print("D2V2_EXPORT_TASK_DONE", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
