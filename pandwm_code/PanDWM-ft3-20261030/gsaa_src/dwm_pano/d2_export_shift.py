"""D2 refiner inputs for the 149 review clips (2026-10-21, PanDWM-d2-refiner-20261021/D2_PLAN_20261021.md Task 9).

For every review clip (the v84i4250 generated latents, clips_root/clipNNN/latents.pt) and every signed lateral
offset dy, the v4 far-view scene (C2 decoder + DA3 scale relabel + drop the 2 bottom decoder rows) is rendered
at the rig moved by dy (d2_render.degraded_canvas mode "shift": current frame + 4D fill, pushpull hole fill)
and encoded exactly like the training pairs (d2_render.encode_condition):

    <out-root>/clipNNN/render_dy<d>.pt    {"render": (35, 2, 16, 32, 288) fp16, "mask": (35, 2, 32, 288) uint8,
                                           "meta": {...}}
    <out-root>/clipNNN/render_dy<d>.mp4   the filled canvases at 128 x 2056 (only for --video-clips)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import dwm.common
from dwm.utils.preview import save_tensor_to_video
from dwm_pano.d2_render import degraded_canvas, encode_condition
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.export_gs_batch149 import FRAMES
from dwm_pano.gs_scene_v3 import SceneBuilder, SceneOptions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--clips-root", required=True, type=Path)
    p.add_argument("--clips", required=True, help="comma list or start:end")
    p.add_argument("--shifts", default="0,1,-1,2,-2", help="signed lateral offsets in metres")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--video-clips", default="", help="clips that also get render_dy<d>.mp4")
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
    opts = SceneOptions(da3_relabel=True, da3_mode="scale", drop_bottom_rows=2)
    scene_desc = {k: getattr(opts, k) for k in ("edge_rel_h", "edge_rel_v", "window", "da3_relabel", "da3_mode",
                                                 "drop_bottom_rows", "static_range", "max_scale")}
    shifts = [float(s) for s in args.shifts.split(",") if s.strip()]
    if any(float(f"{d:g}") != d for d in shifts) or len({f"{d:g}" for d in shifts}) != len(shifts):
        raise ValueError("shifts must have distinct, exactly representable %g names (e.g. 0.5, 1, -2)")
    video_clips = set(parse_clips(args.video_clips)) if args.video_clips else set()
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
        job_start, last = time.monotonic(), 0.0
        for index in parse_clips(args.clips):
            src = args.clips_root / f"clip{index:03d}"
            dst = args.out_root / f"clip{index:03d}"
            todo = [d for d in shifts if not (dst / f"render_dy{d:g}.pt").is_file()]
            if not todo:
                continue
            if args.time_budget_s and last and time.monotonic() - job_start + 1.3 * last > args.time_budget_s:
                print(f"[d2-export] budget reached before clip {index}", flush=True)
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
            out = {d: {"render": [], "mask": [], "coverage": [], "frames": []} for d in todo}
            for t in range(FRAMES):
                for d in todo:
                    rgb, alpha = degraded_canvas(pipe, scene, t, "shift", (0.0, d))
                    lat, mask, canvas = encode_condition(pipe, rgb, alpha)
                    out[d]["render"].append(lat.cpu())
                    out[d]["mask"].append(mask.cpu())
                    out[d]["coverage"].append(float((alpha[pipe.stitch_valid] >= 0.5).float().mean()))
                    if index in video_clips:
                        out[d]["frames"].append(torch.nn.functional.interpolate(
                            canvas[None], size=(128, 2056), mode="area")[0].cpu())
            dst.mkdir(parents=True, exist_ok=True)
            for d in todo:
                cov = out[d]["coverage"]
                meta = {"clip_index": index, "dy": d, "scene_token": summary["scene_token"],
                        "anchor_timestamp_us": int(summary["anchor_timestamp_us"]),
                        "frame_timestamps_us": [int(x) for x in batch["frame_timestamps_us"][0].tolist()],
                        "mode": "shift", "coverage": cov, "coverage_mean": sum(cov) / len(cov),
                        "checkpoint": str(args.checkpoint), "scene": scene_desc, "scene_stats": scene.summary(),
                        "latents": str(src / "latents.pt")}
                path = dst / f"render_dy{d:g}.pt"
                tmp = path.with_suffix(".tmp")
                torch.save({"render": torch.stack(out[d]["render"]), "mask": torch.stack(out[d]["mask"]),
                            "meta": meta}, tmp)
                tmp.replace(path)
                if out[d]["frames"]:
                    save_tensor_to_video(str(dst / f"render_dy{d:g}.mp4"), "libx264", float(batch["fps"][0]),
                                         out[d]["frames"])
            last = time.monotonic() - t0
            print(f"[d2-export] clip {index} shifts {todo} cov "
                  f"{[round(sum(out[d]['coverage']) / FRAMES, 3) for d in todo]} ({last:.0f}s)", flush=True)
        print("D2_EXPORT_TASK_DONE", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
