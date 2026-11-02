"""Training pairs for the D2 refiner (2026-10-21, PanDWM-d2-refiner-20261021/D2_PLAN_20261021.md Tasks 5 / 7).

Train-split clips of 35 frames at 10 fps (one clip start every --stride seconds, the clip enumeration of the
video model's PanoNuScenesDataset on the same pkl) are visited in a fixed hash order; array task k of n takes
positions k, k + n, ... of that order, so any prefix of the work is a near-uniform sample of the train scenes.
Per clip: the GT halves are VAE-encoded (posterior mean) and decoded into a gs_scene_v3 scene with the v4 far-view
settings (C2 decoder + DA3 scale relabel + drop the 2 bottom decoder rows; the scene canvases are the GT halves
themselves), one degradation (mode, dx, dy) = d2_render.draw_degradation(seed, index) is applied to all 35
frames, and the render condition is written as

    <out-root>/clip<NNNNN>.pt    {"render": (35, 2, 16, 32, 288) fp16, "mask": (35, 2, 32, 288) uint8, "meta"}
    <out-root>/clip<NNNNN>.json  meta (scene_token, anchor_timestamp_us, frame_timestamps_us, mode, dx, dy, ...)

The .pt is written first (tmp + rename), then the .json; dwm_pano_d2.data.D2RenderDataset only uses clips that
have both.  Existing clips are skipped, so tasks can be resubmitted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import dwm.common
from dwm_pano.d2_render import degraded_canvas, draw_degradation, encode_condition, encode_halves_mean, unit_hash
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.export_gs_batch149 import FRAMES
from dwm_pano.gs_scene_v3 import SceneBuilder, SceneOptions

VERSION = "d2_pairs_v1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path, help="GS config whose training_dataset reads train")
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path, help="split_195_scene_manifest.json (dev / heldout)")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--tasks", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="only the first N clips of the hash order (0 = all)")
    p.add_argument("--stride", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--time-budget-s", type=float, default=0.0)
    p.add_argument("--preview-dir", type=Path, default=None)
    p.add_argument("--previews", type=int, default=2, help="clips per task with a frame-10 preview image")
    p.add_argument("--count-only", action="store_true")
    p.add_argument("--skip-file", type=Path, default=None,
                   help="json list of clip indices already made elsewhere (another cluster), not redone here")
    p.add_argument("--da3-src", required=True)
    p.add_argument("--da3-anyview", required=True)
    p.add_argument("--da3-metric", required=True)
    return p.parse_args()


def save_preview(path, gt, canvas, alpha):
    from PIL import Image
    rows = [gt, canvas, alpha.float()[None].expand(3, -1, -1)]
    image = torch.cat([r.float().clamp(0, 1).cpu() for r in rows], dim=-2)
    image = torch.nn.functional.interpolate(image[None], scale_factor=0.5, mode="area")[0]
    Image.fromarray((image.permute(1, 2, 0) * 255).round().byte().numpy()).save(path)


@torch.no_grad()
def main():
    args = parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    manifest = json.loads(args.manifest.read_text())
    forbidden = {item["scene_token"] for key in ("dev", "heldout") for item in manifest[key]}
    opts = SceneOptions(da3_relabel=True, da3_mode="scale", drop_bottom_rows=2)
    scene_desc = {k: getattr(opts, k) for k in ("edge_rel_h", "edge_rel_v", "window", "da3_relabel", "da3_mode",
                                                 "drop_bottom_rows", "static_range", "max_scale")}
    started = ensure_single_rank_process_group(device)
    try:
        config = load_config(args.config)
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl,
                                   fps_stride_tuples=[[10, args.stride]])
        cfg["base_dataset"]["sequence_length"] = FRAMES
        if cfg["base_dataset"].get("split") != "train":
            raise ValueError("the GS config's training_dataset must read the train split")
        install_strict_pano_hardseam_stub()
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        order = sorted(range(len(base)), key=lambda i: unit_hash(args.seed, i, "order"))
        if args.limit:
            order = order[:args.limit]
        mine = order[args.task::args.tasks]
        if args.skip_file is not None:
            skip = set(int(x) for x in json.loads(args.skip_file.read_text()))
            mine = [i for i in mine if i not in skip]
        print(f"[d2-pairs] {len(base)} clips, {len(order)} in the order, task {args.task}/{args.tasks}: "
              f"{len(mine)}", flush=True)
        if args.count_only:
            return
        from dwm_pano_4d.da3_depth import Da3Ranger
        da3 = Da3Ranger(args.da3_src, args.da3_anyview, args.da3_metric, device)
        torch.manual_seed(int(config["generator_seed"]))
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(args.out_root),
                                                      config=config, device=device, resume_from=None)
        pipe.vae.eval()
        pipe.decoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
        pipe.decoder.eval()
        pipe.decoder_wrapper.eval()
        pipe.rasterize_mode = "antialiased"
        pipe.active_rasterize_eps2d = 0.1
        from dwm.datasets.common import CollateFnIgnoring
        args.out_root.mkdir(parents=True, exist_ok=True)
        job_start, last, done, previews = time.monotonic(), 0.0, 0, 0
        for index in mine:
            out = args.out_root / f"clip{index:05d}.pt"
            meta_path = out.with_suffix(".json")
            if meta_path.is_file() and out.is_file():
                continue
            if args.time_budget_s and last and time.monotonic() - job_start + 1.3 * last > args.time_budget_s:
                print(f"[d2-pairs] budget reached after {done} clips", flush=True)
                break
            audit = base.clip_audit[index]
            if audit["scene_token"] in forbidden:
                print(f"[d2-pairs] clip {index} is dev/heldout, skipped", flush=True)
                continue
            t0 = time.monotonic()
            mode, dx, dy = draw_degradation(args.seed, index)
            batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [index]), batch_size=1,
                                                          shuffle=False, num_workers=0,
                                                          collate_fn=CollateFnIgnoring(keys=[]))))
            if batch["split"][0] != "train":
                raise ValueError(f"clip {index} is not a train clip")
            halves = batch["vae_images"][0].to(device).float()                           # (35, 2, 3, H, W)
            latents = encode_halves_mean(pipe, halves).cpu()                             # (35, 2, 16, h, w)
            scene = SceneBuilder(pipe, batch, base.clips[index], latents, opts, device, da3=da3)
            for t in range(FRAMES):
                scene._canvas[t] = pipe._unpadded_stitch_canvas(halves[t:t + 1])[0].clamp(0, 1)
            renders, masks, coverage = [], [], []
            for t in range(FRAMES):
                rgb, alpha = degraded_canvas(pipe, scene, t, mode, (dx, dy))
                lat, mask, canvas = encode_condition(pipe, rgb, alpha)
                renders.append(lat.cpu())
                masks.append(mask.cpu())
                coverage.append(float((alpha[pipe.stitch_valid] >= 0.5).float().mean()))
                if t == 10 and args.preview_dir is not None and previews < args.previews:
                    args.preview_dir.mkdir(parents=True, exist_ok=True)
                    save_preview(args.preview_dir / f"clip{index:05d}_{mode}.png", scene.canvas(t), canvas, alpha)
                    previews += 1
            meta = {"version": VERSION, "gs_index": int(index), "scene_token": audit["scene_token"],
                    "anchor_timestamp_us": int(base.clips[index][0]["timestamp"]),
                    "frame_timestamps_us": [int(x) for x in batch["frame_timestamps_us"][0].tolist()],
                    "mode": mode, "dx": dx, "dy": dy, "coverage": coverage,
                    "coverage_mean": sum(coverage) / len(coverage), "seed": args.seed, "stride_s": args.stride,
                    "checkpoint": str(args.checkpoint), "scene": scene_desc, "scene_stats": scene.summary()}
            if meta["frame_timestamps_us"][0] != meta["anchor_timestamp_us"]:
                raise RuntimeError(f"clip {index}: first frame timestamp is not the anchor")
            tmp = out.with_suffix(".tmp")
            torch.save({"render": torch.stack(renders), "mask": torch.stack(masks), "meta": meta}, tmp)
            tmp.replace(out)
            meta_path.write_text(json.dumps(meta) + "\n")
            done += 1
            last = time.monotonic() - t0
            print(f"[d2-pairs] clip {index} {mode} dx={dx:+.2f} dy={dy:+.2f} cov={meta['coverage_mean']:.3f} "
                  f"({last:.0f}s)", flush=True)
        print(f"D2_PAIRS_TASK_DONE task={args.task} wrote={done}", flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
