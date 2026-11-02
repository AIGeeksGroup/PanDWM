"""Smoke / diagnostic for da3_depth.py (2026-10-19, report section 40) on one labelled p2 clip.

Runs DA3 on frames (0, gap, 2 gap) of clip --clip (original frames and their VAE round trip), compares both
scales with the depth label on the decoder grid (log error overall and per row band, sky agreement) and writes
<out>/crops.png (the 18 pinhole crops DA3 sees, round trip) and <out>/ranges_f<gap>.png (rows: RGB canvas, label,
DA3 ground scale, decoder (cov), log2(DA3 ground / label)).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import dwm.common
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano_4d.da3_depth import Da3Ranger
from dwm_pano_4d.eval_label_depth import band_bias, load_label, log_stats


def turbo(values, lo=1.0, hi=100.0):
    t = np.clip((np.log(np.clip(values, lo, hi)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)
    return cv2.applyColorMap((255 * (1 - t)).astype(np.uint8), cv2.COLORMAP_TURBO)


def canvas_of(pipe, halves_value):
    """(2, H, W) decoder-grid map -> (H, 4114) canvas layout, like _unpadded_stitch_canvas."""
    return pipe._unpadded_stitch_canvas(halves_value[None, :, None])[0, 0].float().cpu().numpy()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--clips", default="0", help="comma-separated clip indices (eval_label_depth's dataset)")
    p.add_argument("--video-frames", type=int, default=20, help="sequence length, as eval_label_depth")
    p.add_argument("--gap", type=int, default=4)
    p.add_argument("--far-cut", type=float, default=99.0)
    p.add_argument("--da3-src", required=True)
    p.add_argument("--da3-anyview", required=True)
    p.add_argument("--da3-metric", required=True)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    config = load_config(a.config)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    started = ensure_single_rank_process_group(device)
    try:
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(a.out), config=config,
                                                      device=device, resume_from=None)
        pipe.vae.eval()
        pipe.decoder.load_state_dict(torch.load(a.checkpoint, map_location="cpu"), strict=True)
        pipe.decoder.eval()
        install_strict_pano_hardseam_stub()
        dcfg = build_dataset_config(config["training_dataset"], pkl_path=a.data_pkl, fps_stride_tuples=[[10, 20]])
        dcfg["base_dataset"]["sequence_length"] = max(2 * a.gap + 1, a.video_frames)
        dataset = dwm.common.create_instance_from_config(dcfg)
        step = max(1, len(dataset) // 25)
        test = set(list(range(0, len(dataset), step))[:25])
        print(f"[diag-da3] dataset {len(dataset)} clips; eval_label_depth's 25 test clips = every {step}th", flush=True)
        from dwm.datasets.common import CollateFnIgnoring
        t0 = time.time()
        da3 = Da3Ranger(a.da3_src, a.da3_anyview, a.da3_metric, device)
        print(f"[diag-da3] loaded {da3.anyview_name} + {da3.metric_name} in {time.time() - t0:.1f}s", flush=True)
        for clip in [int(x) for x in a.clips.split(",")]:
            out_dir = a.out / f"clip{clip}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[diag-da3] clip {clip} ({'IN' if clip in test else 'not in'} the 25 test clips)", flush=True)
            run_clip(a, pipe, da3, CollateFnIgnoring(keys=[])([dataset[clip]]), clip, out_dir, device)
    finally:
        if started:
            torch.distributed.destroy_process_group()


def run_clip(a, pipe, da3, batch, clip, out_dir, device):
    if True:
        pipe._prepare_nuscenes6_supervision(batch)
        t0 = time.time()
        rig = da3.rig(pipe)
        print(f"[diag-da3] rig ready in {time.time() - t0:.1f}s, pad {rig['pad']}, crop "
              f"{da3.crop_h}x{da3.crop_w}, scales {[round(c['s'], 3) for c in rig['crops']]}", flush=True)
        frames = [0, a.gap, 2 * a.gap]
        halves = batch["vae_images"][0, frames].to(device).float()
        rt = torch.stack([pipe.decode_half_latents(pipe.encode_half_latents(h[None]))[0] for h in halves]).float()
        e2w = batch["ego_transforms"][0, frames, 0]
        report = {"clip": clip, "scene_token": batch["scene_token"][0], "frames": frames}
        for tag, imgs in (("orig", halves), ("vae", rt)):
            torch.cuda.synchronize()
            t0 = time.time()
            res = da3.frame_ranges(pipe, pipe._unpadded_stitch_canvas(imgs), e2w)
            torch.cuda.synchronize()
            report[f"{tag}_seconds"] = round(time.time() - t0, 2)
            for name in ("scale_pose", "scale_metric", "scale_ground", "pose_over_metric", "ground_over_metric"):
                report[f"{tag}_{name}"] = res[name]
            content = pipe.pixel_content_ray_valid
            for i, f in enumerate(frames):
                label, valid = load_label(pipe, batch, f, "nearest", device)
                use = valid & (label > 0.5) & (label < a.far_cut) & content
                far = valid & (label >= a.far_cut) & content
                for scale in ("pose", "metric", "ground"):
                    report.update({f"{tag}_f{f}_{scale}_{k}": v for k, v in
                                   log_stats("vs_label", res[f"range_{scale}"][i], label, use).items()})
                for scale in ("pose", "ground"):
                    report.update({f"{tag}_f{f}_{scale}_bias_{k}": v for k, v in
                                   band_bias(res[f"range_{scale}"][i], label, use).items()})
                sky = res["sky"][i]
                report[f"{tag}_f{f}_sky_on_far"] = float((sky[far] >= 0.3).float().mean()) if far.any() else None
                report[f"{tag}_f{f}_sky_on_near"] = float((sky[use] >= 0.3).float().mean())
                if tag == "vae" and f == a.gap:
                    sel = torch.tensor([f])
                    lo, po = pipe._frame_ray_origins(batch, sel)
                    out = pipe.decoder_wrapper(
                        pipe.encode_half_latents(halves[i:i + 1]), pipe.latent_directions.unsqueeze(0),
                        pipe.pixel_directions.unsqueeze(0), None, previous_latents=None, temporal_context_mask=0.0,
                        latent_origins=lo, pixel_origins=po, camera_origin_scale=1.0)
                    rng = out[1][0].float()
                    report.update({f"decoder_f{f}_{k}": v for k, v in log_stats("vs_label", rng, label, use).items()})
                    rgb = (pipe._unpadded_stitch_canvas(imgs[i:i + 1])[0].permute(1, 2, 0).cpu().numpy() * 255)
                    lab_c = canvas_of(pipe, torch.where(valid, label, torch.zeros_like(label)))
                    da3_c = canvas_of(pipe, res["range_ground"][i])
                    dec_c = canvas_of(pipe, rng)
                    ratio = np.log2(np.clip(da3_c, 0.1, None) / np.clip(lab_c, 0.1, None))
                    ratio = np.where(lab_c > 0.5, ratio, 0.0)
                    rows = [cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR), turbo(lab_c), turbo(da3_c),
                            turbo(dec_c),
                            cv2.applyColorMap((np.clip(ratio / 2 + 0.5, 0, 1) * 255).astype(np.uint8),
                                              cv2.COLORMAP_JET)]
                    cv2.imwrite(str(out_dir / f"ranges_f{f}.png"), np.concatenate(rows, axis=0))
                    crops = (res["crops"].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                    grid = np.concatenate([np.concatenate(list(crops[r * 6:(r + 1) * 6]), axis=1)
                                           for r in range(len(frames))], axis=0)
                    cv2.imwrite(str(out_dir / "crops.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        (out_dir / "diag.json").write_text(json.dumps(report, indent=2) + "\n")
        keys = [k for k in report if k.startswith("vae_") and ("_vs_label_log_" in k or "scale" in k
                                                               or "over" in k or "sky" in k)]
        print(json.dumps({k: report[k] for k in keys} | {"decoder": {k: v for k, v in report.items()
                                                                  if k.startswith("decoder")}}, indent=1),
              flush=True)


if __name__ == "__main__":
    main()
