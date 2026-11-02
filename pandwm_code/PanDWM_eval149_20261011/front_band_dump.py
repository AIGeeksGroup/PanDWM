"""CAM_FRONT band dumps of 35-frame clip runs in the September common-protocol format (2026-10-19, comparison figure).

The September front-view comparison (HorizonDrive / Epona / Vista / GEM, docs/reports/2026-09-10-*) scored every method
on a 384x768 uint8 CAM_FRONT view resampled from the panorama with the real calibration
(NuScenes6BandFrechetVideoDistance.update_from_canvas, quantize_uint8, PANDWM_COMMON_VIS_DIR). This writes the same
interchange for a clip run of run_eval149_videos.py / run_clips_videos.py:
  real/<anchor>.pt : recorded frames 0..34, hard-seam canvas (blend 0), as the September canonical GT;
  fake/<anchor>.pt : the clip's latents.pt decoded with the configured VAE, stitched with --blend (paper: 128).
Both are uint8 [35, 3, 384, 768]; frame index = clip frame index (frames 0..2 of fake are the decoded history).
2026-10-19: --cams (several real cameras -> [T, V, 3, H, W] in that order), --render H,W and --frames (keep only these
clip frames, in order) for the multi-view figure; the defaults reproduce the single CAM_FRONT 384x768 dump exactly.
  python3 front_band_dump.py --base-config cd_eval_frozen149_hpc3.json --manifest dev149_manifest_20261018.json \
      --dataset-stride 20 --arm-root <dir with clipNNN/> --out <dir> [--positions 31,64,148] [--blend 128]
"""
import argparse
import json
import os
from pathlib import Path

import torch

import dwm.common
import dwm.functional
from dwm_pano.metrics.fvd_pano import PanoPatchFrechetVideoDistance
from dwm_pano.nuscenes6_bridge import (
    CAM_ORDER, PANO_CANVAS_OVERLAP_COLUMNS, apply_camera_sampling_grid, build_rig_sampling_grids, canvas_source_axes)
from dwm_pano.pano_stitch import stitch_pano_halves

FRAMES, CONTEXT = 35, 3


def camera_views(canvas, batch, cache, cams, render):
    """canvas [T, C, H, W] in [0, 1] -> uint8 [T, C, H, W] per camera in cams, as update_from_canvas with start 0."""
    shape = tuple(int(v) for v in canvas.shape[-2:])
    key = (shape, render,
           tuple(torch.round(batch["pano_camera_transforms"][0, 0].double() * 1e4).flatten().tolist()),
           tuple(torch.round(batch["pano_camera_intrinsics"][0, 0].double() * 1e3).flatten().tolist()))
    if key not in cache:
        axes = canvas_source_axes(half_size=(shape[0], (shape[1] + PANO_CANVAS_OVERLAP_COLUMNS) // 2))
        grids, valid, _ = build_rig_sampling_grids(
            batch["pano_camera_transforms"][0, 0].cpu().numpy(), batch["pano_camera_intrinsics"][0, 0].cpu().numpy(),
            batch["pano_camera_image_size"][0, 0].cpu().numpy(), shape, render, band_limit=True, source_axes=axes)
        cache[key] = (grids, valid)
    grids, valid = cache[key]
    views = []
    for cam in cams:
        i = CAM_ORDER.index(cam)
        view = apply_camera_sampling_grid(canvas, grids[i], valid[i])
        views.append((view.clamp(0, 1) * 255).round().to(torch.uint8).cpu())
    return views[0] if len(views) == 1 else torch.stack(views, 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset-stride", type=float, required=True)
    ap.add_argument("--arm-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", default="")
    ap.add_argument("--blend", type=int, default=128)
    ap.add_argument("--cams", default="CAM_FRONT")
    ap.add_argument("--render", default="384,768")
    ap.add_argument("--frames", default="")
    a = ap.parse_args()
    cams = a.cams.split(",")
    render = tuple(int(v) for v in a.render.split(","))
    keep_frames = [int(f) for f in a.frames.split(",")] if a.frames else list(range(FRAMES))
    if any(c not in CAM_ORDER for c in cams) or any(not 0 <= f < FRAMES for f in keep_frames):
        raise ValueError(f"bad --cams {cams} or --frames {keep_frames}")
    if tuple(PanoPatchFrechetVideoDistance.CAM_NAMES) != tuple(CAM_ORDER):
        raise RuntimeError("bridge CAM_ORDER differs from the metric CAM_NAMES")
    manifest = json.load(open(a.manifest))
    clips = manifest["clips"]
    if a.positions:
        keep = {int(p) for p in a.positions.split(",")}
        clips = [c for c in clips if int(c["test_index"]) in keep]
    config = json.loads(Path(a.base_config).read_text())
    ds = config["validation_dataset"]["base_dataset"]
    if ds["sequence_length"] != FRAMES:
        raise RuntimeError("expected the frozen 35-frame protocol config")
    ds["fps_stride_tuples"] = [[10, a.dataset_stride]]
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    if len(dataset) != manifest["dataset_len"]:
        raise RuntimeError(f"protocol drift: {len(dataset)} clips, manifest says {manifest['dataset_len']}")
    collate = dwm.common.instantiate_config(config["validation_dataloader"])["collate_fn"]
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    pc = config["pipeline"]
    vae = AutoencoderKL.from_pretrained(pc["pretrained_model_name_or_path"], subfolder="vae").to(device).eval()
    processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1))
    shift = vae.config.shift_factor or 0
    chunk = pc.get("common_config", {}).get("memory_efficient_batch", -1)

    def decode(latents):  # (T, V, C, h, w) -> (T, V, 3, H, W) in [0, 1], as clip_feats.py
        t, v = latents.shape[:2]
        image = dwm.functional.memory_efficient_split_call(
            vae, latents.flatten(0, 1).to(device, dtype=vae.dtype),
            lambda block, x: block.decode(x / block.config.scaling_factor + shift, return_dict=False)[0], chunk)
        return processor.postprocess(image, output_type="pt").float().unflatten(0, (t, v))

    out = Path(a.out)
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "fake").mkdir(parents=True, exist_ok=True)
    cache, index = {}, []
    for k, row in enumerate(clips):
        pos, anchor = int(row["test_index"]), int(row["anchor_timestamp_us"])
        batch = collate([dataset[int(row["dataset_index"])]])
        if int(batch["anchor_timestamp_us"][0]) != anchor:
            raise RuntimeError(f"clip {pos}: dataset clip is not the manifest clip")
        d = Path(a.arm_root) / f"clip{pos:03d}"
        summary = json.load(open(d / "summary.json"))
        if summary["scene_token"] != row["scene_token"] or int(summary["anchor_timestamp_us"]) != anchor:
            raise RuntimeError(f"{d}: not the manifest clip")
        payload = torch.load(d / "latents.pt", map_location="cpu", weights_only=True)
        latents = payload["latents"]
        if latents.shape[0] != FRAMES or payload.get("context_frames") != CONTEXT:
            raise ValueError(f"{d}: latents {tuple(latents.shape)}")
        real = stitch_pano_halves(batch["vae_images"].to(device), blend_width=0)[0, keep_frames, 0].clamp(0, 1)
        fake = stitch_pano_halves(decode(latents[keep_frames])[None], blend_width=a.blend)[0, :, 0].clamp(0, 1)
        for role, canvas in (("real", real), ("fake", fake)):
            target = out / role / f"{anchor}.pt"
            torch.save(camera_views(canvas, batch, cache, cams, render), str(target) + ".tmp")
            os.replace(str(target) + ".tmp", target)
        index.append({"test_index": pos, "anchor_timestamp_us": anchor, "scene_token": row["scene_token"],
                      "seed": summary["seed"], "checkpoint": summary["checkpoint"]})
        print(f"[dump] {k + 1}/{len(clips)} clip{pos:03d} {anchor}", flush=True)
    meta = {"arm_root": a.arm_root, "blend_fake": a.blend, "blend_real": 0, "render": render, "cams": cams,
            "frames": keep_frames, "context_frames": CONTEXT, "clips": index}
    (out / "index.json").write_text(json.dumps(meta, indent=1))
    print("FRONT_DUMP_DONE", flush=True)


if __name__ == "__main__":
    main()
