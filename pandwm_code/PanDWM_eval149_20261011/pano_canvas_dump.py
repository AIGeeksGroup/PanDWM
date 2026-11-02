"""Panorama canvases and camera sampling grids of selected Dev-149 clips (2026-10-19, six-camera figure).

For each clip: the recorded canvas (hard seam, blend 0) and the decoded generated canvas (blend --blend, paper 128) at
the requested clip frames, uint8 [F, 3, H, W] with W = 4114, plus the bridge's six-camera sampling grids at --render
(the same grids front_band_dump.py and the OmniNWM adapter use) and the canonical 9600-column pano coordinate of every
canvas column. A camera image of the same rig can then be mapped back into the canvas by inverting its grid.
  python3 pano_canvas_dump.py --base-config cd_eval_frozen149_hpc3.json --manifest dev149_manifest_20261018.json \
      --dataset-stride 20 --arm-root <run> --out <dir> --positions 31,56,80,140 --frames 25
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import dwm.common
import dwm.functional
from dwm_pano.datasets.pano_projection import VORONOI_BOUNDARIES
from dwm_pano.nuscenes6_bridge import CAM_ORDER, PANO_CANVAS_OVERLAP_COLUMNS, build_rig_sampling_grids, canvas_source_axes
from dwm_pano.pano_stitch import stitch_pano_halves

FRAMES, CONTEXT = 35, 3


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset-stride", type=float, required=True)
    ap.add_argument("--arm-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--frames", default="25")
    ap.add_argument("--blend", type=int, default=128)
    ap.add_argument("--render", default="448,800")
    a = ap.parse_args()
    keep = {int(p) for p in a.positions.split(",")}
    frames = [int(f) for f in a.frames.split(",")]
    render = tuple(int(v) for v in a.render.split(","))
    manifest = json.load(open(a.manifest))
    clips = [c for c in manifest["clips"] if int(c["test_index"]) in keep]
    config = json.loads(Path(a.base_config).read_text())
    config["validation_dataset"]["base_dataset"]["fps_stride_tuples"] = [[10, a.dataset_stride]]
    device = torch.device("cuda", 0)
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

    def decode(latents):
        t, v = latents.shape[:2]
        image = dwm.functional.memory_efficient_split_call(
            vae, latents.flatten(0, 1).to(device, dtype=vae.dtype),
            lambda block, x: block.decode(x / block.config.scaling_factor + shift, return_dict=False)[0], chunk)
        return processor.postprocess(image, output_type="pt").float().unflatten(0, (t, v))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for row in clips:
        pos, anchor = int(row["test_index"]), int(row["anchor_timestamp_us"])
        batch = collate([dataset[int(row["dataset_index"])]])
        if int(batch["anchor_timestamp_us"][0]) != anchor:
            raise RuntimeError(f"clip {pos}: dataset clip is not the manifest clip")
        d = Path(a.arm_root) / f"clip{pos:03d}"
        summary = json.load(open(d / "summary.json"))
        if int(summary["anchor_timestamp_us"]) != anchor:
            raise RuntimeError(f"{d}: not the manifest clip")
        latents = torch.load(d / "latents.pt", map_location="cpu", weights_only=True)["latents"]
        real = stitch_pano_halves(batch["vae_images"][:, frames].to(device), blend_width=0)[0, :, 0].clamp(0, 1)
        fake = stitch_pano_halves(decode(latents[frames])[None], blend_width=a.blend)[0, :, 0].clamp(0, 1)
        shape = tuple(int(v) for v in real.shape[-2:])
        half = (shape[0], (shape[1] + PANO_CANVAS_OVERLAP_COLUMNS) // 2)
        axes = canvas_source_axes(half_size=half)
        grids, valid, reports = build_rig_sampling_grids(
            batch["pano_camera_transforms"][0, 0].cpu().numpy(), batch["pano_camera_intrinsics"][0, 0].cpu().numpy(),
            batch["pano_camera_image_size"][0, 0].cpu().numpy(), shape, render, band_limit=True, source_axes=axes)
        q = lambda x: (x * 255).round().to(torch.uint8).cpu().numpy()
        np.savez_compressed(out / f"clip{pos:03d}.npz", real=q(real), fake=q(fake), frames=np.array(frames),
                            grids=np.asarray(grids, np.float32), valid=np.asarray(valid), cams=np.array(CAM_ORDER),
                            canvas_rows=np.asarray(axes[0]), canvas_cols=np.asarray(axes[1]),
                            voronoi_boundaries=np.array(VORONOI_BOUNDARIES), anchor=anchor, seed=summary["seed"],
                            checkpoint=summary["checkpoint"], blend_fake=a.blend)
        print(f"[canvas] clip{pos:03d} {anchor} canvas {shape} grids {np.asarray(grids).shape}", flush=True)
    print("CANVAS_DUMP_DONE", flush=True)


if __name__ == "__main__":
    main()
