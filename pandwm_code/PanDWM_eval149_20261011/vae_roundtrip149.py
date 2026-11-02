"""VAE round-trip floor on the frozen 149 clips (2026-10-11).

The GT frames 3..34 of each clip are encoded with the configured SD3.5 VAE (posterior mean, as the evaluation encodes
history) and decoded again, then written exactly as run_eval149_videos.py writes a generated clip (GT top, round trip
bottom, stitched with PANDWM_PANO_BLEND_W, libx264 at 10 fps; PSNR per window in summary.json).  realism_arms.py and
flicker_diag.py can then treat it as one more arm ("vaert"): any extra frame-to-frame change, softening or colour shift
it shows is paid by every latent model before its own errors.
  python vae_roundtrip149.py --base-config cd_eval_frozen149_hpc3.json --out-root <run root>/vaert [--start --end]
"""
import argparse
import json
import time
from pathlib import Path

import torch

import dwm.common
import dwm.functional
import dwm.utils.preview
from dwm_pano.pano_stitch import stitch_pano_halves
from run_history_probe import _psnr_by_frame


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=149)
    args = p.parse_args()
    config = json.loads(Path(args.base_config).read_text())
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)

    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    pc = config["pipeline"]
    vae = AutoencoderKL.from_pretrained(pc["pretrained_model_name_or_path"], subfolder="vae").to(device).eval()
    processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1))
    shift = vae.config.shift_factor or 0
    chunk = pc.get("common_config", {}).get("memory_efficient_batch", -1)

    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    if len(dataset) != 149:
        raise RuntimeError(f"protocol drift: {len(dataset)} clips, expected 149")
    collate = dwm.common.instantiate_config(config["validation_dataloader"])["collate_fn"]
    args.out_root.mkdir(parents=True, exist_ok=True)
    for index in range(args.start, min(args.end, 149)):
        out = args.out_root / f"clip{index:03d}"
        if (out / "summary.json").is_file():
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        batch = collate([dataset[index]])
        rgb = batch["vae_images"][:, 3:35]                       # (1, 32, V=2, 3, H, W) in [0, 1]
        b, t, v = rgb.shape[:3]
        image = processor.preprocess(rgb.flatten(0, 2).to(device)).to(dtype=vae.dtype)
        latent = dwm.functional.memory_efficient_split_call(
            vae, image,
            lambda block, x: (block.encode(x).latent_dist.mode() - shift) * block.config.scaling_factor, chunk)
        decoded = dwm.functional.memory_efficient_split_call(
            vae, latent,
            lambda block, x: block.decode(x / block.config.scaling_factor + shift, return_dict=False)[0], chunk)
        recon = processor.postprocess(decoded, output_type="pt").float().unflatten(0, (b, t, v))
        frames, psnr_windows = [], []
        for w in range(2):
            sl = slice(16 * w, 16 * w + 16)
            fake = stitch_pano_halves(recon[:, sl].to(device)).clamp(0, 1)
            real = stitch_pano_halves(rgb[:, sl].to(device)).clamp(0, 1)
            psnr = _psnr_by_frame(fake, real)
            psnr_windows.append(round(sum(psnr) / len(psnr), 4))
            pair = torch.cat([real[0, :, 0], fake[0, :, 0]], dim=-2)
            frames.extend(pair.mul(255).round().byte().cpu().unbind(0))
        dwm.utils.preview.save_tensor_to_video(str(out / "gt_vs_generated.mp4"), "libx264", 10, frames)
        (out / "summary.json").write_text(json.dumps({
            "protocol": "VAE round trip of GT frames 3..34 (posterior mean), written like run_eval149_videos.py",
            "clip_index": index, "psnr_mean_db_per_window": psnr_windows,
            "runtime_seconds": round(time.monotonic() - t0, 1)}, indent=2) + "\n")
        print(f"[vaert] clip {index} psnr={psnr_windows} t={time.monotonic() - t0:.1f}s", flush=True)
    print("VAERT_DONE", flush=True)


if __name__ == "__main__":
    main()
