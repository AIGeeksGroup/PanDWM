"""Re-decode an evaluated arm with the original and a fine-tuned SD3.5 decoder, plus the GT round trip (2026-10-12).

For each of the frozen 149 clips, in one pass:
  <out>/<tag>_redec : the arm's saved latents.pt frames 3..34 decoded with the ORIGINAL decoder, through the same
                      decode path as vae_roundtrip149.py and compute_eval149_metrics.py: the like-for-like baseline for
                      realism_arms.py and flicker_diag.py;
  <out>/<tag>_dft   : the same latents decoded with the fine-tuned decoder (--decoder);
  <out>/vaert_dft   : GT frames 3..34 encoded (posterior mean) and decoded with the fine-tuned decoder (the new ceiling).
Every clip folder gets gt_vs_generated.mp4 (GT top, decoded bottom, stitched with PANDWM_PANO_BLEND_W, libx264 at
10 fps, 16-frame windows) and summary.json (PSNR per window).  The two latent arms also get a latents.pt symlink to
the source clip, so compute_eval149_metrics.py (through decoder_override_launch.py for the _dft arm) can score them.
  python decode149.py --base-config cd_eval_frozen149_hpc3.json --arm-root <eval>/v83h2250 --tag v83h2250 \
      --decoder <decoder_6000.pt> --out <dir> [--start 0 --end 149] [--device cpu --debug-frames 2]
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

import dwm.common
import dwm.functional
import dwm.utils.preview
from dwm_pano.pano_stitch import stitch_pano_halves
from run_history_probe import _psnr_by_frame

CONTEXT, FRAMES, WINDOW = 3, 35, 16


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True)
    p.add_argument("--arm-root", required=True, type=Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--decoder", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=149)
    p.add_argument("--device", default="cuda")
    p.add_argument("--debug-frames", type=int, default=0, help="smoke test: decode only this many frames per clip")
    args = p.parse_args()
    config = json.loads(Path(args.base_config).read_text())
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)

    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    pc = config["pipeline"]
    vae0 = AutoencoderKL.from_pretrained(pc["pretrained_model_name_or_path"], subfolder="vae").to(device).eval()
    vae1 = AutoencoderKL.from_pretrained(pc["pretrained_model_name_or_path"], subfolder="vae")
    vae1.decoder.load_state_dict(torch.load(args.decoder, map_location="cpu"))
    vae1 = vae1.to(device).eval()
    processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae0.config.block_out_channels) - 1))
    shift = vae0.config.shift_factor or 0
    chunk = pc.get("common_config", {}).get("memory_efficient_batch", -1)

    def encode(rgb):  # (N, 3, H, W) in [0, 1] -> scaled latent
        image = processor.preprocess(rgb.to(device)).to(dtype=vae0.dtype)
        return dwm.functional.memory_efficient_split_call(
            vae0, image,
            lambda block, x: (block.encode(x).latent_dist.mode() - shift) * block.config.scaling_factor, chunk)

    def decode(vae, latent):  # scaled latent (N, C, h, w) -> (N, 3, H, W) in [0, 1]
        out = dwm.functional.memory_efficient_split_call(
            vae, latent.to(device, dtype=vae.dtype),
            lambda block, x: block.decode(x / block.config.scaling_factor + shift, return_dict=False)[0], chunk)
        return processor.postprocess(out, output_type="pt").float()

    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    if len(dataset) != 149:
        raise RuntimeError(f"protocol drift: {len(dataset)} clips, expected 149")
    collate = dwm.common.instantiate_config(config["validation_dataloader"])["collate_fn"]
    names = (f"{args.tag}_redec", f"{args.tag}_dft", "vaert_dft")
    last = FRAMES if args.debug_frames <= 0 else CONTEXT + args.debug_frames
    for index in range(args.start, min(args.end, 149)):
        outs = {name: args.out / name / f"clip{index:03d}" for name in names}
        if all((o / "summary.json").is_file() for o in outs.values()):
            continue
        t0 = time.monotonic()
        batch = collate([dataset[index]])
        rgb = batch["vae_images"][:, CONTEXT:last]                      # (1, T, V, 3, H, W) in [0, 1]
        b, t, v = rgb.shape[:3]
        src = (args.arm_root / f"clip{index:03d}" / "latents.pt").resolve()
        lat = torch.load(src, map_location="cpu")["latents"][CONTEXT:last]   # (T, V, C, h, w), fp16
        if tuple(lat.shape[:2]) != (t, v):
            raise RuntimeError(f"clip {index}: latents {tuple(lat.shape)} do not match frames {(t, v)}")
        flat = lat.flatten(0, 1)
        decoded = {names[0]: decode(vae0, flat), names[1]: decode(vae1, flat),
                   names[2]: decode(vae1, encode(rgb.flatten(0, 2)))}
        for name, rec in decoded.items():
            rec = rec.unflatten(0, (b, t, v))
            out = outs[name]
            out.mkdir(parents=True, exist_ok=True)
            frames, psnr_windows = [], []
            for start in range(0, t, WINDOW):
                sl = slice(start, min(start + WINDOW, t))
                fake = stitch_pano_halves(rec[:, sl].to(device)).clamp(0, 1)
                real = stitch_pano_halves(rgb[:, sl].to(device)).clamp(0, 1)
                psnr = _psnr_by_frame(fake, real)
                psnr_windows.append(round(sum(psnr) / len(psnr), 4))
                pair = torch.cat([real[0, :, 0], fake[0, :, 0]], dim=-2)
                frames.extend(pair.mul(255).round().byte().cpu().unbind(0))
            dwm.utils.preview.save_tensor_to_video(str(out / "gt_vs_generated.mp4"), "libx264", 10, frames)
            if name != "vaert_dft" and not (out / "latents.pt").exists():
                os.symlink(src, out / "latents.pt")
            (out / "summary.json").write_text(json.dumps({
                "protocol": "re-decoded for the decoder fine-tuning test (decode149.py)",
                "clip_index": index, "source": str(src) if name != "vaert_dft" else "GT round trip",
                "decoder": "original" if name.endswith("_redec") else str(args.decoder),
                "psnr_mean_db_per_window": psnr_windows}, indent=2) + "\n")
        print(f"[decode149] clip {index} t={time.monotonic() - t0:.1f}s", flush=True)
    print("DECODE149_DONE", flush=True)


if __name__ == "__main__":
    main()
