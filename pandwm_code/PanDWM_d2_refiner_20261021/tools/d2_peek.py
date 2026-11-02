"""Quick look at D2 outputs (2026-10-21): one PNG per (clip, dy, frame) with rows
generated video at 0 m (review clip latents.pt) | GS render at dy (VAE decode) | each refined arm at dy.

    python tools/d2_peek.py --vae <sd3.5 dir> --clips-root <eval149 v84i4250> --inputs <review_inputs/tag>
        --arm A=<refined/x> --arm B=<refined/y> --clips 0,50 --dy 2 --frames 10,24 --out <dir>
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from dwm_pano.pano_stitch import stitch_pano_halves


@torch.no_grad()
def decode(vae, latents):
    """(T, 2, C, h, w) -> (T, 3, H, 4114) in [0, 1]."""
    shift = vae.config.shift_factor or 0.0
    x = latents.flatten(0, 1).to("cuda", dtype=vae.dtype)
    images = torch.cat([vae.decode(p / vae.config.scaling_factor + shift, return_dict=False)[0]
                        for p in x.split(4)]).float()
    images = ((images + 1) / 2).clamp(0, 1).unflatten(0, (latents.shape[0], 2))
    return stitch_pano_halves(images[None])[0, :, 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", required=True)
    p.add_argument("--clips-root", required=True, type=Path)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--arm", action="append", default=[])
    p.add_argument("--clips", required=True)
    p.add_argument("--dy", default="2")
    p.add_argument("--frames", default="10,24")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae", torch_dtype=torch.float16).to("cuda").eval()
    frames = [int(f) for f in args.frames.split(",")]
    arms = [a.split("=", 1) for a in args.arm]
    args.out.mkdir(parents=True, exist_ok=True)
    for clip in [int(c) for c in args.clips.split(",")]:
        gen = torch.load(args.clips_root / f"clip{clip:03d}" / "latents.pt", map_location="cpu")["latents"]
        for dy in [float(d) for d in args.dy.split(",")]:
            rows = [decode(vae, gen[frames].float())]
            pack = torch.load(args.inputs / f"clip{clip:03d}" / f"render_dy{dy:g}.pt", map_location="cpu")
            rows.append(decode(vae, pack["render"][frames].float()))
            for _, root in arms:
                rows.append(decode(vae, torch.load(Path(root) / f"clip{clip:03d}" / f"d2_dy{dy:g}.pt",
                                                   map_location="cpu")["latents"][frames].float()))
            for i, f in enumerate(frames):
                image = torch.cat([r[i] for r in rows], dim=-2)
                image = torch.nn.functional.interpolate(image[None], scale_factor=0.5, mode="area")[0]
                Image.fromarray((image.permute(1, 2, 0).cpu() * 255).round().byte().numpy()).save(
                    args.out / f"clip{clip:03d}_dy{dy:g}_f{f:02d}.png")
            print(f"[peek] clip {clip} dy {dy:g} rows: gen0 | render | {' | '.join(a for a, _ in arms)}", flush=True)


if __name__ == "__main__":
    main()
