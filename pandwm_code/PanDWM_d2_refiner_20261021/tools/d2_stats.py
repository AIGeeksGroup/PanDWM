"""Image statistics of D2 arms (2026-10-21, report section 43.3 diagnosis).

For each clip and dy, frames 3, 7, ..., 31 of: the generated video at 0 m (review clip latents.pt; the look the
refiner should reach), the GS render input and every refined arm are VAE-decoded and stitched; per arm we report
mean brightness, contrast (per-image std of luma), saturation (HSV S on pixels with V > 0.25), sharpness (mean
|Laplacian| of luma) and the 4 x 4-pooled high-frequency energy ratio.

    python tools/d2_stats.py --vae <sd3.5> --clips-root <eval149 v84i4250> --inputs <review_inputs/tag>
        --arm A=<refined dir> ... --clips <list> --dy 2,-2 --out <json>
"""

import argparse
import json
from pathlib import Path

import torch

from dwm_pano.pano_stitch import stitch_pano_halves

FRAMES = list(range(3, 35, 4))


@torch.no_grad()
def decode(vae, latents):
    shift = vae.config.shift_factor or 0.0
    x = latents.flatten(0, 1).to("cuda", dtype=vae.dtype)
    images = torch.cat([vae.decode(p / vae.config.scaling_factor + shift, return_dict=False)[0]
                        for p in x.split(4)]).float()
    images = ((images + 1) / 2).clamp(0, 1).unflatten(0, (latents.shape[0], 2))
    return stitch_pano_halves(images[None])[0, :, 0]                                   # (T, 3, H, W)


def stats(x):
    luma = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    mx, mn = x.max(1).values, x.min(1).values
    sat = (mx - mn) / mx.clamp_min(1e-6)
    lit = mx > 0.25
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=x.device, dtype=x.dtype)[None, None]
    lap = torch.nn.functional.conv2d(luma[:, None], k).abs().mean()
    low = torch.nn.functional.interpolate(torch.nn.functional.avg_pool2d(luma[:, None], 4), scale_factor=4,
                                          mode="nearest")[..., :luma.shape[-2], :luma.shape[-1]]
    hf = (luma[:, None][..., :low.shape[-2], :low.shape[-1]] - low).square().mean() / luma.var().clamp_min(1e-8)
    return {"brightness": float(luma.mean()), "contrast": float(luma.flatten(1).std(1).mean()),
            "saturation": float(sat[lit].mean()), "sharpness": float(lap), "hf_ratio": float(hf)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", required=True)
    p.add_argument("--clips-root", required=True, type=Path)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--arm", action="append", default=[])
    p.add_argument("--clips", required=True)
    p.add_argument("--dy", default="2,-2")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae", torch_dtype=torch.float16).to("cuda").eval()
    arms = [a.split("=", 1) for a in args.arm]
    acc = {}
    def add(name, s):
        for k, v in s.items():
            acc.setdefault(name, {}).setdefault(k, []).append(v)
    for clip in [int(c) for c in args.clips.split(",")]:
        gen = torch.load(args.clips_root / f"clip{clip:03d}" / "latents.pt", map_location="cpu")["latents"]
        add("gen_0m", stats(decode(vae, gen[FRAMES].float())))
        for dy in [float(d) for d in args.dy.split(",")]:
            pack = torch.load(args.inputs / f"clip{clip:03d}" / f"render_dy{dy:g}.pt", map_location="cpu")
            add("in", stats(decode(vae, pack["render"][FRAMES].float())))
            for name, root in arms:
                lat = torch.load(Path(root) / f"clip{clip:03d}" / f"d2_dy{dy:g}.pt", map_location="cpu")["latents"]
                add(name, stats(decode(vae, lat[FRAMES].float())))
        print(f"[stats] clip {clip}", flush=True)
    out = {name: {k: sum(v) / len(v) for k, v in d.items()} for name, d in acc.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    for name, d in out.items():
        print(name, {k: round(v, 4) for k, v in d.items()}, flush=True)


if __name__ == "__main__":
    main()
