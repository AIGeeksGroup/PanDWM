#!/usr/bin/env python3
"""Seam attribution + circular-padding fix verification.

Adds two fix arms to the v1 attribution experiment:

  B1  padded-decode   latent-space wrap padding before VAE decode, cropped
                      after. Needs NO retraining -- applicable to the existing
                      checkpoint at eval time, because it only touches how a
                      per-half latent is turned back into pixels.
  B2  padded-both     pixel-space wrap padding before encode AND padded decode.
                      The ceiling of this family, but the encode half changes
                      what the model is trained against, so it needs a retrain
                      to be usable end-to-end.

Neighbour geometry (all four half edges have real neighbours):
  left  half covers source [0, 5376)   -> 2304 px
  right half covers source [4224, 9600) -> 2304 px
  left.right_edge  <- right half from px 494  (== 2304 - 1810, the overlap)
  left.left_edge   <- right half's rightmost px (source 9600 == 0, wrap)
  right.left_edge  <- left half up to px 1810 (the overlap)
  right.right_edge <- left half's leftmost px (wrap)

Latent columns: 2304/8 = 288. Overlap 494 px = 61.75 latent cols -> 62 is used
(a 2 px / quarter-column approximation; noted because 494 is not 8-aligned).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

PANO_W, PANO_H = 9600, 600
HALF_LEFT, HALF_RIGHT = (0, 5376), (4224, 9600)
HALF_W, HALF_H = 2304, 256
OVERLAP_W = 494
STITCH_W = 2 * HALF_W - OVERLAP_W          # 4114
SEAM_COL = HALF_W                          # 2304
VAE_DOWN = 8
LAT_W = HALF_W // VAE_DOWN                 # 288
LAT_OVERLAP = round(OVERLAP_W / VAE_DOWN)  # 62
LAT_RIGHT_START = LAT_W - LAT_OVERLAP      # 226: left col aligned to right col 0


def slice_halves(img):
    left = img.crop((HALF_LEFT[0], 0, HALF_LEFT[1], PANO_H))
    right = img.crop((HALF_RIGHT[0], 0, HALF_RIGHT[1], PANO_H))
    return (left.resize((HALF_W, HALF_H), Image.LANCZOS),
            right.resize((HALF_W, HALF_H), Image.LANCZOS))


def stitch(left, right):
    return np.concatenate([left, right[:, OVERLAP_W:]], axis=1)


def measure(pano):
    h, w, _ = pano.shape
    col_diff = np.abs(pano[:, 1:] - pano[:, :-1]).mean(axis=(0, 2))
    out = {}
    if w == STITCH_W:
        i = SEAM_COL - 1
        out["jump_lr"] = float(col_diff[i])
        win = np.concatenate([col_diff[i - 20:i], col_diff[i + 1:i + 21]])
        out["base_lr_local"] = float(win.mean())
        out["ratio_lr"] = out["jump_lr"] / out["base_lr_local"]
    out["jump_wrap"] = float(np.abs(pano[:, -1] - pano[:, 0]).mean())
    win = np.concatenate([col_diff[:20], col_diff[-20:]])
    out["base_wrap_local"] = float(win.mean())
    out["ratio_wrap"] = out["jump_wrap"] / out["base_wrap_local"]
    out["base_global"] = float(col_diff.mean())
    return out


def to_float(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def pad_pixels(la, ra, p):
    """Wrap-pad both halves in pixel space with real neighbour content."""
    left = np.concatenate([ra[:, HALF_W - p:],        # wrap: right half's tail
                           la,
                           ra[:, OVERLAP_W:OVERLAP_W + p]], axis=1)
    right = np.concatenate([la[:, HALF_W - OVERLAP_W - p:HALF_W - OVERLAP_W],
                            ra,
                            la[:, :p]], axis=1)       # wrap: left half's head
    return left, right


def pad_latents(zl, zr, q):
    """Same geometry, in latent space. zl/zr: (C, h, LAT_W) tensors."""
    left = torch.cat([zr[..., LAT_W - q:],
                      zl,
                      zr[..., LAT_OVERLAP:LAT_OVERLAP + q]], dim=-1)
    right = torch.cat([zl[..., LAT_RIGHT_START - q:LAT_RIGHT_START],
                       zr,
                       zl[..., :q]], dim=-1)
    return left, right


def vae_encode(vae, arrs, device):
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
    x = (x * 2 - 1).to(device, dtype=vae.dtype)
    return vae.encode(x).latent_dist.mode()


def vae_decode(vae, z):
    y = vae.decode(z).sample
    return ((y + 1) / 2).clamp(0, 1).float().cpu().permute(0, 2, 3, 1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default="/path/to/pandwm_root/data/val/rgb")
    ap.add_argument("--dump-dir", default="/path/to/pandwm_root/eval_dumps/smoke2")
    ap.add_argument("--vae-path",
                    default="/path/to/pandwm_root/ckpts/stable-diffusion-3.5-medium")
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--pad-latent", type=int, default=8,
                    help="latent columns of wrap padding (8 = 64 px)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    q = args.pad_latent
    p = q * VAE_DOWN

    arms = ("A_pipeline_real", "B_vae_plain", "B1_padded_decode",
            "B2_padded_both", "C_global_resize", "D_dump_fake", "E_dump_real")
    results = {a: [] for a in arms}

    sources = sorted(glob.glob(os.path.join(args.source_dir, "*.jpg")))
    step = max(1, len(sources) // args.count)
    sources = sources[::step][:args.count]
    print("sources: {}   pad: {} latent cols / {} px".format(
        len(sources), q, p), flush=True)

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae")
    vae = vae.to(args.device).eval()

    for n, path in enumerate(sources):
        img = Image.open(path).convert("RGB")
        if img.size != (PANO_W, PANO_H):
            continue
        left, right = slice_halves(img)
        la, ra = to_float(left), to_float(right)

        results["A_pipeline_real"].append(measure(stitch(la, ra)))
        results["C_global_resize"].append(
            measure(to_float(img.resize((STITCH_W, HALF_H), Image.LANCZOS))))

        with torch.no_grad():
            # ---- plain round trip (current behaviour) --------------------
            z = vae_encode(vae, [la, ra], args.device)
            rec = vae_decode(vae, z)
            results["B_vae_plain"].append(measure(stitch(rec[0], rec[1])))

            # ---- B1: plain encode, PADDED decode (no retrain needed) -----
            zl_pad, zr_pad = pad_latents(z[0], z[1], q)
            rec_pad = vae_decode(vae, torch.stack([zl_pad, zr_pad]))
            b1_l = rec_pad[0][:, p:p + HALF_W]
            b1_r = rec_pad[1][:, p:p + HALF_W]
            results["B1_padded_decode"].append(measure(stitch(b1_l, b1_r)))

            # ---- B2: padded encode AND padded decode ---------------------
            pl, pr = pad_pixels(la, ra, p)
            z2 = vae_encode(vae, [pl, pr], args.device)
            rec2 = vae_decode(vae, z2)
            b2_l = rec2[0][:, p:p + HALF_W]
            b2_r = rec2[1][:, p:p + HALF_W]
            results["B2_padded_both"].append(measure(stitch(b2_l, b2_r)))

        if (n + 1) % 4 == 0:
            print("  {}/{}".format(n + 1, len(sources)), flush=True)

    for arm, sub in (("D_dump_fake", "fake"), ("E_dump_real", "real")):
        files = sorted(glob.glob(os.path.join(args.dump_dir, sub, "*.png")))
        step = max(1, len(files) // args.count)
        for path in files[::step][:args.count]:
            arr = to_float(Image.open(path).convert("RGB"))
            if arr.shape[1] == STITCH_W:
                results[arm].append(measure(arr))

    summary = {}
    for arm, rows in results.items():
        if not rows:
            summary[arm] = {"n": 0}
            continue
        summary[arm] = {"n": len(rows)}
        summary[arm].update({k: float(np.mean([r[k] for r in rows]))
                             for k in rows[0]})

    with open(args.out, "w") as f:
        json.dump({"pad_latent_cols": q, "pad_px": p,
                   "summary": summary, "per_image": results}, f, indent=2)

    print("\n=== SEAM: attribution + padding fix (pad={} px) ===".format(p))
    hdr = "{:<19} {:>3} {:>9} {:>9} {:>7} {:>10} {:>10} {:>7}"
    print(hdr.format("arm", "n", "jump_lr", "base_lr", "ratio",
                     "jump_wrap", "base_wrap", "ratio"))
    for arm in ("C_global_resize", "A_pipeline_real", "B_vae_plain",
                "B1_padded_decode", "B2_padded_both",
                "E_dump_real", "D_dump_fake"):
        s = summary[arm]
        if not s.get("n"):
            continue
        print(hdr.format(
            arm, s["n"],
            "{:.4f}".format(s.get("jump_lr", float("nan"))),
            "{:.4f}".format(s.get("base_lr_local", float("nan"))),
            "{:.2f}x".format(s.get("ratio_lr", float("nan"))),
            "{:.4f}".format(s["jump_wrap"]),
            "{:.4f}".format(s["base_wrap_local"]),
            "{:.2f}x".format(s["ratio_wrap"])))
    print("\nwritten to {}".format(args.out))


if __name__ == "__main__":
    main()
