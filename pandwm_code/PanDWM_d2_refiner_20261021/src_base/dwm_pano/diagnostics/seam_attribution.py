#!/usr/bin/env python3
"""Seam attribution experiment.

Question: of the measured seam discontinuity in generated panoramas, how much
comes from the VAE round-trip (independent per-half encode/decode with zero
padding at half boundaries) versus from the DiT's generation?

Five arms, all measured with the same metric code so the numbers are comparable:

  A  pipeline-real   source 9600 -> slice halves -> LANCZOS 2304x256 -> stitch
  B  vae-roundtrip   same halves -> VAE encode -> VAE decode -> stitch
  C  global-resize   source 9600 -> single resize to 4114x256 (no slicing)
  D  dump-fake       generated PNG from an eval dump
  E  dump-real       real PNG from the same eval dump (cross-check vs A)

Arm C isolates the "two independent LANCZOS resizes" contamination flagged in
the literature review: it is what the seam region looks like with no slicing at
all.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

PANO_W = 9600
PANO_H = 600
HALF_LEFT = (0, 5376)
HALF_RIGHT = (4224, 9600)
HALF_W, HALF_H = 2304, 256
OVERLAP_W = 494
STITCH_W = 2 * HALF_W - OVERLAP_W  # 4114
SEAM_COL = HALF_W                  # 2304: first column taken from the right half


def slice_halves(img):
    """Replicates dwm_pano.datasets.nuscenes_pano._slice_full_pano_to_halves."""
    left = img.crop((HALF_LEFT[0], 0, HALF_LEFT[1], PANO_H))
    right = img.crop((HALF_RIGHT[0], 0, HALF_RIGHT[1], PANO_H))
    return (left.resize((HALF_W, HALF_H), Image.LANCZOS),
            right.resize((HALF_W, HALF_H), Image.LANCZOS))


def stitch(left, right):
    """Replicates ctsd_pano._stitch_pano: cat(left, right[..., 494:])."""
    return np.concatenate([left, right[:, OVERLAP_W:]], axis=1)


def measure(pano):
    """pano: float array (H, W, 3) in [0, 1]. Returns discontinuity stats.

    jump_*   mean |col_i - col_{i+1}| across the seam
    base_*   same statistic over neighbouring column pairs, as the local
             texture floor to normalise against
    """
    h, w, _ = pano.shape
    col_diff = np.abs(pano[:, 1:] - pano[:, :-1]).mean(axis=(0, 2))  # (w-1,)

    out = {}
    # left/right seam: boundary between column SEAM_COL-1 and SEAM_COL
    if w == STITCH_W:
        i = SEAM_COL - 1
        out["jump_lr"] = float(col_diff[i])
        win = np.concatenate([col_diff[i - 20:i], col_diff[i + 1:i + 21]])
        out["base_lr_local"] = float(win.mean())
        out["ratio_lr"] = out["jump_lr"] / out["base_lr_local"]

    # wrap seam: last column vs first column
    out["jump_wrap"] = float(np.abs(pano[:, -1] - pano[:, 0]).mean())
    win = np.concatenate([col_diff[:20], col_diff[-20:]])
    out["base_wrap_local"] = float(win.mean())
    out["ratio_wrap"] = out["jump_wrap"] / out["base_wrap_local"]

    out["base_global"] = float(col_diff.mean())
    return out


def to_float(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default="/path/to/pandwm_root/data/val/rgb")
    ap.add_argument("--dump-dir", default="/path/to/pandwm_root/eval_dumps/smoke2")
    ap.add_argument("--vae-path",
                    default="/path/to/pandwm_root/ckpts/stable-diffusion-3.5-medium")
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = {arm: [] for arm in ("A_pipeline_real", "B_vae_roundtrip",
                                   "C_global_resize", "D_dump_fake",
                                   "E_dump_real")}

    # ---- arms A / B / C: from source panoramas -----------------------------
    sources = sorted(glob.glob(os.path.join(args.source_dir, "*.jpg")))
    step = max(1, len(sources) // args.count)
    sources = sources[::step][:args.count]
    print("source panoramas: {}".format(len(sources)), flush=True)

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae")
    vae = vae.to(args.device).eval()
    print("VAE loaded on {}".format(args.device), flush=True)

    for n, path in enumerate(sources):
        img = Image.open(path).convert("RGB")
        if img.size != (PANO_W, PANO_H):
            print("  skip {} (size {})".format(path, img.size))
            continue

        left, right = slice_halves(img)
        la, ra = to_float(left), to_float(right)
        results["A_pipeline_real"].append(measure(stitch(la, ra)))

        # single global resize, no slicing: what the seam region looks like
        # when no per-half resize is involved at all
        results["C_global_resize"].append(
            measure(to_float(img.resize((STITCH_W, HALF_H), Image.LANCZOS))))

        with torch.no_grad():
            batch = torch.from_numpy(np.stack([la, ra])).permute(0, 3, 1, 2)
            batch = (batch * 2 - 1).to(args.device, dtype=vae.dtype)
            latent = vae.encode(batch).latent_dist.mode()
            recon = vae.decode(latent).sample
            recon = ((recon + 1) / 2).clamp(0, 1).float().cpu()
            recon = recon.permute(0, 2, 3, 1).numpy()
        results["B_vae_roundtrip"].append(measure(stitch(recon[0], recon[1])))

        if (n + 1) % 4 == 0:
            print("  {}/{}".format(n + 1, len(sources)), flush=True)

    # ---- arms D / E: from an existing eval dump ----------------------------
    for arm, sub in (("D_dump_fake", "fake"), ("E_dump_real", "real")):
        files = sorted(glob.glob(os.path.join(args.dump_dir, sub, "*.png")))
        step = max(1, len(files) // args.count)
        files = files[::step][:args.count]
        print("{}: {} files".format(arm, len(files)), flush=True)
        for path in files:
            arr = to_float(Image.open(path).convert("RGB"))
            if arr.shape[1] != STITCH_W:
                continue
            results[arm].append(measure(arr))

    # ---- aggregate ---------------------------------------------------------
    summary = {}
    for arm, rows in results.items():
        if not rows:
            summary[arm] = {"n": 0}
            continue
        keys = rows[0].keys()
        summary[arm] = {"n": len(rows)}
        summary[arm].update(
            {k: float(np.mean([r[k] for r in rows])) for k in keys})

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_image": results}, f, indent=2)

    print("\n=== SEAM ATTRIBUTION SUMMARY ===")
    hdr = "{:<18} {:>4} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8}"
    print(hdr.format("arm", "n", "jump_lr", "base_lr", "ratio", "jump_wrap",
                     "base_wrap", "ratio"))
    for arm in ("A_pipeline_real", "B_vae_roundtrip", "C_global_resize",
                "E_dump_real", "D_dump_fake"):
        s = summary[arm]
        if not s.get("n"):
            print("{:<18} {:>4}".format(arm, 0))
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
