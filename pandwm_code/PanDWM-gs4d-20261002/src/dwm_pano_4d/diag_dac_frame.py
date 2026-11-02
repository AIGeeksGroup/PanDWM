"""Diagnostic for report section 37 (2026-10-18): DAC range vs the depth label on one real pano.

Writes <out>/<ts>_dac_vs_label.png: rows = RGB, DAC range on the 9600 x 600 jpg, label converted to range (z / cos,
the pipeline's full-pano approximation), log2(DAC / label) (blue = DAC nearer); and prints statistics per row band
and for the label's 100 m pixels (sky / holes).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from dwm_pano.gs_geometry import pano_source_camera_cosine_map
from dwm_pano_4d.dac_depth import DacRanger


def turbo(values, lo, hi):
    t = np.clip((np.log(np.clip(values, lo, hi)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)
    return cv2.applyColorMap((255 * (1 - t)).astype(np.uint8), cv2.COLORMAP_TURBO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timestamps", nargs="+", required=True)
    p.add_argument("--rgb-dir", required=True)
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--dac-root", required=True)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = a.dac_root
    dac = DacRanger(d, f"{d}/checkpoints/dac_swinl_outdoor.json", f"{d}/checkpoints/dac_swinl_outdoor.pt",
                    torch.device("cuda", 0))
    cos = pano_source_camera_cosine_map().astype(np.float32)
    for ts in a.timestamps:
        rgb = cv2.cvtColor(cv2.imread(f"{a.rgb_dir}/{ts}.jpg"), cv2.COLOR_BGR2RGB)
        r_dac = dac.pano_range(rgb.astype(np.float32) / 255.0)
        with h5py.File(f"{a.depth_dir}/{ts}.h5", "r") as h:
            z = np.asarray(h["depth"], dtype=np.float32)
        lab = z / np.clip(cos, 1e-3, None)
        sky = z >= 99.0
        ratio = np.log2(np.clip(r_dac, 0.1, None) / np.clip(lab, 0.1, None))
        print(f"[diag-dac] {ts}: dac p5/p50/p95 {np.percentile(r_dac, [5, 50, 95]).round(2)}  "
              f"label p5/p50/p95 {np.percentile(lab[~sky], [5, 50, 95]).round(2)}  sky {sky.mean():.3f}", flush=True)
        print(f"[diag-dac]   dac on label-sky pixels p5/p50/p95 "
              f"{np.percentile(r_dac[sky], [5, 50, 95]).round(2) if sky.any() else 'none'}", flush=True)
        for r0 in range(0, 600, 100):
            band = slice(r0, r0 + 100)
            m = ~sky[band]
            print(f"[diag-dac]   rows {r0:3d}-{r0 + 99}: median log2(dac/label) {np.median(ratio[band][m]):+.3f}  "
                  f"|log2| {np.median(np.abs(ratio[band][m])):.3f}", flush=True)
        w = 4800
        rows = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)[:, :w], turbo(r_dac, 1, 100)[:, :w], turbo(lab, 1, 100)[:, :w],
                cv2.applyColorMap((np.clip(ratio / 2 + 0.5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :w]]
        img = np.concatenate(rows, axis=0)
        cv2.imwrite(str(a.out / f"{ts}_dac_vs_label.png"), cv2.resize(img, (2400, 1200), interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    main()
