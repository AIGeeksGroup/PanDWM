"""Band and sky masks of the tab:gs_recon targets (2026-10-22, PanDWM-gs-recon-20261022/RECON_PLAN_20261022.md).

* band: recon/band.py (exact inverse of the OneBEV legacy warp) once per distinct rig (sensor2ego + K of the six
  cameras) and resolution -> <out>/band_<rig>_<H>x<W>.npz {"valid": (6, H, W) bool}; <out>/rigs.json maps
  "<test_index>/<target>" to its rig id.
* sky: DA3METRIC-LARGE sky probability of every recorded target image (1600x900 -> 392x700, ImageNet-normalised),
  resized bilinearly to each resolution -> <out>/sky_<test_index>.npz {"<target>_<H>x<W>": (6, H, W) uint8 x 255}.
  A pixel is sky where the probability >= 0.3 (dwm_pano_4d.da3_depth.SKY_THRESHOLD).
Camera order = targets.json cam_order (pano_projection.CAM_ORDER).

    python -m dwm_pano.recon.masks --targets targets.json --out-dir masks --resolutions 224x400,352x640
        --da3-src <repo/src> --da3-metric <DA3METRIC-LARGE> [--band-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dwm_pano.recon.band import band_geometry

DA3_HW = (392, 700)


def parse_res(text):
    return [tuple(int(v) for v in r.split("x")) for r in text.split(",") if r.strip()]


def rig_of(target):
    s2e = np.asarray([c["sensor2ego"] for c in target["cams"]], dtype=np.float64)
    K = np.asarray([c["K"] for c in target["cams"]], dtype=np.float64)
    key = json.dumps([np.round(s2e, 5).tolist(), np.round(K, 3).tolist()])
    return hashlib.sha1(key.encode()).hexdigest()[:12], s2e, K


def load_image(path, hw=None):
    image = Image.open(path).convert("RGB")
    if hw is not None:
        image = image.resize((hw[1], hw[0]), Image.BICUBIC)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--resolutions", default="224x400,352x640")
    p.add_argument("--band-only", action="store_true")
    p.add_argument("--da3-src", default="")
    p.add_argument("--da3-metric", default="")
    args = p.parse_args()
    data = json.loads(args.targets.read_text())
    res = parse_res(args.resolutions)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rigs, rig_of_target = {}, {}
    for s in data["samples"]:
        for name, t in s["targets"].items():
            rid, s2e, K = rig_of(t)
            rigs.setdefault(rid, (s2e, K))
            rig_of_target[f"{s['test_index']}/{name}"] = rid
    (args.out_dir / "rigs.json").write_text(json.dumps({"rigs": sorted(rigs), "target_rig": rig_of_target}) + "\n")
    for rid, (s2e, K) in sorted(rigs.items()):
        for hw in res:
            path = args.out_dir / f"band_{rid}_{hw[0]}x{hw[1]}.npz"
            if path.is_file():
                continue
            geo = band_geometry(s2e, K, [[1600, 900]] * 6, hw)
            np.savez_compressed(path, valid=geo["valid"])
            print(f"[recon-masks] band {rid} {hw}: valid {geo['valid'].mean(axis=(1, 2)).round(3).tolist()}",
                  flush=True)
    print(f"[recon-masks] {len(rigs)} rigs", flush=True)
    if args.band_only:
        return
    from dwm_pano_4d.da3_depth import IMAGENET_MEAN, IMAGENET_STD, SKY_THRESHOLD, load_da3_net
    device = torch.device("cuda", 0)
    model, name = load_da3_net(args.da3_src, args.da3_metric, device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    for s in data["samples"]:
        path = args.out_dir / f"sky_{s['test_index']:03d}.npz"
        if path.is_file():
            continue
        out = {}
        for tname, t in s["targets"].items():
            x = torch.stack([load_image(c["image"], DA3_HW) for c in t["cams"]]).to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sky = model(((x - mean) / std)[None]).sky[0].float()                   # (6, 392, 700)
            for hw in res:
                prob = F.interpolate(sky[:, None], size=hw, mode="bilinear", align_corners=False)[:, 0]
                out[f"{tname}_{hw[0]}x{hw[1]}"] = (prob.clamp(0, 1) * 255).round().byte().cpu().numpy()
        np.savez_compressed(path, **out)
        frac = {k: round(float((v >= SKY_THRESHOLD * 255).mean()), 3) for k, v in out.items() if "224x400" in k}
        print(f"[recon-masks] sky {s['test_index']}: {frac}", flush=True)
    print("RECON_MASKS_DONE", flush=True)


if __name__ == "__main__":
    main()
