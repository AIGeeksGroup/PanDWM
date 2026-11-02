"""Scores of tab:gs_recon (2026-10-22, RECON_PLAN_20261022.md), the same for every method.

Inputs: targets.json, the masks of masks.py, and a method's renders <renders>/<H>x<W>/sample_<NNN>.pt
({"targets", "rgb" (n, 6, 3, H, W) uint8, "alpha" (n, 6, H, W) uint8, "depth" (n, 6, H, W) fp16}; a method without
opacity gives alpha 255; rgb is premultiplied, i.e. rendered over black).  Every render is first composited over
a 0.5 gray background, rgb + (1 - alpha) * 0.5, as a renderer with a gray background would give.  Per image
(sample x target x camera), with mask M = band & (sky probability < 0.3) and covered C = alpha >= 0.5:
  psnr_cov        PSNR over M & C;
  psnr_fill       PSNR over M (uncovered pixels show the gray background);
  ssim_cov / ssim_fill   mean of the SSIM map (11 x 11 Gaussian window, sigma 1.5, on the composited render) over
                  M & C / over M;
  lpips_vgg_cov / _fill, lpips_alex_cov / _fill   the same for the spatial LPIPS maps (lpips 0.1, spatial=True);
  coverage        |M & C| / |M|.
Depth per image: LiDAR points of the target (0 < z <= 80 m, projected with the scaled K, pixel inside M, rendered
pixel covered): abs_rel, rmse, delta1 on metres, abs_rel_med after scaling by median(z) / median(d); "_static" drops
points inside the dynamic 3D boxes (0.3 m margin).  The sweep is targets[...]["lidar_sync"] when present (the LIDAR_TOP
sample_data nearest in time to the target, moved with its own ego pose, lidar_sync.py; the 12 Hz pkl's own sweep can
be 250 ms older than the cameras), else the pkl's; points enter the box test through lidar_to_box, the boxes being
read as the pkl stores them, mmdet3d v0.x (x, y, z, w, l, h, -yaw - pi/2).
The GT image is the raw 1600x900 frame resized with PIL BICUBIC (antialiased).  Summary: mean over images (per target
type and overall); depth pooled per image then averaged over images with points.

    python -m dwm_pano.recon.score --targets targets.json --masks masks --renders renders/ours/4d --res 224x400
        --out score_ours4d_224x400.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SKY_THRESHOLD = 0.3
MAX_DEPTH = 80.0
BOX_MARGIN = 0.3


def gaussian_window(size=11, sigma=1.5, device="cpu"):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_map(a, b, window):
    """(N, 3, H, W) in [0, 1] -> (N, H, W) SSIM map (channel mean), reflect padding."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    pad = window.shape[-1] // 2
    w = window.expand(3, 1, -1, -1)
    def filt(x):
        return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), w, groups=3)
    mu_a, mu_b = filt(a), filt(b)
    saa, sbb, sab = filt(a * a) - mu_a ** 2, filt(b * b) - mu_b ** 2, filt(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return m.mean(1)


def psnr(err, mask):
    n = mask.sum()
    if n == 0:
        return None
    mse = (err * mask[None]).sum() / (3 * n)
    return float(-10 * torch.log10(mse.clamp_min(1e-10)))


def masked_mean(x, mask):
    n = mask.sum()
    return None if n == 0 else float((x * mask).sum() / n)


def inside_boxes(points, boxes, margin):
    """points (N, 3) in the boxes' LiDAR frame, boxes (M, 7) mmdet3d v0.x (x, y, z, w, l, h, yaw') with
    yaw' = -yaw - pi/2 and z the box centre -> (N,) bool."""
    inside = np.zeros(points.shape[0], dtype=bool)
    for x, y, z, w, l, h, yaw_v0x in boxes:
        heading = -yaw_v0x - np.pi / 2
        c, s = np.cos(heading), np.sin(heading)
        rel = points - np.array([x, y, z])
        along = rel[:, 0] * c + rel[:, 1] * s
        across = -rel[:, 0] * s + rel[:, 1] * c
        inside |= (np.abs(along) <= l / 2 + margin) & (np.abs(across) <= w / 2 + margin) & \
            (np.abs(rel[:, 2]) <= h / 2 + margin)
    return inside


def depth_metrics(d_hat, z):
    if z.size == 0:
        return None
    ratio = np.maximum(d_hat / z, z / d_hat)
    med = np.median(z) / max(np.median(d_hat), 1e-6)
    return {"abs_rel": float(np.mean(np.abs(d_hat - z) / z)), "rmse": float(np.sqrt(np.mean((d_hat - z) ** 2))),
            "delta1": float(np.mean(ratio < 1.25)), "abs_rel_med": float(np.mean(np.abs(med * d_hat - z) / z)),
            "points": int(z.size)}


def scaled_K(K, hw):
    """Native 1600x900 K -> output size with the half-pixel convention (calibrated_stitch.prepare_gsplat_intrinsics)."""
    h, w = hw
    sx, sy = w / 1600.0, h / 900.0
    k = np.array(K, dtype=np.float64).copy()
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] = (k[0, 2] + 0.5) * sx - 0.5
    k[1, 2] = (k[1, 2] + 0.5) * sy - 0.5
    return k


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--masks", required=True, type=Path)
    p.add_argument("--renders", required=True, type=Path, help="method root containing <H>x<W>/")
    p.add_argument("--res", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--samples", default="")
    args = p.parse_args()
    import lpips
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    hw = tuple(int(v) for v in args.res.split("x"))
    data = json.loads(args.targets.read_text())
    rigs = json.loads((args.masks / "rigs.json").read_text())["target_rig"]
    bands = {}
    nets = {"vgg": lpips.LPIPS(net="vgg", spatial=True, verbose=False).to(device).eval(),
            "alex": lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()}
    window = gaussian_window(device=device)
    rows, depth_rows = [], []
    positions = list(range(data["count"])) if not args.samples else \
        [int(x) for x in args.samples.split(",") if x.strip()]
    t0 = time.monotonic()
    for pos in positions:
        s = data["samples"][pos]
        rpath = args.renders / f"{hw[0]}x{hw[1]}" / f"sample_{s['test_index']:03d}.pt"
        if not rpath.is_file():
            raise FileNotFoundError(rpath)
        pack = torch.load(rpath, map_location="cpu", weights_only=True)
        sky = np.load(args.masks / f"sky_{s['test_index']:03d}.npz")
        for ti, name in enumerate(pack["targets"]):
            t = s["targets"][name]
            rid = rigs[f"{s['test_index']}/{name}"]
            if (rid, hw) not in bands:
                bands[(rid, hw)] = np.load(args.masks / f"band_{rid}_{hw[0]}x{hw[1]}.npz")["valid"]
            band = torch.from_numpy(bands[(rid, hw)]).to(device)
            skym = torch.from_numpy(sky[f"{name}_{hw[0]}x{hw[1]}"]).to(device).float() / 255.0 >= SKY_THRESHOLD
            gt = torch.stack([torch.from_numpy(np.asarray(Image.open(c["image"]).convert("RGB").resize(
                (hw[1], hw[0]), Image.BICUBIC)).copy()).permute(2, 0, 1) for c in t["cams"]]).to(device).float() / 255
            pred = pack["rgb"][ti].to(device).float() / 255
            alpha = pack["alpha"][ti].to(device).float() / 255
            covered = alpha >= 0.5
            mask = band & ~skym
            fill = pred + (1.0 - alpha[:, None]) * 0.5                       # over a 0.5 gray background
            err = (fill - gt) ** 2
            ss = ssim_map(fill, gt, window)
            lp = {k: net(fill * 2 - 1, gt * 2 - 1)[:, 0] for k, net in nets.items()}
            for c in range(6):
                m, mc = mask[c].float(), (mask[c] & covered[c]).float()
                row = {"test_index": s["test_index"], "target": name, "camera": t["cams"][c]["name"],
                       "coverage": masked_mean(covered[c].float(), mask[c].float()),
                       "psnr_cov": psnr(err[c], mc), "psnr_fill": psnr(err[c], m),
                       "ssim_cov": masked_mean(ss[c], mc), "ssim_fill": masked_mean(ss[c], m)}
                for k in nets:
                    row[f"lpips_{k}_cov"] = masked_mean(lp[k][c], mc)
                    row[f"lpips_{k}_fill"] = masked_mean(lp[k][c], m)
                rows.append(row)
            sweep = t.get("lidar_sync") or {**t["lidar"], "lidar_to_box": np.eye(4).tolist()}
            pts = np.fromfile(sweep["path"], dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
            to_box = np.asarray(sweep["lidar_to_box"])
            dynamic = inside_boxes(pts @ to_box[:3, :3].T + to_box[:3, 3],
                                   np.asarray(t["lidar"]["dynamic_boxes"]).reshape(-1, 7), BOX_MARGIN)
            depth = pack["depth"][ti].float().numpy()
            alpha_np = alpha.cpu().numpy()
            mask_np = mask.cpu().numpy()
            for c in range(6):
                l2c = np.asarray(sweep["lidar_to_cam"][c])
                cam = pts @ l2c[:3, :3].T + l2c[:3, 3]
                z = cam[:, 2]
                ok = (z > 0.5) & (z <= MAX_DEPTH)
                k = scaled_K(t["cams"][c]["K"], hw)
                u = k[0, 0] * cam[:, 0] / np.maximum(z, 1e-6) + k[0, 2]
                v = k[1, 1] * cam[:, 1] / np.maximum(z, 1e-6) + k[1, 2]
                ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
                ok &= (ui >= 0) & (ui < hw[1]) & (vi >= 0) & (vi < hw[0])
                idx = np.nonzero(ok)[0]
                ui, vi = ui[idx], vi[idx]
                keep = mask_np[c, vi, ui] & (alpha_np[c, vi, ui] >= 0.5)
                idx, ui, vi = idx[keep], ui[keep], vi[keep]
                d_hat = depth[c, vi, ui].astype(np.float64)
                pos_d = d_hat > 1e-3
                idx, d_hat = idx[pos_d], d_hat[pos_d]
                drow = {"test_index": s["test_index"], "target": name, "camera": t["cams"][c]["name"]}
                for tag, sel in (("all", np.ones(idx.size, dtype=bool)), ("static", ~dynamic[idx])):
                    drow[tag] = depth_metrics(d_hat[sel], z[idx][sel])
                depth_rows.append(drow)
        print(f"[recon-score] sample {pos} ({time.monotonic() - t0:.0f}s)", flush=True)

    def summarize(rs, key_rows):
        out = {}
        for group in ["all"] + sorted({r["target"] for r in rs}):
            sel = [r for r in rs if group == "all" or r["target"] == group]
            agg = {"images": len(sel)}
            for k in [k for k in sel[0] if k not in ("test_index", "target", "camera")]:
                vals = [r[k] for r in sel if r[k] is not None]
                agg[k] = float(np.mean(vals)) if vals else None
                agg[f"{k}_n"] = len(vals)
            out[group] = agg
        return out

    def summarize_depth(rs):
        out = {}
        for group in ["all"] + sorted({r["target"] for r in rs}):
            sel = [r for r in rs if group == "all" or r["target"] == group]
            agg = {}
            for tag in ("all", "static"):
                vals = [r[tag] for r in sel if r[tag] is not None]
                agg[tag] = {k: float(np.mean([v[k] for v in vals])) for k in
                            ("abs_rel", "rmse", "delta1", "abs_rel_med")} if vals else None
                if vals:
                    agg[tag]["images"] = len(vals)
                    agg[tag]["points"] = int(sum(v["points"] for v in vals))
            out[group] = agg
        return out

    result = {"renders": str(args.renders), "resolution": list(hw), "samples": len(positions),
              "image": summarize(rows, rows), "depth": summarize_depth(depth_rows),
              "lidar": "lidar_sync" if all("lidar_sync" in data["samples"][q]["targets"][n] for q in positions
                                            for n in data["samples"][q]["targets"]) else "pkl",
              "protocol": "tab:gs_recon (dwm_pano/recon/score.py): over 0.5 gray, band & non-sky mask, "
                          "covered = alpha >= 0.5"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    (args.out.with_suffix(".rows.json")).write_text(json.dumps({"image": rows, "depth": depth_rows}) + "\n")
    a = result["image"]["all"]
    print(f"[recon-score] {args.renders.name} {hw}: psnr_cov {a['psnr_cov']:.2f} psnr_fill {a['psnr_fill']:.2f} "
          f"ssim {a['ssim_cov']:.3f} lpips_vgg {a['lpips_vgg_cov']:.3f} coverage {a['coverage']:.3f} "
          f"depth {json.dumps(result['depth']['all'])}", flush=True)


if __name__ == "__main__":
    main()
