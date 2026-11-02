"""Validation of the 256x448 six-camera band protocol (cam_band_feats.py) before any arm is scored (2026-10-20).

(a) pixel check: the RECORDED canvas through the bridge vs the RECORDED 256x448 views (rec448 run: OpenDWM adapter of
    the raw nuScenes images), inside the band mask, per camera: MAE (0..255) and PSNR, split into pixels whose canvas
    column is owned by the camera itself and pixels in a neighbour-owned column (expected parallax residue). A shift
    search (integer dx, dy at 256x448, own-column pixels) and an isotropic zoom search about the principal point tell
    a geometric misalignment from appearance differences.
(b) mask coverage over all 150 clips: pixel fraction, fraction of rows with any covered pixel, central-column row
    fraction (the memory note's 53.8% CAM_BACK / ~84% others), neighbour-owned fraction (by sampled canvas column and
    by the bridge's 50 m probe), first covered row (MagicDrive-V2's 26/450 padded top rows must be outside the mask).
(c) preview of one clip and frame: rows = cameras (CAM_ORDER), columns = recorded views | recorded canvas -> bridge |
    ours P s42 -> bridge | UniMLVG P s42, all masked; red = mask edge, yellow = own / neighbour column boundary.
  python3 cam_band_check.py --base-config cd_eval_frozen149_hpc3.json --rec448 RUN --unimlvg RUN --ours R1+R2 --out D
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cam_band_feats import (
    CAM_ORDER, FIRST, GEOMETRIES, OUT_HW, batch_rig, canvas_to_cams, load_val150, make_decoder, masked_float,
    ours_canvas, recorded_canvas, run_views)


def psnr(err2):
    return float(10 * np.log10(255.0 ** 2 / max(err2, 1e-12)))


def zoom(img, s, K):
    """Isotropic zoom of [N, 3, H, W] about the principal point (scaled to 256x448) by factor s (s > 1 magnifies)."""
    n, _, h, w = img.shape
    cx, cy = K[0, 2] * w / 1600.0, K[1, 2] * h / 900.0
    ys, xs = torch.meshgrid(torch.arange(h, device=img.device) + 0.5, torch.arange(w, device=img.device) + 0.5,
                            indexing="ij")
    sx, sy = cx + (xs - cx) / s, cy + (ys - cy) / s
    grid = torch.stack([sx / w * 2 - 1, sy / h * 2 - 1], -1)[None].expand(n, -1, -1, -1).to(img.dtype)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--rec448", required=True)
    ap.add_argument("--unimlvg", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--map", default="val150_scene_map.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", default="31,56,80,140")
    ap.add_argument("--frames", default="3,18,25,34")
    ap.add_argument("--preview", default="56:25")
    ap.add_argument("--blend", type=int, default=128)
    ap.add_argument("--geometry", choices=sorted(GEOMETRIES), default="bridge")
    ap.add_argument("--skip-coverage", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", 0)
    config, dataset, collate, rows = load_val150(a.base_config)
    by_test = {int(v["test_index"]): int(k) for k, v in json.load(open(a.map)).items()}
    geometry = GEOMETRIES[a.geometry]()
    report = {"cams": list(CAM_ORDER), "resolution_hw": list(OUT_HW), "geometry": a.geometry}

    # (b) coverage over all clips (calibration only)
    cov = []
    for row in ([] if a.skip_coverage else rows):
        b = collate([dataset[int(row["relaxed_index"])]])
        geo = geometry(*batch_rig(b))
        v, own = geo["valid"], geo["owner"]
        cov.append({
            "pixel": v.mean((1, 2)).tolist(),
            "rows_any": v.any(2).mean(1).tolist(),
            "rows_center_col": v[:, :, OUT_HW[1] // 2].mean(1).tolist(),
            "foreign_col": [float((own[c][v[c]] != c).mean()) for c in range(6)],
            "foreign_probe50": [r.get("foreign_region_fraction", float("nan")) for r in geo["reports"]],
            "first_row": [int(np.argmax(v[c].any(1))) for c in range(6)],
            "last_row": [int(OUT_HW[0] - 1 - np.argmax(v[c].any(1)[::-1])) for c in range(6)]})
    agg = {} if not cov else {k: {c: [float(np.min([r[k][i] for r in cov])), float(np.mean([r[k][i] for r in cov])),
                   float(np.max([r[k][i] for r in cov]))] for i, c in enumerate(CAM_ORDER)} for k in cov[0]}
    report["coverage_min_mean_max"] = agg
    report["rigs"] = len(geometry.cache)
    print(f"[coverage] {len(rows)} clips, {len(geometry.cache)} distinct rigs", flush=True)
    for k in (agg and ("pixel", "rows_any", "rows_center_col", "foreign_col", "foreign_probe50", "first_row")) or ():
        print(f"  {k:<16}" + "".join(f"{c[4:]:>18}={agg[k][c][1]:.3f}" for c in CAM_ORDER), flush=True)

    # (a) pixel check
    decode = make_decoder(config, device)
    keep = [int(c) for c in a.clips.split(",")]
    frames = [int(f) for f in a.frames.split(",")]
    sel = [f - FIRST for f in frames]
    pix = {c: {"own": [0.0, 0.0, 0], "foreign": [0.0, 0.0, 0]} for c in CAM_ORDER}
    shifts, zooms = {c: [] for c in CAM_ORDER}, {c: [] for c in CAM_ORDER}
    ref_shift = {c: [] for c in CAM_ORDER}
    pc, pf = (int(x) for x in a.preview.split(":"))
    for row in rows:
        pos = int(row["test_index"])
        if pos not in keep and pos != pc:
            continue
        b = collate([dataset[int(row["relaxed_index"])]])
        rig = batch_rig(b)
        geo = geometry(*rig)
        band = canvas_to_cams(recorded_canvas(b, device, a.blend), geo)          # [32, 6, 3, H, W] uint8
        rec, _, _ = run_views(a.rec448, by_test[pos], row, rig, device)
        if pos in keep:
            for c, cam in enumerate(CAM_ORDER):
                v = torch.as_tensor(geo["valid"][c], device=device)
                own = torch.as_tensor(geo["owner"][c] == c, device=device) & v
                frn = ~own & v
                x, y = band[sel, c].float(), rec[sel, c].float()
                e = (x - y)
                for key, m in (("own", own), ("foreign", frn)):
                    mm = m[None, None].expand_as(e)
                    pix[cam][key][0] += float(e.abs()[mm].sum())
                    pix[cam][key][1] += float((e ** 2)[mm].sum())
                    pix[cam][key][2] += int(mm.sum())
                best = (1e9, 0, 0)
                for dy in range(-8, 9):
                    for dx in range(-8, 9):
                        ys = slice(max(0, dy), OUT_HW[0] + min(0, dy))
                        yd = slice(max(0, -dy), OUT_HW[0] + min(0, -dy))
                        xs = slice(max(0, dx), OUT_HW[1] + min(0, dx))
                        xd = slice(max(0, -dx), OUT_HW[1] + min(0, -dx))
                        m = own[ys, xs] & own[yd, xd]
                        d = (x[:, :, ys, xs] - y[:, :, yd, xd]).abs().mean(1)[:, m].mean()
                        if float(d) < best[0]:
                            best = (float(d), dx, dy)
                shifts[cam].append(best)
                K = rig[1][c]
                zb = (1e9, 1.0)
                for s in np.arange(0.85, 1.151, 0.01):
                    d = (zoom(x / 255.0, float(s), K) * 255.0 - y).abs().mean(1)[:, own].mean()
                    if float(d) < zb[0]:
                        zb = (float(d), float(s))
                zooms[cam].append(zb)
                # reference: the same recorded views one frame later (temporal change at the same geometry)
                y2 = rec[[min(s + 1, 31) for s in sel], c].float()
                ref_shift[cam].append(float((y2 - y).abs().mean(1)[:, own].mean()))
            print(f"[pixel] clip{pos:03d} done", flush=True)
        if pos == pc:
            ours, _, _ = ours_canvas(a.ours.split("+"), row, decode, a.blend)
            ours = canvas_to_cams(ours, geo)
            uni, _, _ = run_views(a.unimlvg, by_test[pos], row, rig, device)
            k = pf - FIRST
            cols = []
            for src in (rec, band, ours, uni):
                x = masked_float(src[k:k + 1], geo["valid"])[0]                  # [6, 3, H, W]
                cols.append(x)
            grid = torch.stack(cols, 1)                                          # [6, 4, 3, H, W]
            v = torch.as_tensor(geo["valid"], device=device)
            edge = v ^ F.max_pool2d(v.float()[:, None], 3, 1, 1)[:, 0].bool()
            own = torch.as_tensor(geo["owner"] == np.arange(6)[:, None, None], device=device)
            oedge = (own ^ F.max_pool2d(own.float()[:, None], 3, 1, 1)[:, 0].bool()) & v
            red = torch.tensor([1.0, 0.0, 0.0], device=device)[:, None, None]
            yel = torch.tensor([1.0, 1.0, 0.0], device=device)[:, None, None]
            for j in range(4):
                grid[:, j] = torch.where(edge[:, None], red, grid[:, j])
            grid[:, 1] = torch.where(oedge[:, None], yel, grid[:, 1])
            img = grid.permute(0, 3, 1, 4, 2).reshape(6 * OUT_HW[0], 4 * OUT_HW[1], 3)
            img = (img.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            from PIL import Image
            Image.fromarray(img).save(out / f"preview_clip{pc:03d}_f{pf}.png")
            np.savez_compressed(out / f"preview_clip{pc:03d}_f{pf}.npz", rec=rec[k].cpu().numpy(),
                                band=band[k].cpu().numpy(), ours=ours[k].cpu().numpy(), uni=uni[k].cpu().numpy(),
                                valid=geo["valid"], owner=geo["owner"])
            print(f"[preview] {out}/preview_clip{pc:03d}_f{pf}.png", flush=True)
    report["pixel"] = {}
    for cam in CAM_ORDER:
        r = {}
        for key in ("own", "foreign"):
            s1, s2, n = pix[cam][key]
            r[key] = {"pixels": n, "mae": s1 / max(n, 1), "psnr_db": psnr(s2 / max(n, 1))}
        allp = pix[cam]["own"][2] + pix[cam]["foreign"][2]
        r["all"] = {"mae": (pix[cam]["own"][0] + pix[cam]["foreign"][0]) / allp,
                    "psnr_db": psnr((pix[cam]["own"][1] + pix[cam]["foreign"][1]) / allp)}
        r["best_shift_own"] = shifts[cam]
        r["best_zoom_own"] = zooms[cam]
        r["next_frame_mae_own"] = float(np.mean(ref_shift[cam]))
        report["pixel"][cam] = r
        print(f"[pixel] {cam:<16} all MAE {r['all']['mae']:.2f} PSNR {r['all']['psnr_db']:.2f} | own MAE "
              f"{r['own']['mae']:.2f} PSNR {r['own']['psnr_db']:.2f} | neighbour MAE {r['foreign']['mae']:.2f} PSNR "
              f"{r['foreign']['psnr_db']:.2f} ({r['foreign']['pixels'] / allp:.1%}) | next-frame MAE "
              f"{r['next_frame_mae_own']:.2f} | best shift (mae,dx,dy) {[tuple(round(v, 2) for v in s) for s in shifts[cam]]}"
              f" | best zoom {[tuple(round(v, 3) for v in z) for z in zooms[cam]]}", flush=True)
    report["clips"], report["frames"] = keep, frames
    (out / "cam_band_check.json").write_text(json.dumps(report, indent=1))
    print("CAMCHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
