"""G2 overlay previews of the controllability GT (2026-10-21): recorded views (rec448, recband) with required boxes
(green: vehicle, lime: pedestrian), ignore regions (yellow), road GT in the region (green tint), region non-road (blue
tint), and optionally the network outputs on these frames (--run-models: detections >= 0.5 magenta / red, predicted
road contour cyan). Runs on CPU. Writes <out>/clipNNN_fFF.png: rows = CAM_ORDER cameras, columns = arms.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import cam_band_feats as cbf
import ctrl_common as cc
from ctrl_predict import make_models


def draw(u8, mask, g, k, ci, det=None, road=None):
    img = u8.permute(1, 2, 0).numpy().copy()
    img[~mask] = 128
    over = img.astype(np.float32)
    gt = cc.unpack(g["road"][k, ci]) & cc.unpack(g["region"][k, ci])
    reg = cc.unpack(g["region"][k, ci]) & ~gt
    over[gt] = 0.65 * over[gt] + 0.35 * np.array([0, 255, 0])
    over[reg] = 0.75 * over[reg] + 0.25 * np.array([0, 80, 255])
    img = np.clip(over, 0, 255).astype(np.uint8)
    sel = (g["b_t"] == k) & (g["b_cam"] == ci)
    for b, cls, role in zip(g["b_box"][sel], g["b_cls"][sel], g["b_role"][sel]):
        if role == cc.IGN and cls == cc.PED:
            continue  # vehicle ignore regions only, to keep the picture readable
        col = (255, 220, 0) if role == cc.IGN else ((0, 255, 0) if cls == cc.VEH else (160, 255, 60))
        cv2.rectangle(img, (int(round(b[0])), int(round(b[1]))), (int(round(b[2])) - 1, int(round(b[3])) - 1), col, 1)
    if road is not None:
        cnt, _ = cv2.findContours(road.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnt, -1, (0, 255, 255), 1)
    if det is not None:
        for b, s, lab in zip(*det):
            if s >= cc.SCORE_T and int(lab) in cc.COCO_CLASS:
                col = (255, 0, 255) if cc.COCO_CLASS[int(lab)] == cc.VEH else (255, 0, 0)
                cv2.rectangle(img, (int(round(b[0])), int(round(b[1]))), (int(round(b[2])) - 1, int(round(b[3])) - 1), col, 1)
    return img


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rec448", default="/path/to/pandwm_root/pandwm_runs/baselines_20261019/rec448_val150")
    ap.add_argument("--positions", default="31,56,80,140")
    ap.add_argument("--frames", default="3,18,25,34")
    ap.add_argument("--map", default="val150_scene_map.json")
    ap.add_argument("--run-models", default="", help="e.g. frcnn,m2f_cs (CPU)")
    ap.add_argument("--model-dir", default="/path/to/pandwm_root/models_2026/ctrl_eval")
    ap.add_argument("--pred-dir", default="", help="draw stored predictions: <pred-dir>/{rec448,recband}__<tag>/clipNNN.npz")
    ap.add_argument("--tag", default="frcnn+m2f_cs")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keep = [int(p) for p in a.positions.split(",")]
    frames = [int(f) for f in a.frames.split(",")]
    config, dataset, collate, rows = cbf.load_val150(a.base_config)
    rows = [r for r in rows if int(r["test_index"]) in keep]
    by_test = {int(v["test_index"]): int(k) for k, v in json.load(open(a.map)).items()}
    geometry = cbf.LegacyInverseOwnGeometry()
    cpu = torch.device("cpu")
    models = make_models(a.run_models.split(","), cpu, a.model_dir) if a.run_models else None
    for row in rows:
        pos = int(row["test_index"])
        batch = collate([dataset[int(row["relaxed_index"])]])
        rig = cbf.batch_rig(batch)
        geo, mask = cc.eval_mask(geometry, *rig)
        geo = dict(geo, valid=mask)
        rec, _, _ = cbf.run_views(a.rec448, by_test[pos], row, rig, cpu)
        band = cbf.canvas_to_cams(cbf.recorded_canvas(batch, cpu, 128), geo)
        gts = {t: dict(np.load(Path(a.gt_dir) / t / f"clip{pos:03d}.npz")) for t in ("opendwm", "band")}
        for f in frames:
            k = f - cc.FIRST
            cols = []
            for u8, timing, arm in ((rec, "opendwm", "rec448"), (band, "band", "recband")):
                det = road = None
                pf = Path(a.pred_dir) / f"{arm}__{a.tag}" / f"clip{pos:03d}.npz" if a.pred_dir else None
                if pf is not None and pf.exists():
                    z = np.load(pf)
                    det, road = [], cc.unpack(z["road"][k])
                    for ci in range(6):
                        s = z["det_img"] == k * 6 + ci
                        det.append((z["det_box"][s], z["det_score"][s], z["det_label"][s]))
                elif models is not None:
                    x = cbf.masked_float(u8[k:k + 1], mask)[0]
                    det, road = models[0](x), models[1](x)
                cols.append(np.concatenate([draw(u8[k, ci], mask[ci], gts[timing], k, ci,
                                                 None if det is None else det[ci], None if road is None else road[ci])
                                            for ci in range(6)], 0))
            grid = np.concatenate(cols, 1)
            cv2.imwrite(str(out / f"clip{pos:03d}_f{f:02d}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"[preview] clip{pos:03d} f{f:02d}", flush=True)
    print("CTRL_PREVIEW_DONE", flush=True)


if __name__ == "__main__":
    main()
