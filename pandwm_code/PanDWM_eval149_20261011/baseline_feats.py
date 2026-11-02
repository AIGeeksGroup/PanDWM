"""Val-150 features of a six-camera baseline run, with the paper's instrument (2026-10-19, runs on HPC3).

Input: a run directory with clipNNN/{fake.pt, calib.pt, meta.json} (run_unimlvg_clips.py layout): uint8 six views
[T, 6, 3, h, w] in OpenDWM camera order and the clip's rig calibration. Each clip's frames 3..34 are stitched into
the 256x4114 panorama canvas with stitch_views.stitch_six (OneBEV hard seam, validated on recorded frames) and passed
through clip_feats.features, the function that produced every other arm's Val-150 features: per-frame Inception
(torchmetrics FID path) and pano-patch I3D on the two 16-frame windows. Output npz has clip_feats.py's fields, with
test_index = Val-150 position (Dev-149 test_index, scene-0018 = 149), so tests_analyze.py can pair it with the
recorded Val-150 features.
  python3 baseline_feats.py --runs RUN --map val150_scene_map.json --out feats_val150/<name>.npz
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from clip_feats import I3D, features, self_check
from dwm_pano.metrics.fvd_pano import PanoPatchFrechetVideoDistance
from stitch_views import stitch_six

FIRST, LAST = 3, 34  # clip frames scored for every arm


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--map", required=True, help="json {baseline clip index: {test_index, scene_token}}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--allow-partial", action="store_true", help="smoke tests only: score the finished subset")
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"SKIP {a.out} (exists)", flush=True)
        return
    mapping = {int(k): v for k, v in json.load(open(a.map)).items()}
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    fid = FrechetInceptionDistance(normalize=True).to(device)
    fvd = PanoPatchFrechetVideoDistance(I3D, sequence_count=16).to(device)
    runs = Path(a.runs)
    clips = sorted(int(p.name[4:]) for p in runs.glob("clip*") if (p / "meta.json").exists())
    if any(c not in mapping for c in clips) or (not a.allow_partial and sorted(mapping) != clips):
        raise RuntimeError(f"{len(clips)} finished clips in {runs}, map has {len(mapping)}")
    n = len(clips)
    inc = np.zeros((n, 32, 2048), np.float32)
    i3d = np.zeros((n, 2, 6, 400), np.float32)
    meta_out = {"test_index": [], "scene_token": [], "seed": []}
    checked, t0 = False, time.time()
    for k, ci in enumerate(clips):
        d = runs / f"clip{ci:03d}"
        meta = json.loads((d / "meta.json").read_text())
        fake = torch.load(d / "fake.pt", map_location="cpu", weights_only=True)
        frames = meta["generated_frames"]
        sel = [frames.index(f) for f in range(FIRST, LAST + 1)]
        canvas, _ = stitch_six(fake[sel], torch.load(d / "calib.pt", weights_only=True), device=device)
        pano = (canvas.to(device).float() / 255.0)[None, :, None]  # (1, 32, 1, 3, 256, 4114)
        f_inc, f_i3d = features(fid, fvd, pano)
        if not checked:
            self_check(fid, fvd, pano, f_inc, f_i3d, device)
            checked = True
        inc[k], i3d[k] = f_inc.cpu().numpy(), f_i3d.cpu().numpy()
        meta_out["test_index"].append(int(mapping[ci]["test_index"]))
        meta_out["scene_token"].append(mapping[ci]["scene_token"])
        meta_out["seed"].append(int(meta["seed"]))
        if (k + 1) % 20 == 0 or k + 1 == n:
            print(f"[bfeats] {k + 1}/{n} {time.time() - t0:.0f}s", flush=True)
    order = np.argsort(meta_out["test_index"])
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, inc=inc[order], i3d=i3d[order], test_index=np.array(meta_out["test_index"])[order],
             scene_token=np.array(meta_out["scene_token"])[order], seed=np.array(meta_out["seed"])[order],
             manifest="val150", arm_root=str(runs) + (f" ({a.name})" if a.name else ""))
    os.replace(tmp, a.out)
    print(f"SAVED {a.out}", flush=True)


if __name__ == "__main__":
    main()
