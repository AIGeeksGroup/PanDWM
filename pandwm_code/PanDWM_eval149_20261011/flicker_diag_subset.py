"""flicker_diag.py restricted to a clip list (2026-10-13), e.g. the Dev-78 selection set.

Copy of flicker_diag.py (md5 67d333e0); the only change is --clips <json> [key]: per-clip statistics, pooled ratios,
motion tertiles (computed on the listed clips' GT) and the --pair numbers use the listed clips only.
  python3 flicker_diag_subset.py <root> <out.json> --clips dev149_nontest_split_20261013.json --pair <b> <a> <arm> ...

Original description:
Where does each arm's frame-to-frame change, softening and colour shift come from? (2026-10-11, v8.2 analysis)

Reads <root>/<arm>/clipNNN/gt_vs_generated.mp4 (32 generated frames = 2 windows x 16, GT top half, generated bottom
half) for every arm and the 149 clips.  Per clip and arm, for the generated and the GT frames:
  * within-window steps (window 1: steps ending at frames 1..15, window 2: 17..31) and the window-boundary step 15 -> 16,
    each as mean absolute grey-level change, split into a low-pass part (Gaussian blur, sigma 4 px) and the
    high-frequency residual, plus the whole-frame mean brightness change;
  * Laplacian sharpness and HSV saturation on frames 3, 7, ..., 31.
Reports ratios over GT (pooled means), per-clip distributions of the within-window change ratio (how many clips are
too static < 0.85 or flickery > 1.2), the ratio by GT motion tertile, the boundary step's low/high-frequency and
brightness make-up, and the per-clip link between sharpness loss and change reduction for the first two arms named in
--pair.  CPU only.
  python3 flicker_diag.py <root> <out.json> --pair <before> <after> <arm> [<arm> ...]
"""
import json
import sys

import cv2
import numpy as np

args = sys.argv[1:]
root, out_path = args[0], args[1]
pair = None
if "--pair" in args:
    i = args.index("--pair")
    pair = (args[i + 1], args[i + 2])
    args = args[:i] + args[i + 3:]
clip_key = "non_test_clip_indices"
if "--clips" not in args:
    raise SystemExit("--clips <json> is required")
i = args.index("--clips")
clips_path = args[i + 1]
args = args[:i] + args[i + 2:]
payload = json.load(open(clips_path))
indices = sorted(int(x) for x in (payload[clip_key] if isinstance(payload, dict) else payload))
if len(set(indices)) != len(indices) or not indices or indices[0] < 0 or indices[-1] > 148:
    raise SystemExit(f"bad clip list: {len(indices)} entries")
arms = args[2:]
PARTS = ("full", "lf", "hf", "bright")


def grey_parts(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lf = cv2.GaussianBlur(g, (0, 0), 4.0)
    return g, lf, g - lf


def step(a, b):
    return {"full": float(np.abs(a[0] - b[0]).mean()), "lf": float(np.abs(a[1] - b[1]).mean()),
            "hf": float(np.abs(a[2] - b[2]).mean()), "bright": float(abs(a[0].mean() - b[0].mean()))}


def clip_stats(path):
    cap = cv2.VideoCapture(path)
    k, prev = 0, None
    acc = {side: {"win": {p: [] for p in PARTS}, "bnd": None, "sharp": [], "sat": []} for side in ("g", "r")}
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h = fr.shape[0] // 2
        cur = {"g": grey_parts(fr[h:]), "r": grey_parts(fr[:h])}
        if prev is not None:
            for side in ("g", "r"):
                s = step(cur[side], prev[side])
                if k == 16:
                    acc[side]["bnd"] = s
                else:
                    for p in PARTS:
                        acc[side]["win"][p].append(s[p])
        if k >= 3 and k % 4 == 3:
            for side, img in (("g", fr[h:]), ("r", fr[:h])):
                acc[side]["sharp"].append(float(cv2.Laplacian(cur[side][0], cv2.CV_32F).var()))
                acc[side]["sat"].append(float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean()))
        prev = cur
        k += 1
    if k != 32:
        raise RuntimeError(f"{path}: {k} frames, expected 32")
    return {side: {"win": {p: float(np.mean(v)) for p, v in acc[side]["win"].items()}, "bnd": acc[side]["bnd"],
                   "sharp": float(np.mean(acc[side]["sharp"])), "sat": float(np.mean(acc[side]["sat"]))}
            for side in ("g", "r")}


per = {a: [clip_stats(f"{root}/{a}/clip{i:03d}/gt_vs_generated.mp4") for i in indices] for a in arms}
gt = per[arms[0]]
arr = lambda a, side, key, part=None: np.array([(c[side][key][part] if part else c[side][key]) for c in per[a]])
gt_win = arr(arms[0], "r", "win", "full")
tert = np.quantile(gt_win, [1 / 3, 2 / 3])
out = {"root": root, "arms": arms, "gt_from": arms[0], "clips": len(indices), "clip_list": clips_path,
       "clip_indices": indices, "summary": {}, "per_clip": {}}
print(f"{'arm':9s} | within-window change/GT: full   low   high  bright | boundary/GT: full   low   high  bright | "
      f"sharp/GT  sat/GT | per-clip change ratio p10 p50 p90, <0.85, >1.2 | by GT motion low/mid/high")
for a in arms:
    s = {}
    for part in PARTS:
        s[f"win_{part}"] = float(arr(a, "g", "win", part).mean() / arr(arms[0], "r", "win", part).mean())
        s[f"bnd_{part}"] = float(arr(a, "g", "bnd", part).mean() / arr(arms[0], "r", "bnd", part).mean())
    s["sharp"] = float(arr(a, "g", "sharp").mean() / arr(arms[0], "r", "sharp").mean())
    s["sat"] = float(arr(a, "g", "sat").mean() / arr(arms[0], "r", "sat").mean())
    ratio = arr(a, "g", "win", "full") / gt_win
    s["clip_ratio_q"] = [float(x) for x in np.quantile(ratio, [0.1, 0.25, 0.5, 0.75, 0.9])]
    s["clips_static"] = int((ratio < 0.85).sum())
    s["clips_flicker"] = int((ratio > 1.2).sum())
    s["by_motion"] = [float(ratio[(gt_win <= tert[0])].mean()), float(ratio[(gt_win > tert[0]) & (gt_win <= tert[1])].mean()),
                      float(ratio[(gt_win > tert[1])].mean())]
    out["summary"][a] = s
    out["per_clip"][a] = {"win_ratio": ratio.tolist(), "sharp": arr(a, "g", "sharp").tolist(),
                          "bnd_full": arr(a, "g", "bnd", "full").tolist()}
    q = s["clip_ratio_q"]
    print(f"{a:9s} | {s['win_full']:.3f} {s['win_lf']:.3f} {s['win_hf']:.3f} {s['win_bright']:.3f} | "
          f"{s['bnd_full']:.3f} {s['bnd_lf']:.3f} {s['bnd_hf']:.3f} {s['bnd_bright']:.3f} | {s['sharp']:.3f} "
          f"{s['sat']:.3f} | {q[0]:.2f} {q[2]:.2f} {q[4]:.2f}, {s['clips_static']}, {s['clips_flicker']} | "
          f"{s['by_motion'][0]:.2f} {s['by_motion'][1]:.2f} {s['by_motion'][2]:.2f}")
out["gt"] = {"win_full": float(gt_win.mean()), "bnd_full": float(arr(arms[0], "r", "bnd", "full").mean()),
             "sharp": float(arr(arms[0], "r", "sharp").mean()), "sat": float(arr(arms[0], "r", "sat").mean()),
             "motion_tertiles": [float(x) for x in tert]}
if pair:
    b, a = pair
    d_sharp = arr(a, "g", "sharp") - arr(b, "g", "sharp")
    d_change = arr(a, "g", "win", "full") - arr(b, "g", "win", "full")
    d_hf = arr(a, "g", "win", "hf") - arr(b, "g", "win", "hf")
    d_lf = arr(a, "g", "win", "lf") - arr(b, "g", "win", "lf")
    corr = float(np.corrcoef(d_sharp, d_change)[0, 1])
    out["pair"] = {"before": b, "after": a, "corr_dsharp_dchange": corr,
                   "mean_d_change_lf": float(d_lf.mean()), "mean_d_change_hf": float(d_hf.mean()),
                   "clips_sharper": int((d_sharp > 0).sum()), "clips_calmer": int((d_change < 0).sum())}
    print(f"pair {b} -> {a}: within-window change lf {d_lf.mean():+.3f} hf {d_hf.mean():+.3f} grey levels; "
          f"sharpness lower on {(d_sharp < 0).sum()}/{len(indices)} clips; corr(d sharpness, d change) = {corr:+.2f}")
print(f"GT: within-window change {out['gt']['win_full']:.2f}, boundary {out['gt']['bnd_full']:.2f}, sharp "
      f"{out['gt']['sharp']:.0f}, sat {out['gt']['sat']:.1f}, motion tertiles {tert[0]:.2f} / {tert[1]:.2f}")
json.dump(out, open(out_path, "w"))
print("FLICKER_DIAG_DONE")
