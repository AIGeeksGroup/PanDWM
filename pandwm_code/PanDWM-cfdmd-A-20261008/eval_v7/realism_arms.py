"""Look stats and frame-to-frame change for several arms of one 149-clip run (2026-10-11).

realism3.py generalised to any arm list (it hard-codes b0 / dmd1585 / one tag under the HPC1 run root).
Each clip's gt_vs_generated.mp4 holds the 32 GENERATED frames (2 windows x 16; GT top half, generated bottom half);
the 3 context frames are not in the video.  realism3.py assumed 35 frames, so its "w1" / "w2" frame change is the
change ending at video frames 4..18 and 19..31: "w1" holds 1 window-boundary step (15 -> 16) and the first 3 steps
of window 2, and the steps 0 -> 1 .. 2 -> 3 are skipped.  Both splits are reported:
  * flick_w1 / flick_w2: realism3.py's split, unchanged (comparable with the earlier HPC1 numbers);
  * change_win1 / change_win2: mean absolute grey-level change between consecutive frames inside window 1
    (frames 0..15) and window 2 (16..31), 15 steps each; change_boundary: the step 15 -> 16 between the windows.
Look stats (saturation, contrast = grey std, sharpness = Laplacian variance, brightness) on frames 3, 7, ..., 31 and
PSNR per window from summary.json, as in realism3.py.  GT values come from the first arm's videos (realism3.py used
b0's; list b0 first to match it).
  python3 realism_arms.py <root> <out.json> <arm> [<arm> ...]     (clips in <root>/<arm>/clipNNN)
"""
import json
import sys

import cv2
import numpy as np

root, out_path, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
KEYS = ("sat", "con", "sharp", "bright")
CHANGE = ("flick_w1", "flick_w2", "change_win1", "change_win2", "change_boundary")
FRAMES = 32


def stats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return hsv[..., 1].mean(), g.std(), cv2.Laplacian(g, cv2.CV_32F).var(), g.mean()


def grey(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


res = {a: {"psnr_w1": [], "psnr_w2": [], **{k: [] for k in CHANGE}, **{k: [] for k in KEYS}} for a in arms}
gt = {**{k: [] for k in CHANGE}, **{k: [] for k in KEYS}}
for i in range(149):
    for a in arms:
        d = f"{root}/{a}/clip{i:03d}"
        s = json.load(open(f"{d}/summary.json"))
        res[a]["psnr_w1"].append(s["psnr_mean_db_per_window"][0])
        res[a]["psnr_w2"].append(s["psnr_mean_db_per_window"][1])
        cap = cv2.VideoCapture(f"{d}/gt_vs_generated.mp4")
        k, prev = 0, None
        steps = {side: {name: [] for name in CHANGE} for side in ("g", "r")}
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            h = fr.shape[0] // 2
            cur = (grey(fr[h:]), grey(fr[:h]))
            if prev is not None:
                for side, c, p in (("g", cur[0], prev[0]), ("r", cur[1], prev[1])):
                    v = np.abs(c - p).mean()
                    if k >= 4:                                   # realism3.py's split
                        steps[side]["flick_w1" if k < 19 else "flick_w2"].append(v)
                    steps[side]["change_win1" if k < 16 else "change_boundary" if k == 16 else "change_win2"].append(v)
            prev = cur
            if k >= 3 and k % 4 == 3:
                for key, v in zip(KEYS, stats(fr[h:])):
                    res[a][key].append(v)
                if a == arms[0]:
                    for key, v in zip(KEYS, stats(fr[:h])):
                        gt[key].append(v)
            k += 1
        if k != FRAMES:
            raise RuntimeError(f"{d}: {k} frames, expected {FRAMES}")
        for name in CHANGE:
            res[a][name].append(np.mean(steps["g"][name]))
            if a == arms[0]:
                gt[name].append(np.mean(steps["r"][name]))

m = lambda x: float(np.mean(x))
out = {"clips": 149, "root": root, "frames_per_video": FRAMES, "gt_from": arms[0],
       "gt": {k: m(v) for k, v in gt.items()}, "arms": {}, "paired_psnr": {}, "paired_change": {}}
for a in arms:
    r = res[a]
    out["arms"][a] = {k: m(v) for k, v in r.items()}
    for name in CHANGE:
        out["arms"][a][f"{name}_vs_gt"] = m(r[name]) / m(gt[name]) - 1
    q = out["arms"][a]
    print(f"{a:10s} PSNR w1 {q['psnr_w1']:.2f} w2 {q['psnr_w2']:.2f} | sat {q['sat']:.1f} contrast {q['con']:.1f} "
          f"sharp {q['sharp']:.0f} bright {q['bright']:.1f} | realism3 change w1 {q['flick_w1']:.2f} "
          f"({q['flick_w1_vs_gt']:+.1%}) w2 {q['flick_w2']:.2f} ({q['flick_w2_vs_gt']:+.1%}) | window 1 "
          f"{q['change_win1']:.2f} ({q['change_win1_vs_gt']:+.1%}) window 2 {q['change_win2']:.2f} "
          f"({q['change_win2_vs_gt']:+.1%}) boundary {q['change_boundary']:.2f} ({q['change_boundary_vs_gt']:+.1%})")
g = out["gt"]
print(f"{'GT':10s} {'':24s}| sat {g['sat']:.1f} contrast {g['con']:.1f} sharp {g['sharp']:.0f} bright "
      f"{g['bright']:.1f} | realism3 change w1 {g['flick_w1']:.2f} w2 {g['flick_w2']:.2f} | window 1 "
      f"{g['change_win1']:.2f} window 2 {g['change_win2']:.2f} boundary {g['change_boundary']:.2f}")
for a in arms:
    for b in arms:
        if a == b:
            continue
        for w in ("w1", "w2"):
            d = np.array(res[a][f"psnr_{w}"]) - np.array(res[b][f"psnr_{w}"])
            out["paired_psnr"][f"{a}-{b}_{w}"] = [float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))),
                                                   int((d > 0).sum())]
        for name in CHANGE:
            d = np.array(res[a][name]) - np.array(res[b][name])
            out["paired_change"][f"{a}-{b}_{name}"] = [float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))),
                                                       int((d < 0).sum())]
for i, a in enumerate(arms):
    for b in arms[i + 1:]:
        for w in ("w1", "w2"):
            v = out["paired_psnr"][f"{a}-{b}_{w}"]
            print(f"paired PSNR {a}-{b} {w}: {v[0]:+.3f} dB (se {v[1]:.3f}; better on {v[2]}/149)")
        for name in ("change_win1", "change_win2", "change_boundary"):
            v = out["paired_change"][f"{a}-{b}_{name}"]
            print(f"paired {name} {a}-{b}: {v[0]:+.3f} grey levels (se {v[1]:.3f}; {a} calmer on {v[2]}/149)")
json.dump(out, open(out_path, "w"), indent=1)
print("REALISM_DONE")
