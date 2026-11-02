"""long3.py for any arm list and run root (2026-10-11): long videos (8 dev scenes x 12 windows, seed 42).

Same measurements as long3.py, unchanged so the numbers stay comparable with the HPC1 tables (long3_v7d1000.json,
long3_v81f1600.json): PSNR per window from summary.json; frame-to-frame change and Laplacian sharpness of the generated
half vs the GT half of gt_vs_generated.mp4.  Note: the video holds the 192 generated frames, while long3.py's split
assumes 3 context frames first, so its per-window change / sharpness buckets are offset by 3 frames; PSNR per window
and all 12-window means are unaffected.  GT values come from the first arm's videos.
  python3 long3_arms.py <root> <out.json> <arm> [<arm> ...]      (scenes in <root>/<arm>_scene<idx>_seed42)
"""
import json
import sys

import cv2
import numpy as np

root, out_path, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
SCENES = (0, 14, 21, 32, 41, 49, 62, 82)
W = 12
psnr = {a: np.zeros((len(SCENES), W)) for a in arms}
change = {a: np.zeros((len(SCENES), W)) for a in arms + ["gt"]}
sharp = {a: np.zeros((len(SCENES), W)) for a in arms + ["gt"]}
for si, sc in enumerate(SCENES):
    for a in arms:
        d = f"{root}/{a}_scene{sc}_seed42"
        s = json.load(open(f"{d}/summary.json"))
        psnr[a][si] = [w["psnr_mean_db"] for w in s["windows"]]
        cap = cv2.VideoCapture(f"{d}/gt_vs_generated.mp4")
        k, prev = 0, None
        acc = {x: [[] for _ in range(W)] for x in ("cg", "cr", "sg", "sr")}
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            h = fr.shape[0] // 2
            g = cv2.cvtColor(fr[h:], cv2.COLOR_BGR2GRAY).astype(np.float32)
            r = cv2.cvtColor(fr[:h], cv2.COLOR_BGR2GRAY).astype(np.float32)
            if k >= 3:
                w = (k - 3) // 16
                if w < W:
                    if k >= 4 and prev is not None:
                        acc["cg"][w].append(np.abs(g - prev[0]).mean())
                        acc["cr"][w].append(np.abs(r - prev[1]).mean())
                    if k % 4 == 3:
                        acc["sg"][w].append(cv2.Laplacian(g, cv2.CV_32F).var())
                        acc["sr"][w].append(cv2.Laplacian(r, cv2.CV_32F).var())
            prev = (g, r)
            k += 1
        change[a][si] = [np.mean(x) for x in acc["cg"]]
        sharp[a][si] = [np.mean(x) for x in acc["sg"]]
        if a == arms[0]:
            change["gt"][si] = [np.mean(x) for x in acc["cr"]]
            sharp["gt"][si] = [np.mean(x) for x in acc["sr"]]
f = lambda v: " ".join(f"{x:6.2f}" for x in v)
print("window      " + " ".join(f"{w:6d}" for w in range(1, W + 1)) + "   mean  late(7-12)")
for a in arms:
    m = psnr[a].mean(0)
    print(f"PSNR {a:9s} {f(m)}  {m.mean():5.2f}  {m[6:].mean():5.2f}")
for a in arms:
    ratio = (change[a] / change["gt"]).mean(0)
    print(f"change/GT {a:9s} {f(ratio)}  {ratio.mean():5.2f}  {ratio[6:].mean():5.2f}")
for a in arms:
    ratio = (sharp[a] / sharp["gt"]).mean(0)
    print(f"sharp/GT  {a:9s} {f(ratio)}  {ratio.mean():5.2f}  {ratio[6:].mean():5.2f}")
out = {"scenes": SCENES, "root": root, "psnr": {a: psnr[a].tolist() for a in arms},
       "change_ratio": {a: (change[a] / change["gt"]).tolist() for a in arms},
       "sharp_ratio": {a: (sharp[a] / sharp["gt"]).tolist() for a in arms}}
json.dump(out, open(out_path, "w"))
print("LONG3_DONE")
