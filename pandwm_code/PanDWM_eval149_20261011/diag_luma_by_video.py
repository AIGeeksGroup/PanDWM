"""Per video: recorded luma at window 1 and generated / recorded luma ratio per window (every 4th frame)."""
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)
root, arm = sys.argv[1], sys.argv[2]
scenes = [0, 14, 21, 32, 41, 49, 62, 82, 10, 31, 52, 72]
print(f"== {arm}: per video luma (L of CIELAB), generated / recorded per window; rec_w1 = recorded window-1 luma")
print("  scene seed rec_w1 " + " ".join(f"{w:5d}" for w in range(1, 13)))
allr = []
for sc in scenes:
    for sd in (42, 43):
        cap = cv2.VideoCapture(f"{root}/{arm}_scene{sc}_seed{sd}/gt_vs_generated.mp4")
        g, r = [[] for _ in range(12)], [[] for _ in range(12)]
        k = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if k % 4 == 0:
                h = fr.shape[0] // 2
                w = min(k // 16, 11)
                r[w].append(cv2.cvtColor(fr[:h], cv2.COLOR_BGR2LAB)[..., 0].mean())
                g[w].append(cv2.cvtColor(fr[h:], cv2.COLOR_BGR2LAB)[..., 0].mean())
            k += 1
        assert k == 192, (sc, sd, k)
        ratio = np.array([np.mean(a) for a in g]) / np.array([np.mean(b) for b in r])
        allr.append((np.mean(r[0]), ratio))
        print(f"  {sc:5d} {sd:4d} {np.mean(r[0]):6.1f} " + " ".join(f"{x:5.2f}" for x in ratio), flush=True)
night = [x for x in allr if x[0] < 70]
day = [x for x in allr if x[0] >= 70]
for name, grp in (("night (rec_w1 < 70)", night), ("day", day)):
    if grp:
        m = np.mean([x[1] for x in grp], axis=0)
        print(f"  {name:20s} n={len(grp):2d} " + " ".join(f"{x:5.2f}" for x in m))
