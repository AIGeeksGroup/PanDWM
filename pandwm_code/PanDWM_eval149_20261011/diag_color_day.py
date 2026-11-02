"""Day-only (recorded window-1 luma >= 70) generated / recorded saturation (lit pixels), chroma, sharpness per window."""
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)
root = sys.argv[1]
arms = sys.argv[2:]
scenes = [0, 14, 21, 32, 41, 49, 62, 82, 10, 31, 52, 72]
for arm in arms:
    res = {"sat": [], "chroma": [], "sharp": [], "contrast": []}
    nday = 0
    for sc in scenes:
        for sd in (42, 43):
            cap = cv2.VideoCapture(f"{root}/{arm}_scene{sc}_seed{sd}/gt_vs_generated.mp4")
            acc = {s: {k: [[] for _ in range(12)] for k in res} for s in ("g", "r")}
            k, r0 = 0, []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if k % 4 == 0:
                    h = fr.shape[0] // 2
                    w = min(k // 16, 11)
                    for side, img in (("r", fr[:h]), ("g", fr[h:])):
                        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                        lit = hsv[..., 2] > 60
                        acc[side]["sat"][w].append(hsv[..., 1][lit].mean())
                        acc[side]["chroma"][w].append(np.sqrt((lab[..., 1] - 128) ** 2 + (lab[..., 2] - 128) ** 2).mean())
                        acc[side]["contrast"][w].append(lab[..., 0].std())
                        acc[side]["sharp"][w].append(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
                        if side == "r" and w == 0:
                            r0.append(lab[..., 0].mean())
                k += 1
            assert k == 192
            if np.mean(r0) < 70:
                continue
            nday += 1
            for key in res:
                res[key].append(np.array([np.mean(a) for a in acc["g"][key]]) / np.array([np.mean(b) for b in acc["r"][key]]))
    print(f"== {arm}: day videos only (n={nday}), generated / recorded per window (sharp: geometric mean)")
    print("  window   " + " ".join(f"{w:6d}" for w in range(1, 13)))
    for key, v in res.items():
        v = np.array(v)
        m = np.exp(np.log(v).mean(0)) if key == "sharp" else v.mean(0)
        print(f"  {key:8s} " + " ".join(f"{x:6.3f}" for x in m), flush=True)
