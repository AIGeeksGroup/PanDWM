"""Round-3 visual check (GS_FT3_PLAN_20261030.md gate 5): one frame of the GS-Dev sample videos, the same row of
every arm stacked under the real frame and labelled.  Light (decodes one frame per video); run on HPC3.
    python ft3_frames.py <clip 3 digits> <frame> <out prefix> <x0> <x1> <rows: real,0m,+2m,-2m,+4m> [arms...]"""
import sys

import av
import cv2
import numpy as np

import os
B = "/path/to/pandwm_root/pandwm_runs/gs_dev_hpc3/shift"
TEST = "/path/to/pandwm_root/pandwm_runs/tab5_gen_gs_ft3_20261030"      # FT3_TEST=1: Test-S 71 samples
ROWS = {"0m": 1, "+2m": 2, "-2m": 3, "+4m": 4}
clip, frame, out, x0, x1, rows = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), \
    sys.argv[6].split(",")
arms = sys.argv[7:] or ["FT3B_V0", "FT3B_V2", "FT3_G0", "FT3_G2", "FT3_G3b"]


def get(arm):
    path = f"{TEST}/samples_{arm}/clip{clip}_shift.mp4" if os.environ.get("FT3_TEST") else \
        f"{B}/{arm}/samples/clip{clip}_shift.mp4"
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == frame:
                return f.to_ndarray(format="rgb24")


imgs = {a: get(a) for a in arms}
h = imgs[arms[0]].shape[0] // 5
for row in rows:
    tiles = [("real", imgs[arms[0]][0:h, x0:x1])] + [(f"{a} {row}", imgs[a][ROWS[row] * h:(ROWS[row] + 1) * h, x0:x1])
                                                      for a in arms]
    canvas = []
    for lab, t in tiles:
        t = np.ascontiguousarray(t)
        cv2.putText(t, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
        canvas.append(t)
    big = np.concatenate(canvas, 0)
    big = cv2.resize(big, (big.shape[1] * 2, big.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"{out}_{row}.png", cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    print(out, row, big.shape)
