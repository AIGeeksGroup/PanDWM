"""Frame-to-frame change of the generated vs the recorded frames inside window 1 and window 7 (2026-10-16 diagnostic
for the side finding in report section 30: from the same drifted history the teacher's window 7 beats the student's).
For each 16-frame window video (recorded half on top, generated half below): mean |frame_t - frame_t-1| of the
generated half / the same for the recorded half, at full resolution and low-pass (8x area downsample), for the
student's own Dev-76-L windows and the teacher evaluations (E2 = window 1 from GT, E1 = window 7 from the student's
history).  Ratio 1.0 = moves like the recording; < 1 under-moves, > 1 over-moves / boils.
  python motion_w1w7.py OUT.json"""
import json
import os
import sys
import time

import cv2
import numpy as np

cv2.setNumThreads(1)
DR = "/path/to/pandwm_root/pandwm_runs"
LV76 = f"{DR}/long_videos_dev76_20261014"
EV = f"{DR}/teacher_v3_20261015/eval"
ARMS = {
    "w1": {"student": (LV76, "v84i4250", "window01.mp4"), "t0": (EV, "t0_gt1", None), "tc": (EV, "tc_gt1", None)},
    "w7": {"student": (LV76, "v84i4250", "window07.mp4"), "t0": (EV, "t0_d7", None), "tc": (EV, "tc_d7", None),
           "b_srr": (EV, "b_srr_d7", None), "b_noise": (EV, "b_noise_d7", None)},
}
SCENES = range(76)


def measure(path):
    cap = cv2.VideoCapture(path)
    prev, k = None, 0
    acc = {"g": [], "r": [], "g_lf": [], "r_lf": []}
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h = fr.shape[0] // 2
        g = cv2.cvtColor(fr[h:], cv2.COLOR_BGR2GRAY).astype(np.float32)
        r = cv2.cvtColor(fr[:h], cv2.COLOR_BGR2GRAY).astype(np.float32)
        lf = [cv2.resize(x, (x.shape[1] // 8, x.shape[0] // 8), interpolation=cv2.INTER_AREA) for x in (g, r)]
        if prev is not None:
            acc["g"].append(np.abs(g - prev[0]).mean())
            acc["r"].append(np.abs(r - prev[1]).mean())
            acc["g_lf"].append(np.abs(lf[0] - prev[2]).mean())
            acc["r_lf"].append(np.abs(lf[1] - prev[3]).mean())
        prev = (g, r, lf[0], lf[1])
        k += 1
    if k != 16:
        raise RuntimeError(f"{path}: decoded {k} frames, expected 16")
    return {key: float(np.mean(v)) for key, v in acc.items()}


def measure_retry(path, tries=3):
    for attempt in range(tries):
        try:
            return measure(path)
        except RuntimeError as e:
            print(f"RETRY {attempt + 1}/{tries}: {e}", flush=True)
            time.sleep(10)
    raise RuntimeError(f"{path}: could not be decoded after {tries} tries")


def main():
    out_path = sys.argv[1]
    res = {}
    for win, arms in ARMS.items():
        res[win] = {}
        for name, (root, tag, fname) in arms.items():
            rows = []
            for i in SCENES:
                d = f"{root}/{tag}_scene{i}_seed42"
                p = f"{d}/{fname}" if fname else f"{d}/gt_vs_generated.mp4"
                if not os.path.exists(p):
                    raise SystemExit(f"MISSING {p}")
                rows.append(measure_retry(p))
            res[win][name] = rows
            g = np.array([x["g"] for x in rows]); r = np.array([x["r"] for x in rows])
            gl = np.array([x["g_lf"] for x in rows]); rl = np.array([x["r_lf"] for x in rows])
            print(f"{win} {name:8s} full: mean-of-ratios {np.mean(g / r):.3f} median {np.median(g / r):.3f} "
                  f"ratio-of-means {g.mean() / r.mean():.3f} | low-pass: mean-of-ratios {np.mean(gl / rl):.3f} "
                  f"median {np.median(gl / rl):.3f} ratio-of-means {gl.mean() / rl.mean():.3f}", flush=True)
    for win in res:
        s = res[win]["student"]
        for name in res[win]:
            if name == "student":
                continue
            o = res[win][name]
            for key, lab in (("g", "full"), ("g_lf", "low-pass")):
                rk = "r" if key == "g" else "r_lf"
                d = np.log(np.array([a[key] / a[rk] for a in s])) - np.log(np.array([b[key] / b[rk] for b in o]))
                print(f"{win} student vs {name:8s} {lab:8s}: mean log-ratio diff {d.mean():+.3f} "
                      f"(se {d.std(ddof=1) / np.sqrt(len(d)):.3f}); student moves less on {int((d < 0).sum())}/{len(d)}",
                      flush=True)
    with open(out_path, "w") as f:
        json.dump(res, f)
    print("MOTION_W1W7_DONE", flush=True)


if __name__ == "__main__":
    main()
