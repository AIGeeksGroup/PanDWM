"""Per-window pixel statistics of long videos, generated / recorded (2026-10-14; a diagnostic).

What kind of systematic drift lifts the late-window FID: brightness, contrast, colour or sharpness.  Per 16-frame
window of gt_vs_generated.mp4 (recorded top, generated bottom; every 2nd frame), on the stitched pano:
  luma     mean of L (CIELAB, 0-255 OpenCV scale)
  contrast std of L over the frame
  sat      HSV saturation on lit pixels (V > 60; dark-pixel sensor noise reads as saturation, report section 6)
  chroma   mean sqrt(a^2 + b^2) (a, b centred at 128)
  sharp    Laplacian variance of the grey frame
Ratios generated / recorded per video and window, then the mean over videos (log-mean for sharp).  Also the share of
videos whose window-12 luma moved more than 5% away from the recorded one, by direction, and the same table for night
videos (recorded window-1 luma < 70; dark-pixel noise makes their saturation meaningless) and day videos apart.
  python3 long_color.py <root> <out.json> --arms A B --scenes ... --seeds 42 43
"""
import argparse
import json
import time

import cv2
import numpy as np

cv2.setNumThreads(2)
W, FRAMES = 12, 192          # defaults; --windows sets both
KEYS = ("luma", "contrast", "sat", "chroma", "sharp")


def stats(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lit = hsv[..., 2] > 60
    return {"luma": float(lab[..., 0].mean()), "contrast": float(lab[..., 0].std()),
            "sat": float(hsv[..., 1][lit].mean()) if lit.any() else float("nan"),
            "chroma": float(np.sqrt((lab[..., 1] - 128) ** 2 + (lab[..., 2] - 128) ** 2).mean()),
            "sharp": float(cv2.Laplacian(grey, cv2.CV_32F).var())}


def measure(path):
    global W, FRAMES
    for attempt in range(3):
        cap = cv2.VideoCapture(path)
        acc = {s: {k: [[] for _ in range(W)] for k in KEYS} for s in ("gen", "rec")}
        k = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if k % 2 == 0:
                h = fr.shape[0] // 2
                w = min(k // 16, W - 1)
                for side, img in (("rec", fr[:h]), ("gen", fr[h:])):
                    for key, v in stats(img).items():
                        acc[side][key][w].append(v)
            k += 1
        if k == FRAMES:
            return {s: {key: [float(np.nanmean(x)) for x in acc[s][key]] for key in KEYS} for s in acc}
        print(f"RETRY {path}: {k} frames", flush=True)
        time.sleep(10)
    raise RuntimeError(f"{path}: could not decode {FRAMES} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--scenes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--windows", type=int, default=12)
    a = ap.parse_args()
    global W, FRAMES
    W, FRAMES = a.windows, 16 * a.windows
    vids = [(sc, sd) for sc in a.scenes for sd in a.seeds]
    out = {"videos": vids, "arms": {}}
    for arm in a.arms:
        ratio = {key: np.zeros((len(vids), W)) for key in KEYS}
        rec_luma = np.zeros((len(vids), W))
        for i, (sc, sd) in enumerate(vids):
            m = measure(f"{a.root}/{arm}_scene{sc}_seed{sd}/gt_vs_generated.mp4")
            for key in KEYS:
                ratio[key][i] = np.asarray(m["gen"][key]) / np.maximum(np.asarray(m["rec"][key]), 1e-6)
            rec_luma[i] = m["rec"]["luma"]
        res = {}
        for key in KEYS:
            r = ratio[key]
            res[key] = (np.exp(np.log(r).mean(0)) if key == "sharp" else r.mean(0)).tolist()
        night = rec_luma[:, 0] < 70
        for name, sel in (("night", night), ("day", ~night)):
            if sel.any():
                res[name] = {"n": int(sel.sum()), **{key: (np.exp(np.log(ratio[key][sel]).mean(0)) if key == "sharp"
                                                         else ratio[key][sel].mean(0)).tolist() for key in KEYS}}
        d12 = ratio["luma"][:, -1]
        res["luma_w12_brighter_5pct"] = float((d12 > 1.05).mean())
        res["luma_w12_darker_5pct"] = float((d12 < 0.95).mean())
        res["abs_log_luma_w12"] = float(np.abs(np.log(d12)).mean())
        res["abs_log_luma_w1"] = float(np.abs(np.log(ratio["luma"][:, 0])).mean())
        out["arms"][arm] = res
        print(f"== {arm} (generated / recorded, mean over {len(vids)} videos; sharp: geometric mean)")
        print("  window   " + " ".join(f"{w:6d}" for w in range(1, W + 1)))
        for key in KEYS:
            print(f"  {key:8s} " + " ".join(f"{x:6.3f}" for x in res[key]))
        print(f"  |log luma ratio| window 1 {res['abs_log_luma_w1']:.3f}, window 12 {res['abs_log_luma_w12']:.3f}; "
              f"window 12 brighter >5%: {res['luma_w12_brighter_5pct']:.2f}, darker >5%: "
              f"{res['luma_w12_darker_5pct']:.2f}", flush=True)
        for name in ("night", "day"):
            if name in res:
                keys = ("luma",) if name == "night" else ("luma", "sat", "chroma", "contrast", "sharp")
                for key in keys:
                    print(f"  {name:5s} n={res[name]['n']:2d} {key:8s} " + " ".join(f"{x:6.3f}" for x in res[name][key]))
    json.dump(out, open(a.out, "w"), indent=1)
    print("LONG_COLOR_DONE")


if __name__ == "__main__":
    main()
