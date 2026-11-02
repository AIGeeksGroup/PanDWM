"""long3_arms.py for any scene and seed list, plus the pre-registered v8.4 long-video gate (2026-10-13).

Measurements are long3_arms.py's, unchanged (PSNR per window from summary.json; frame-to-frame change and Laplacian
sharpness of the generated half vs the GT half of gt_vs_generated.mp4, per 16-frame window; GT values from the first
arm's videos), so 8-scene seed-42 numbers reproduce long3_drift*.json exactly.  "late" = windows 7-12.
  python3 long4_arms.py <root> <out.json> --scenes 0 14 ... --seeds 42 43 --arms A B ... [--ref v83h2000]
Videos are read from <root>/<arm>_scene<idx>_seed<seed>.
Gate (eval_v7/V82_EVAL_HPC3_20261011.md section 25), for every arm other than --ref:
  L1  mean late sharp/GT over all scene x seed videos >= 0.95
  L2  every scene (mean over seeds) late sharp/GT >= the reference's value for that scene - 0.15
  L3  mean late change/GT over all videos >= 0.95
"""
import argparse
import json
import time

import cv2
import numpy as np

cv2.setNumThreads(2)
W = 12
FRAMES = 192                     # generated frames in gt_vs_generated.mp4 (12 windows x 16)


def measure(d):
    s = json.load(open(f"{d}/summary.json"))
    psnr = [w["psnr_mean_db"] for w in s["windows"]]
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
    if k != FRAMES:
        raise RuntimeError(f"{d}: decoded {k} frames, expected {FRAMES}")
    return psnr, [[np.mean(x) for x in acc[key]] for key in ("cg", "cr", "sg", "sr")]


def measure_retry(d, tries=3):
    """A login node short of threads can make OpenCV return no frames (2026-10-14: 4 videos read as empty while
    flicker_diag ran, all NaN); retry, then fail loudly instead of averaging NaN."""
    for attempt in range(tries):
        try:
            return measure(d)
        except RuntimeError as e:
            print(f"RETRY {attempt + 1}/{tries}: {e}", flush=True)
            time.sleep(10)
    raise RuntimeError(f"{d}: could not be decoded after {tries} tries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--scenes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    vids = [(sc, sd) for sc in a.scenes for sd in a.seeds]
    n = len(vids)
    psnr = {arm: np.zeros((n, W)) for arm in a.arms}
    change = {arm: np.zeros((n, W)) for arm in a.arms + ["gt"]}
    sharp = {arm: np.zeros((n, W)) for arm in a.arms + ["gt"]}
    for i, (sc, sd) in enumerate(vids):
        for arm in a.arms:
            p, (cg, cr, sg, sr) = measure_retry(f"{a.root}/{arm}_scene{sc}_seed{sd}")
            psnr[arm][i], change[arm][i], sharp[arm][i] = p, cg, sg
            if arm == a.arms[0]:
                change["gt"][i], sharp["gt"][i] = cr, sr
    cr = {arm: change[arm] / change["gt"] for arm in a.arms}
    sr = {arm: sharp[arm] / sharp["gt"] for arm in a.arms}
    f = lambda v: " ".join(f"{x:6.2f}" for x in v)
    print(f"{n} videos: scenes {a.scenes} x seeds {a.seeds}")
    print("window           " + " ".join(f"{w:6d}" for w in range(1, W + 1)) + "   mean  late(7-12)")
    for name, table in (("PSNR", psnr), ("change/GT", cr), ("sharp/GT", sr)):
        for arm in a.arms:
            m = table[arm].mean(0)
            print(f"{name:9s} {arm:12s} {f(m)}  {m.mean():5.2f}  {m[6:].mean():5.2f}")
    per_scene = {}
    print("late sharp/GT per scene (mean over seeds) " + " ".join(f"{sc:>6d}" for sc in a.scenes))
    for arm in a.arms:
        late = sr[arm][:, 6:].mean(1)
        per_scene[arm] = [float(np.mean([late[i] for i, v in enumerate(vids) if v[0] == sc])) for sc in a.scenes]
        print(f"  {arm:12s}" + " " * 29 + " ".join(f"{x:6.3f}" for x in per_scene[arm]))
    gate = {}
    if a.ref is not None:
        ref = per_scene[a.ref]
        for arm in a.arms:
            if arm == a.ref:
                continue
            l1 = float(sr[arm][:, 6:].mean())
            drops = [r - x for r, x in zip(ref, per_scene[arm])]
            l3 = float(cr[arm][:, 6:].mean())
            g = {"L1_mean_late_sharp": l1, "L1": l1 >= 0.95, "L2_worst_drop_vs_ref": max(drops),
                 "L2_worst_scene": a.scenes[int(np.argmax(drops))], "L2": max(drops) <= 0.15,
                 "L3_mean_late_change": l3, "L3": l3 >= 0.95}
            g["LONG_PASS"] = g["L1"] and g["L2"] and g["L3"]
            gate[arm] = g
            print(f"GATE {arm}: L1 {l1:.3f} {'ok' if g['L1'] else 'FAIL'} | L2 worst drop {max(drops):+.3f} "
                  f"(scene {g['L2_worst_scene']}) {'ok' if g['L2'] else 'FAIL'} | L3 {l3:.3f} "
                  f"{'ok' if g['L3'] else 'FAIL'} -> {'LONG_PASS' if g['LONG_PASS'] else 'LONG_FAIL'}")
    out = {"scenes": a.scenes, "seeds": a.seeds, "videos": vids, "root": a.root, "ref": a.ref,
           "psnr": {arm: psnr[arm].tolist() for arm in a.arms},
           "change_ratio": {arm: cr[arm].tolist() for arm in a.arms},
           "sharp_ratio": {arm: sr[arm].tolist() for arm in a.arms},
           "late_sharp_per_scene": per_scene, "gate": gate}
    json.dump(out, open(a.out, "w"))
    print("LONG4_DONE")


if __name__ == "__main__":
    main()
