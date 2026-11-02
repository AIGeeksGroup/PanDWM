"""FID of the early and late windows of the long-video dev set, per arm (2026-10-14; a diagnostic, not a gate).

The long-video gate (long4_arms.py) only compares sharpness and frame change with GT, and those ratios cannot rank
the arms: v83h2000 is over-sharp in some scenes and v83h4000 blurry in others, and their mean |log sharp ratio| ties
(report section 26).  This measures how real the frames look, as a distribution: torchmetrics FID (normalize=True on
the stitched pano, the same metric as the 149-clip evaluation) between the generated half and the real half of
gt_vs_generated.mp4.  All arms use the same real frames, from the first arm's videos, so they compare at equal n.
Windows of the 192 generated frames: early = windows 1-6 (frames 0-95), late = windows 7-12 (frames 96-191).
  python3 long_fid.py <root> <out.json> --scenes ... --seeds 42 43 --arms A B ...
"""
import argparse
import json

import cv2
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

cv2.setNumThreads(2)
FRAMES, HALF = 192, 96


def read(path):
    cap = cv2.VideoCapture(path)
    real, fake = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h = fr.shape[0] // 2
        real.append(cv2.cvtColor(fr[:h], cv2.COLOR_BGR2RGB))
        fake.append(cv2.cvtColor(fr[h:], cv2.COLOR_BGR2RGB))
    if len(fake) != FRAMES:
        raise RuntimeError(f"{path}: decoded {len(fake)} frames, expected {FRAMES}")
    return np.stack(real), np.stack(fake)


def to_tensor(frames, device):
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--scenes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = {(arm, part): FrechetInceptionDistance(normalize=True).to(dev)
           for arm in a.arms for part in ("early", "late")}
    vids = [(sc, sd) for sc in a.scenes for sd in a.seeds]
    for i, (sc, sd) in enumerate(vids):
        gt = None
        for arm in a.arms:
            real, fake = read(f"{a.root}/{arm}_scene{sc}_seed{sd}/gt_vs_generated.mp4")
            if gt is None:
                gt = real                                  # one shared set of real frames for every arm
            for part, sl in (("early", slice(0, HALF)), ("late", slice(HALF, FRAMES))):
                m = fid[(arm, part)]
                for s in range(sl.start, sl.stop, a.batch):
                    e = min(s + a.batch, sl.stop)
                    m.update(to_tensor(gt[s:e], dev), real=True)
                    m.update(to_tensor(fake[s:e], dev), real=False)
        print(f"video {i + 1}/{len(vids)} scene {sc} seed {sd}", flush=True)
    out = {"scenes": a.scenes, "seeds": a.seeds, "n_frames_per_part": len(vids) * HALF, "fid": {}}
    print(f"{len(vids)} videos, {len(vids) * HALF} frames per part and arm (FID at equal n; compare only across arms)")
    for arm in a.arms:
        early, late = (float(fid[(arm, p)].compute()) for p in ("early", "late"))
        out["fid"][arm] = {"early": early, "late": late}
        print(f"FID {arm:14s} early (windows 1-6) {early:7.2f}   late (windows 7-12) {late:7.2f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("LONG_FID_DONE")


if __name__ == "__main__":
    main()
