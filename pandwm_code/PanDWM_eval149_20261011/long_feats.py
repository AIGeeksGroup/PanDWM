"""Per-frame Inception and per-window pano-patch I3D features of long-rollout videos (2026-10-14; a diagnostic).

long_fid.py gives one FID per arm and half, which cannot say how much of the early -> late rise is real loss of image
quality and how much is the model inventing other, plausible content than the one recorded (the late real frames show
places the first 3 frames never showed).  long_analyze.py needs the features per video and frame to split that, to
bootstrap over scenes and to compute FVD per time bucket, so this only extracts and saves them:
  - inc_*: torchmetrics FrechetInceptionDistance(normalize=True) features (2048) of every stitched-pano frame, the same
    instrument as long_fid.py and the 149-clip FID (uint8 video -> /255 -> the metric's own *255 .byte());
  - i3d_*: PanoPatchFrechetVideoDistance features (6 Voronoi camera patches x 400 I3D logits) of every 16-frame window,
    the same cut, resize and I3D weights as the 149-clip FVD (patches cut exactly as in fvd_pano.update).
gt_vs_generated.mp4 holds the real pano on top of the generated one, 16 frames per window from --first-window on.
  python3 long_feats.py <out_dir> --arm TAG=ROOT [...] --scenes 0 14 ... --seeds 42 43 [--first-window 7]
Writes <out_dir>/feats_<TAG>.npz (float32 features; scenes, seeds, first_window, windows).  Existing files are kept.
Pool mode: the REAL half of 149-clip eval videos (<dir>/clipNNN/gt_vs_generated.mp4, 32 frames = 2 windows) as an
unpaired real reference from many scenes:
  python3 long_feats.py <out_dir> --pool NAME=<eval arm dir> --pool-clips <json list or split file key>
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from dwm_pano.metrics.fvd_pano import PanoPatchFrechetVideoDistance

cv2.setNumThreads(4)
I3D = "/path/to/pandwm_root/pandwm_code/TATS_fvd/i3d_pretrained_400.pt"


def read(path, frames):
    cap = cv2.VideoCapture(path)
    real, fake = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h = fr.shape[0] // 2
        real.append(cv2.cvtColor(fr[:h], cv2.COLOR_BGR2RGB))
        fake.append(cv2.cvtColor(fr[h:], cv2.COLOR_BGR2RGB))
    if len(fake) != frames:
        raise RuntimeError(f"{path}: decoded {len(fake)} frames, expected {frames}")
    return np.stack(real), np.stack(fake)


def patches_of(frames):
    """fvd_pano.PanoPatchFrechetVideoDistance.update's six camera patches of (B, T, C, H, 4114) frames."""
    b0, b1, b2, b3, b4, b5 = PanoPatchFrechetVideoDistance.PANO_PATCH_COLS
    w = PanoPatchFrechetVideoDistance.PANO_W_V1
    return [torch.cat([frames[..., b5:w], frames[..., 0:b0]], dim=-1), frames[..., b0:b1], frames[..., b1:b2],
            frames[..., b2:b3], frames[..., b3:b4], frames[..., b4:b5]]


@torch.no_grad()
def i3d_features(fvd, clip):
    """(1, 16, 3, H, W) in [0, 1] -> (6, 400): fvd_pano._accumulate_patches up to the I3D forward."""
    resized = []
    for p in patches_of(clip):
        p = p * 2 - 1
        p = torch.nn.functional.interpolate(p.flatten(0, 1), size=fvd.target_resolution,
                                            mode="bilinear").unflatten(0, p.shape[:2])
        resized.append(p)
    return fvd.inception(torch.cat(resized, dim=0).transpose(1, 2)).float()


@torch.no_grad()
def inc_features(fid, frames, batch=32):
    out = []
    for s in range(0, frames.shape[0], batch):
        x = frames[s:s + batch].permute(0, 3, 1, 2).float().div(255.0)
        out.append(fid.inception((x * 255).byte()).float())
    return torch.cat(out)


def self_check(fvd, fid, real_u8, dev):
    """The extracted features must give the metric objects' own numbers on one window (real vs a shifted copy)."""
    clip_a = torch.from_numpy(real_u8[:16]).to(dev).permute(0, 3, 1, 2).float().div(255.0)[None]
    if len(real_u8) >= 32:
        clip_b = torch.from_numpy(real_u8[16:32]).to(dev).permute(0, 3, 1, 2).float().div(255.0)[None]
    else:   # a one-window video (2026-10-15): the same window reversed in time is another clip
        clip_b = clip_a.flip(1)
    m = PanoPatchFrechetVideoDistance(I3D, sequence_count=16).to(dev)
    m.update(clip_a, real=True)
    m.update(clip_b, real=False)
    fa, fb = i3d_features(fvd, clip_a).double(), i3d_features(fvd, clip_b).double()
    ok_v = (torch.allclose(m.real_features_sum, fa.sum(0), rtol=1e-4, atol=1e-3)
            and torch.allclose(m.fake_features_sum, fb.sum(0), rtol=1e-4, atol=1e-3))
    f = FrechetInceptionDistance(normalize=True).to(dev)
    f.update(clip_a[0], real=True)
    mine = inc_features(fid, torch.from_numpy(real_u8[:16]).to(dev)).double()
    ok_i = torch.allclose(f.real_features_sum, mine.sum(0), rtol=1e-4, atol=1e-3)
    print(f"SELF_CHECK i3d {'ok' if ok_v else 'FAIL'} inception {'ok' if ok_i else 'FAIL'}", flush=True)
    if not (ok_v and ok_i):
        raise RuntimeError("feature extraction differs from the metric objects")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--arm", action="append", default=[], help="TAG=ROOT")
    ap.add_argument("--pool", default=None, help="NAME=<eval arm dir with clipNNN/gt_vs_generated.mp4>")
    ap.add_argument("--pool-clips", default=None,
                    help="json file: a list of clip indices, or a dict holding non_test_clip_indices")
    ap.add_argument("--scenes", type=int, nargs="+", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--first-window", type=int, default=1)
    ap.add_argument("--windows", type=int, default=12, help="last window index in the video")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    fid = FrechetInceptionDistance(normalize=True).to(dev)
    fvd = PanoPatchFrechetVideoDistance(I3D, sequence_count=16).to(dev)
    nwin = a.windows - a.first_window + 1
    frames = 16 * nwin
    checked = False
    if a.pool:
        name, root = a.pool.split("=", 1)
        out = f"{a.out_dir}/feats_pool_{name}.npz"
        spec = json.load(open(a.pool_clips))
        clips = spec["non_test_clip_indices"] if isinstance(spec, dict) else spec
        inc_r = np.zeros((len(clips), 32, 2048), np.float32)
        i3d_r = np.zeros((len(clips), 2, 6, 400), np.float32)
        for i, c in enumerate(clips):
            real, _ = read(f"{root}/clip{int(c):03d}/gt_vs_generated.mp4", 32)
            if not checked:
                self_check(fvd, fid, real, dev)
                checked = True
            inc_r[i] = inc_features(fid, torch.from_numpy(real).to(dev)).cpu().numpy()
            x = torch.from_numpy(real).to(dev)
            for w in range(2):
                i3d_r[i, w] = i3d_features(fvd, x[16 * w:16 * w + 16].permute(0, 3, 1, 2).float().div(255.0)[None]
                                           ).cpu().numpy()
        np.savez(out + ".tmp.npz", inc_real=inc_r, i3d_real=i3d_r, clips=np.array(clips))
        os.replace(out + ".tmp.npz", out)
        print(f"SAVED {out} ({len(clips)} clips)", flush=True)
    for spec in a.arm:
        tag, root = spec.split("=", 1)
        out = f"{a.out_dir}/feats_{tag}.npz"
        if os.path.exists(out):
            print(f"SKIP {tag} (exists)", flush=True)
            continue
        vids = [(sc, sd) for sc in a.scenes for sd in a.seeds]
        inc_g = np.zeros((len(vids), frames, 2048), np.float32)
        inc_r = np.zeros_like(inc_g)
        i3d_g = np.zeros((len(vids), nwin, 6, 400), np.float32)
        i3d_r = np.zeros_like(i3d_g)
        t0 = time.time()
        for i, (sc, sd) in enumerate(vids):
            real, fake = read(f"{root}/{tag}_scene{sc}_seed{sd}/gt_vs_generated.mp4", frames)
            if not checked:
                self_check(fvd, fid, real, dev)
                checked = True
            for arr, src in ((inc_r, real), (inc_g, fake)):
                arr[i] = inc_features(fid, torch.from_numpy(src).to(dev)).cpu().numpy()
            for arr, src in ((i3d_r, real), (i3d_g, fake)):
                x = torch.from_numpy(src).to(dev)
                for w in range(nwin):
                    clip = x[16 * w:16 * w + 16].permute(0, 3, 1, 2).float().div(255.0)[None]
                    arr[i, w] = i3d_features(fvd, clip).cpu().numpy()
            print(f"{tag} video {i + 1}/{len(vids)} scene {sc} seed {sd} {time.time() - t0:.0f}s", flush=True)
        np.savez(out + ".tmp.npz", inc_gen=inc_g, inc_real=inc_r, i3d_gen=i3d_g, i3d_real=i3d_r,
                 scenes=np.array([v[0] for v in vids]), seeds=np.array([v[1] for v in vids]),
                 first_window=a.first_window, windows=a.windows)
        os.replace(out + ".tmp.npz", out)
        print(f"SAVED {out}", flush=True)
    print("LONG_FEATS_DONE", flush=True)


if __name__ == "__main__":
    main()
