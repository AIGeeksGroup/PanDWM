"""Window-wise appearance diagnostics from GT/generated comparison videos.

These MP4s are lossy visualizations, so the numbers here diagnose failure
types; official PSNR must come from the existing per-frame JSON/latent path.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def js_distance(left, right):
    left = left.astype(np.float64) + 1e-12
    right = right.astype(np.float64) + 1e-12
    left /= left.sum()
    right /= right.sum()
    middle = 0.5 * (left + right)
    return float(np.sqrt(0.5 * np.sum(left * np.log(left / middle)) +
                         0.5 * np.sum(right * np.log(right / middle))))


def frame_stats(gt_bgr, fake_bgr):
    # Quarter-size working copies preserve broad color/structure trends and
    # avoid decoding each 4114-wide frame into a large feature extractor.
    width = max(1, gt_bgr.shape[1] // 4)
    height = max(1, gt_bgr.shape[0] // 4)
    gt = cv2.resize(gt_bgr, (width, height), interpolation=cv2.INTER_AREA)
    fake = cv2.resize(fake_bgr, (width, height), interpolation=cv2.INTER_AREA)
    gt_f = gt.astype(np.float32) / 255.0
    fake_f = fake.astype(np.float32) / 255.0
    gt_gray = cv2.cvtColor(gt_f, cv2.COLOR_BGR2GRAY)
    fake_gray = cv2.cvtColor(fake_f, cv2.COLOR_BGR2GRAY)
    gt_edge = cv2.Laplacian(gt_gray, cv2.CV_32F)
    fake_edge = cv2.Laplacian(fake_gray, cv2.CV_32F)
    gt_hist = np.histogram(gt_gray, bins=32, range=(0, 1))[0]
    fake_hist = np.histogram(fake_gray, bins=32, range=(0, 1))[0]
    gt_hsv = cv2.cvtColor(gt_f, cv2.COLOR_BGR2HSV)
    fake_hsv = cv2.cvtColor(fake_f, cv2.COLOR_BGR2HSV)
    mse = float(np.mean((gt_f - fake_f) ** 2))
    return {
        "gt_rgb_mean": gt_f.mean(axis=(0, 1))[::-1].tolist(),
        "fake_rgb_mean": fake_f.mean(axis=(0, 1))[::-1].tolist(),
        "gt_rgb_std": gt_f.std(axis=(0, 1))[::-1].tolist(),
        "fake_rgb_std": fake_f.std(axis=(0, 1))[::-1].tolist(),
        "rgb_mean_l1": float(np.abs(gt_f.mean(axis=(0, 1)) -
                                      fake_f.mean(axis=(0, 1))).mean()),
        "luma_hist_js": js_distance(gt_hist, fake_hist),
        "gt_saturation": float(gt_hsv[:, :, 1].mean()),
        "fake_saturation": float(fake_hsv[:, :, 1].mean()),
        "gt_edge_energy": float(np.mean(gt_edge ** 2)),
        "fake_edge_energy": float(np.mean(fake_edge ** 2)),
        "mp4_psnr_db": float(-10.0 * np.log10(max(mse, 1e-12))),
        "gt_small_gray": gt_gray,
        "fake_small_gray": fake_gray,
    }


def _lpips_frame(gt_bgr, fake_bgr, metric):
    import torch
    size = (512, 64)
    a = cv2.cvtColor(cv2.resize(gt_bgr, size), cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(cv2.resize(fake_bgr, size), cv2.COLOR_BGR2RGB)
    a = torch.from_numpy(a.copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    b = torch.from_numpy(b.copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    with torch.inference_mode():
        return float(metric(a, b))


def analyze(video, lpips_metric=None):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open {video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count not in (176, 192):
        raise ValueError(f"expected 176/192 generated frames, got {frame_count}: {video}")
    per_frame = []
    prev_gt = prev_fake = None
    while True:
        ok, joined = cap.read()
        if not ok:
            break
        if joined.shape[0] % 2:
            raise ValueError(f"odd video height: {video}")
        gt, fake = np.split(joined, 2, axis=0)
        row = frame_stats(gt, fake)
        if lpips_metric is not None:
            row["lpips_512x64"] = _lpips_frame(gt, fake, lpips_metric)
        gray_gt = row.pop("gt_small_gray")
        gray_fake = row.pop("fake_small_gray")
        row["gt_temporal_l1"] = (None if prev_gt is None else
                                  float(np.abs(gray_gt - prev_gt).mean()))
        row["fake_temporal_l1"] = (None if prev_fake is None else
                                    float(np.abs(gray_fake - prev_fake).mean()))
        prev_gt, prev_fake = gray_gt, gray_fake
        row["generated_frame"] = len(per_frame) + 3
        per_frame.append(row)
    cap.release()
    if len(per_frame) != frame_count:
        raise RuntimeError(f"decoded {len(per_frame)} of {frame_count}: {video}")

    scalar_keys = [key for key, value in per_frame[1].items()
                   if isinstance(value, (float, int)) and key != "generated_frame"]
    windows = []
    for start in range(0, frame_count, 16):
        chunk = per_frame[start:start + 16]
        out = {"window": start // 16 + 1,
               "frames": [chunk[0]["generated_frame"], chunk[-1]["generated_frame"]]}
        for key in scalar_keys:
            vals = [row[key] for row in chunk if row[key] is not None]
            out[key] = float(np.mean(vals)) if vals else None
        for key in ("gt_rgb_mean", "fake_rgb_mean", "gt_rgb_std", "fake_rgb_std"):
            out[key] = np.mean([row[key] for row in chunk], axis=0).tolist()
        out["edge_energy_ratio"] = out["fake_edge_energy"] / max(out["gt_edge_energy"], 1e-12)
        out["temporal_l1_ratio"] = out["fake_temporal_l1"] / max(out["gt_temporal_l1"], 1e-12)
        windows.append(out)
    return {"video": str(video), "frame_count": frame_count, "windows": windows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="baseline")
    parser.add_argument("--lpips", action="store_true",
                        help="compute AlexNet LPIPS on 512x64 resized MP4 frames")
    args = parser.parse_args()
    paths = sorted(args.input_root.glob(f"max195_{args.arm}_scene*_seed42/gt_vs_generated.mp4"))
    if not paths:
        raise FileNotFoundError(f"no 195-frame {args.arm} videos in {args.input_root}")
    lpips_metric = None
    if args.lpips:
        import lpips
        import torch
        torch.set_num_threads(4)
        lpips_metric = lpips.LPIPS(net="alex", verbose=False).eval()
    records = [analyze(path, lpips_metric) for path in paths]
    aggregates = []
    for window in range(1, 13):
        rows = [record["windows"][window - 1] for record in records]
        keys = ("rgb_mean_l1", "luma_hist_js", "edge_energy_ratio",
                "temporal_l1_ratio", "mp4_psnr_db", "gt_saturation",
                "fake_saturation")
        if args.lpips:
            keys += ("lpips_512x64",)
        aggregates.append({"window": window, **{
            key: float(np.mean([row[key] for row in rows])) for key in keys}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"note": "Lossy MP4 diagnostics; not official metrics",
                                        "arm": args.arm, "scenes": records,
                                        "window_means": aggregates}, indent=2) + "\n")
    print(args.output)
    for row in aggregates:
        print(row)


if __name__ == "__main__":
    main()
