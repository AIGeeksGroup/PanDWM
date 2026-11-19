"""Estimate the finite-sample real-vs-real floor on the common protocol.

The 149 frozen validation clips cannot provide two independent sets of 149.
Instead, each repeat makes a deterministic random 74/74 clip-disjoint split
and omits the remaining clip.  The resulting number is therefore labelled
``n=74`` and must not be subtracted from the model-vs-real ``n=149`` score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--i3d", required=True)
    parser.add_argument("--condition-frames", type=int, default=3)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--expected-clips", type=int, default=149)
    parser.add_argument("--split-size", type=int, default=74)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=20260909)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_future(path: Path, start: int, frames: int):
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError("{} must be [T,3,H,W], got {}".format(
            path, tuple(tensor.shape)))
    stop = start + frames
    if tensor.shape[0] < stop:
        raise ValueError("{} has {} frames, need at least {}".format(
            path, tensor.shape[0], stop))
    if tensor.dtype != torch.uint8:
        raise TypeError("{} must be uint8, got {}".format(path, tensor.dtype))
    return tensor[start:stop]


def moments(features: torch.Tensor):
    """Return float64 mean/covariance using the evaluation metric formula."""
    values = features.double()
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T.mm(centered) / (values.shape[0] - 1)
    return mean, covariance


def frechet_from_features(first: torch.Tensor, second: torch.Tensor):
    from dwm_pano.metrics.fvd_pano import _compute_fid

    mean_first, cov_first = moments(first)
    mean_second, cov_second = moments(second)
    return float(_compute_fid(
        mean_first, cov_first, mean_second, cov_second).float())


def split_indices(count: int, split_size: int, repeats: int, seed: int):
    if split_size < 2:
        raise ValueError("split_size must be at least 2")
    if 2 * split_size > count:
        raise ValueError(
            "two disjoint splits of {} need {} clips, only {} available"
            .format(split_size, 2 * split_size, count))
    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(int(repeats)):
        permutation = torch.randperm(count, generator=generator)
        yield permutation[:split_size], permutation[split_size:2 * split_size]


def summary(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=True)) if len(values) > 1 else 0.0,
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "values": [float(value) for value in values],
    }


@torch.no_grad()
def main():
    args = parse_args()
    paths = sorted(Path(args.real_dir).glob("*.pt"))
    if len(paths) != args.expected_clips:
        raise ValueError("expected {} real clips, found {}".format(
            args.expected_clips, len(paths)))

    # Reuse the exact TATS-I3D module and preprocessing path as common FVD.
    from dwm_pano.metrics.fvd_pano import PanoPatchFrechetVideoDistance
    fvd_metric = PanoPatchFrechetVideoDistance(
        inception_3d_checkpoint_path=args.i3d,
        sequence_count=-1).to(args.device)
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_metric = FrechetInceptionDistance(
        feature=2048, normalize=True).to(args.device)

    fvd_features = []
    fid_features = []
    for index, path in enumerate(paths):
        video_uint8 = load_future(
            path, args.condition_frames, args.frames).to(args.device)
        video = video_uint8.to(torch.float32).div_(255)
        normalized = video * 2 - 1
        resized = torch.nn.functional.interpolate(
            normalized, size=fvd_metric.target_resolution,
            mode="bilinear")[None]
        fvd_features.append(
            fvd_metric.inception(resized.transpose(1, 2))[0].cpu())
        # torchmetrics' normalized public update path scales [0,1] floats
        # back to uint8 before calling this feature module.  We already hold
        # the lossless uint8 interchange, so pass it directly and avoid a
        # float divide/multiply round trip.
        fid_features.append(fid_metric.inception(video_uint8).cpu())
        if index % 10 == 0:
            print("feature clips {}/{}".format(index + 1, len(paths)),
                  flush=True)

    fvd_features = torch.stack(fvd_features).to(args.device)
    fid_features = torch.stack(fid_features).to(args.device)
    fvd_values = []
    fid_values = []
    for repeat, (first, second) in enumerate(split_indices(
            len(paths), args.split_size, args.repeats, args.split_seed)):
        first = first.to(args.device)
        second = second.to(args.device)
        fvd_values.append(frechet_from_features(
            fvd_features[first], fvd_features[second]))
        fid_values.append(frechet_from_features(
            fid_features[first].flatten(0, 1),
            fid_features[second].flatten(0, 1)))
        print("split {}/{} FVD={:.3f} FID={:.3f}".format(
            repeat + 1, args.repeats, fvd_values[-1], fid_values[-1]),
            flush=True)

    payload = {
        "protocol": "clip-disjoint real-vs-real random half split",
        "metric_label": (
            "R2R-CAM_FRONT-band-{}-n{}-uint8".format(
                args.frames, args.split_size)),
        "source_clips": len(paths),
        "split_size_per_side": args.split_size,
        "omitted_clips_per_repeat": len(paths) - 2 * args.split_size,
        "repeats": args.repeats,
        "split_seed": args.split_seed,
        "condition_frames": args.condition_frames,
        "frames": args.frames,
        "i3d_checkpoint": str(Path(args.i3d).resolve()),
        "real_dir": str(Path(args.real_dir).resolve()),
        "quantization": "round(clamp(x,0,1)*255)/255",
        "fvd": summary(fvd_values),
        "fid": summary(fid_values),
        "warning": (
            "This n=74 finite-sample floor is descriptive and must not be "
            "subtracted from n=149 model-vs-real scores."),
    }
    print(json.dumps(payload, indent=1))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
