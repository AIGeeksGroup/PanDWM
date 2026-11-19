"""Score lossless HorizonDrive dumps with PanDWM's CAM_FRONT-band metric.

This intentionally imports the existing TATS-I3D implementation instead of
the StyleGAN-V FVD bundled with HorizonDrive.  A result from this script is
comparable only to a PanDWM nusc6band per-camera score produced with the same
frame count, sample count, I3D checkpoint and uint8 quantisation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dwm_pano.metrics.fvd_pano import PanoPatchFrechetVideoDistance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True,
                        help="directory containing real/*.pt and fake/*.pt")
    parser.add_argument("--i3d", required=True)
    parser.add_argument("--condition-frames", type=int, default=3)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--expected-clips", type=int, default=149)
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def paired_files(root: Path, expected: int):
    real = {path.name: path for path in (root / "real").glob("*.pt")}
    fake = {path.name: path for path in (root / "fake").glob("*.pt")}
    if set(real) != set(fake):
        raise ValueError(
            "real/fake dump sets differ: real-only={} fake-only={}".format(
                sorted(set(real) - set(fake))[:5],
                sorted(set(fake) - set(real))[:5]))
    if len(real) != expected:
        raise ValueError(
            "expected {} paired clips, found {}".format(expected, len(real)))
    return [(real[name], fake[name]) for name in sorted(real)]


def load_future(path: Path, start: int, frames: int):
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError("{} must be [T,3,H,W], got {}".format(
            path, tuple(tensor.shape)))
    stop = start + frames
    if tensor.shape[0] < stop:
        raise ValueError(
            "{} has {} frames, need at least {}".format(
                path, tensor.shape[0], stop))
    if tensor.dtype != torch.uint8:
        raise TypeError("{} must be uint8, got {}".format(path, tensor.dtype))
    return tensor[start:stop].to(torch.float32).div_(255)


@torch.no_grad()
def main():
    args = parse_args()
    pairs = paired_files(Path(args.dump), args.expected_clips)
    # _accumulate_patches is the single shared normalize/resize/I3D path used
    # by both PanoPatch-FVD and nusc6band.  Supplying one patch here means one
    # CAM_FRONT-band sample per clip, with no panorama column slicing.
    metric = PanoPatchFrechetVideoDistance(
        inception_3d_checkpoint_path=args.i3d,
        sequence_count=-1).to(args.device)
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(args.device)
    for index, (real_path, fake_path) in enumerate(pairs):
        real = load_future(
            real_path, args.condition_frames, args.frames)[None].to(args.device)
        fake = load_future(
            fake_path, args.condition_frames, args.frames)[None].to(args.device)
        metric._accumulate_patches([real], real=True)
        metric._accumulate_patches([fake], real=False)
        fid.update(real.flatten(0, 1), real=True)
        fid.update(fake.flatten(0, 1), real=False)
        if index % 10 == 0:
            print("clips {}/{}".format(index + 1, len(pairs)), flush=True)

    value = float(metric.compute())
    fid_value = float(fid.compute())
    samples = int(metric.real_features_num_samples)
    label = "FVD-I3D-CAM_FRONT-band-{}-n{}-uint8".format(
        args.frames, samples)
    payload = {
        "metric_label": label,
        "fvd": value,
        "fid": fid_value,
        "clips": len(pairs),
        "generation_seed": args.generation_seed,
        "samples": samples,
        "condition_frames": args.condition_frames,
        "frames": args.frames,
        "i3d_checkpoint": str(Path(args.i3d).resolve()),
        "dump": str(Path(args.dump).resolve()),
        "quantization": "round(clamp(x,0,1)*255)/255",
    }
    print("{} = {:.3f}".format(label, value))
    print("FID-CAM_FRONT-band-n{}-uint8 = {:.3f}".format(
        len(pairs) * args.frames, fid_value))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
