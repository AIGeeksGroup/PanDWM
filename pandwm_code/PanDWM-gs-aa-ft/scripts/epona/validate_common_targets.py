"""Verify that PanDWM and canonical dumps score the same 16 GT frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epona-manifest", required=True)
    parser.add_argument("--canonical-real", required=True)
    parser.add_argument("--pandwm-real", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-abs-difference", type=int, default=1)
    parser.add_argument("--max-difference-fraction", type=float, default=1e-5)
    return parser.parse_args()


def load_uint8(path: Path, expected_frames: int):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.dtype != torch.uint8 or value.ndim != 4 or value.shape[1] != 3:
        raise ValueError("invalid uint8 video {}: {} {}".format(
            path, tuple(value.shape), value.dtype))
    if value.shape[0] != expected_frames:
        raise ValueError("{} has {} frames, expected {}".format(
            path, value.shape[0], expected_frames))
    return value


def main():
    args = parse_args()
    manifest = json.loads(Path(args.epona_manifest).read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    if len(records) != 149:
        raise ValueError("expected 149 Epona manifest records")
    canonical_dir = Path(args.canonical_real)
    pandwm_dir = Path(args.pandwm_real)
    total_values = 0
    different_values = 0
    maximum = 0
    per_clip = []
    for record in sorted(records, key=lambda row: int(row["clip_index"])):
        index = int(record["clip_index"])
        canonical_paths = list(canonical_dir.glob("{}_*.pt".format(index)))
        if len(canonical_paths) != 1:
            raise ValueError("canonical index {} matched {} files".format(
                index, len(canonical_paths)))
        anchor = int(record["timestamps_us"][0])
        pandwm_path = pandwm_dir / "{}.pt".format(anchor)
        canonical = load_uint8(canonical_paths[0], 35)[10:26]
        pandwm = load_uint8(pandwm_path, 32)[7:23]
        difference = canonical.to(torch.int16) - pandwm.to(torch.int16)
        clip_different = int((difference != 0).sum())
        clip_maximum = int(difference.abs().max())
        total_values += difference.numel()
        different_values += clip_different
        maximum = max(maximum, clip_maximum)
        per_clip.append({
            "clip_index": index,
            "anchor_timestamp_us": anchor,
            "maximum_absolute_uint8_difference": clip_maximum,
            "different_channel_values": clip_different,
            "total_channel_values": difference.numel(),
        })
    fraction = different_values / total_values
    if (maximum > args.max_abs_difference
            or fraction > args.max_difference_fraction):
        raise ValueError(
            "GT mismatch: max abs {}, fraction {:.9g}".format(
                maximum, fraction))
    output = {
        "protocol": "canonical[10:26] vs PanDWM-future32[7:23]",
        "clips": 149,
        "frames": 16,
        "maximum_absolute_uint8_difference": maximum,
        "different_channel_values": different_values,
        "total_channel_values": total_values,
        "different_fraction": fraction,
        "thresholds": {
            "maximum_absolute_uint8_difference": args.max_abs_difference,
            "different_fraction": args.max_difference_fraction,
        },
        "per_clip": per_clip,
    }
    Path(args.out).write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: output[key] for key in (
            "clips", "frames", "maximum_absolute_uint8_difference",
            "different_channel_values", "total_channel_values",
            "different_fraction")
    }, indent=2))


if __name__ == "__main__":
    main()
