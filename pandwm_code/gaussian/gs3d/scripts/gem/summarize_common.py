#!/usr/bin/env python3
"""Summarize the three official GEM common149 seed evaluations."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


EXPECTED_SEEDS = (0, 42, 1000)


def parse_seed_path(value: str):
    seed, separator, path = value.partition("=")
    if not separator or not path:
        raise argparse.ArgumentTypeError("expected SEED=PATH")
    return int(seed), Path(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", type=parse_seed_path,
                        required=True)
    parser.add_argument("--real-vs-real", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    return parser.parse_args()


def metric_summary(rows, key):
    values = [float(row[key]) for row in rows]
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "values": values,
    }


def fmt(summary):
    return "{:.3f} ± {:.3f}".format(
        summary["mean"], summary["sample_std"])


def main():
    args = parse_args()
    rows = []
    for seed, path in args.result:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("clips", -1)) != 149:
            raise ValueError(f"{path} is not a 149-clip result")
        if int(payload.get("frames", -1)) != 16:
            raise ValueError(f"{path} is not a 16-frame result")
        if int(payload.get("generation_seed", -1)) != seed:
            raise ValueError(f"seed mismatch in {path}")
        rows.append({
            "seed": seed,
            "fid": float(payload["fid"]),
            "fvd": float(payload["fvd"]),
            "source": str(path.resolve()),
        })
    rows.sort(key=lambda row: row["seed"])
    if tuple(row["seed"] for row in rows) != EXPECTED_SEEDS:
        raise ValueError("expected exactly seeds 0, 42, 1000")

    real_vs_real = json.loads(args.real_vs_real.read_text(encoding="utf-8"))
    if (int(real_vs_real.get("source_clips", -1)) != 149
            or int(real_vs_real.get("frames", -1)) != 16):
        raise ValueError("real-vs-real file uses a different target protocol")

    payload = {
        "method": "GEM",
        "official_repository_commit": (
            "f8dae58270dca14d2646d4833391856fefd5e488"),
        "official_checkpoint_sha256": (
            "0c243cddd1bad88cfd20ce367724b15266692ecbf37e5effbfb859493ff04c0d"),
        "protocol": {
            "clips": 149,
            "camera": "CAM_FRONT band",
            "condition_frames": 1,
            "official_generated_frames": 25,
            "official_sampling_steps": 50,
            "scored_generated_indices": [1, 16],
            "ground_truth_common_offsets": [10, 25],
            "scored_frames": 16,
            "resolution": [576, 1024],
            "generation_seeds": list(EXPECTED_SEEDS),
        },
        "per_seed": rows,
        "fid": metric_summary(rows, "fid"),
        "fvd": metric_summary(rows, "fvd"),
        "real_vs_real": real_vs_real,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown = (
        "| Method | Seeds | FID ↓ | FVD ↓ |\n"
        "| --- | --- | ---: | ---: |\n"
        f"| GEM (official, 50-step) | 0, 42, 1000 | "
        f"{fmt(payload['fid'])} | {fmt(payload['fvd'])} |\n"
    )
    args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
