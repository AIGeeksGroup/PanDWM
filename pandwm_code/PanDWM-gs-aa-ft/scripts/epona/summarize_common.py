"""Summarize completed 16-frame common-protocol JSON results."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_SEEDS = {0, 42, 1000}


def parse_result(value: str):
    identity, separator, path = value.partition("=")
    method, colon, seed = identity.rpartition(":")
    if not separator or not colon or not method or not seed or not path:
        raise argparse.ArgumentTypeError("expected METHOD:SEED=PATH")
    return method, int(seed), Path(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", type=parse_result,
                        required=True)
    parser.add_argument("--real-vs-real", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out", required=True)
    return parser.parse_args()


def summarize(rows):
    result = {"per_seed": rows}
    for metric in ("fid", "fvd"):
        values = [row[metric] for row in rows]
        result[metric] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values),
            "values": values,
        }
    return result


def formatted(summary):
    return "{:.3f} ± {:.3f}".format(
        summary["mean"], summary["sample_std"])


def main():
    args = parse_args()
    grouped = defaultdict(list)
    for method, seed, path in args.result:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("clips", -1)) != 149:
            raise ValueError("{} is not a 149-clip result".format(path))
        if int(payload.get("frames", -1)) != 16:
            raise ValueError("{} is not a 16-frame result".format(path))
        recorded_seed = payload.get("generation_seed")
        if recorded_seed is None or int(recorded_seed) != seed:
            raise ValueError(
                "{} records seed {}, CLI says {}".format(
                    path, recorded_seed, seed))
        grouped[method].append({
            "seed": seed,
            "fid": float(payload["fid"]),
            "fvd": float(payload["fvd"]),
            "dump_offset": int(payload["condition_frames"]),
            "source": str(path.resolve()),
        })

    summaries = {}
    for method, rows in grouped.items():
        rows.sort(key=lambda row: row["seed"])
        seeds = {row["seed"] for row in rows}
        if seeds != EXPECTED_SEEDS:
            raise ValueError(
                "{} seed set is {}, expected {}".format(
                    method, sorted(seeds), sorted(EXPECTED_SEEDS)))
        summaries[method] = summarize(rows)

    r2r = json.loads(Path(args.real_vs_real).read_text(encoding="utf-8"))
    if (int(r2r.get("source_clips", -1)) != 149
            or int(r2r.get("frames", -1)) != 16):
        raise ValueError("real-vs-real file uses a different protocol")
    payload = {
        "protocol": {
            "clips": 149,
            "fps": 10,
            "target_frames_in_full_sequence": [10, 25],
            "target_frame_count": 16,
            "camera": "CAM_FRONT band",
            "resolution": [384, 768],
            "quantization": "round(clamp(x,0,1)*255)",
            "generation_seeds": sorted(EXPECTED_SEEDS),
            "conditioning_disclosure": (
                "Epona uses 10 history frames and 100 sampling steps per "
                "generated frame. HorizonDrive and PanDWM use 3 history "
                "frames and 4 denoising steps. Targets, GT, and metrics are "
                "identical; conditioning and compute are not."),
        },
        "methods": dict(sorted(summaries.items())),
        "real_vs_real": r2r,
    }
    Path(args.out).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "| Method | Seeds | FID ↓ | FVD ↓ |",
        "| --- | --- | ---: | ---: |",
    ]
    preferred = ["PanDWM", "HorizonDrive", "Epona"]
    ordered = [name for name in preferred if name in summaries]
    ordered.extend(sorted(set(summaries) - set(ordered)))
    for name in ordered:
        summary = summaries[name]
        lines.append("| {} | 0, 42, 1000 | {} | {} |".format(
            name, formatted(summary["fid"]), formatted(summary["fvd"])))
    markdown = "\n".join(lines) + "\n"
    Path(args.markdown_out).write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
