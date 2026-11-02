"""Build a paper-ready common-protocol table from completed seed runs."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def parse_seed_path(value: str):
    seed, separator, path = value.partition("=")
    if not separator or not path:
        raise argparse.ArgumentTypeError("expected SEED=PATH")
    return int(seed), Path(path)


def parse_pandwm_log(text: str):
    fid_matches = re.findall(r"front_fid:\s*([0-9.eE+-]+)", text)
    per_camera_lines = re.findall(
        r"per-camera n6bridge:\s*([^\n\r]+)", text)
    if not fid_matches or not per_camera_lines:
        raise ValueError("PanDWM log has no completed front metrics")
    camera_matches = dict(re.findall(
        r"(CAM_[A-Z_]+)=([0-9.eE+-]+)", per_camera_lines[-1]))
    if "CAM_FRONT" not in camera_matches:
        raise ValueError("PanDWM per-camera line has no CAM_FRONT value")
    pooled_matches = re.findall(
        r"(?:pooled_fvd|front_fvd):\s*([0-9.eE+-]+)", text)
    return {
        "fid": float(fid_matches[-1]),
        "fvd": float(camera_matches["CAM_FRONT"]),
        "pooled_six_camera_fvd": (
            float(pooled_matches[-1]) if pooled_matches else None),
    }


def metric_summary(rows):
    output = {}
    for key in ("fid", "fvd"):
        values = [float(row[key]) for row in rows]
        output[key] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizon", action="append", type=parse_seed_path, required=True,
        help="repeat as SEED=common_front_fvd.json")
    parser.add_argument(
        "--pandwm", action="append", type=parse_seed_path, required=True,
        help="repeat as SEED=pandwm_common_JOB.log")
    parser.add_argument("--real-vs-real")
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out")
    return parser.parse_args()


def fmt_metric(summary):
    return "{:.3f} ± {:.3f}".format(
        summary["mean"], summary["sample_std"])


def main():
    args = parse_args()
    horizon = []
    for seed, path in args.horizon:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("clips") != 149 or payload.get("frames") != 32:
            raise ValueError("{} is not a 149×32 common result".format(path))
        recorded_seed = payload.get("generation_seed")
        if recorded_seed is not None and int(recorded_seed) != seed:
            raise ValueError("{} records seed {}, CLI says {}".format(
                path, recorded_seed, seed))
        horizon.append({
            "seed": seed, "fid": float(payload["fid"]),
            "fvd": float(payload["fvd"]), "source": str(path),
        })

    pandwm = []
    for seed, path in args.pandwm:
        metrics = parse_pandwm_log(path.read_text(encoding="utf-8"))
        pandwm.append({"seed": seed, "source": str(path), **metrics})

    horizon.sort(key=lambda row: row["seed"])
    pandwm.sort(key=lambda row: row["seed"])
    horizon_seeds = [row["seed"] for row in horizon]
    pandwm_seeds = [row["seed"] for row in pandwm]
    if horizon_seeds != pandwm_seeds:
        raise ValueError("seed sets differ: HorizonDrive={} PanDWM={}".format(
            horizon_seeds, pandwm_seeds))

    payload = {
        "protocol": {
            "clips": 149, "context_frames": 3, "prediction_frames": 32,
            "camera": "CAM_FRONT-band", "resolution": [384, 768],
            "quantization": "uint8 round", "seed_set": horizon_seeds,
        },
        "HorizonDrive": {
            "per_seed": horizon, **metric_summary(horizon)},
        "PanDWM": {"per_seed": pandwm, **metric_summary(pandwm)},
    }
    if args.real_vs_real:
        payload["real_vs_real"] = json.loads(
            Path(args.real_vs_real).read_text(encoding="utf-8"))

    Path(args.out).write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    markdown = [
        "| Method | Seeds | FID ↓ | FVD ↓ |",
        "| --- | --- | ---: | ---: |",
    ]
    for name in ("HorizonDrive", "PanDWM"):
        row = payload[name]
        markdown.append("| {} | {} | {} | {} |".format(
            name, ", ".join(map(str, horizon_seeds)),
            fmt_metric(row["fid"]), fmt_metric(row["fvd"])))
    markdown_text = "\n".join(markdown) + "\n"
    print(markdown_text, end="")
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown_text, encoding="utf-8")


if __name__ == "__main__":
    main()
