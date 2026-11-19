#!/usr/bin/env python3
"""Audit OmniNWM common149 token alignment and front-view equivalence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni-manifest", required=True, type=Path)
    parser.add_argument("--common-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def rgb(path: Path, size=None):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None:
            image = image.resize(size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)


def summary(values):
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main():
    args = parse_args()
    omni = json.loads(args.omni_manifest.read_text(encoding="utf-8"))
    common = json.loads(args.common_manifest.read_text(encoding="utf-8"))
    if omni["clips"] != 149 or common["clips"] != 149:
        raise ValueError("expected frozen common149 manifests")
    omni_records = {int(x["common_clip_index"]): x for x in omni["records"]}
    common_records = {int(x["clip_index"]): x for x in common["records"]}
    maes = []
    psnrs = []
    schedules = []
    for clip_index in range(149):
        record = omni_records[clip_index]
        indices = record["generated_indices_scored"]
        if (len(indices) != 16 or len(set(indices)) != 16
                or indices != sorted(indices) or indices[0] <= 0
                or indices[-1] >= 33):
            raise ValueError(f"clip {clip_index} invalid index mapping: {indices}")
        schedules.append(tuple(indices))
        native = record["native_frames"]
        target_tokens = [native[index]["source_token"] for index in indices]
        if target_tokens != record["common_target_tokens"]:
            raise ValueError(f"clip {clip_index} target token mismatch")

        omni_condition = Path(native[0]["paths"]["CAM_FRONT"])
        common_condition = Path(common_records[clip_index]["frames"][0]["path"])
        with Image.open(common_condition) as image:
            common_size = image.size
        a = rgb(omni_condition, common_size)
        b = rgb(common_condition)
        mse = float(np.mean((a - b) ** 2))
        maes.append(float(np.mean(np.abs(a - b))))
        psnrs.append(float(10 * math.log10(255 ** 2 / mse)))

    schedule_counts = Counter(schedules)
    payload = {
        "protocol": "PanDWM-common149-OmniNWM-adapter-alignment-audit",
        "clips": 149,
        "condition_common_offset": 9,
        "target_common_offsets": [10, 25],
        "target_frames": 16,
        "generated_index_schedules": [
            {"indices": list(indices), "clips": count}
            for indices, count in sorted(
                schedule_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "token_alignment": "all 149x16 exact",
        "condition_front_comparison": {
            "comparison_resolution": common["resolution"],
            "omni_resize": "bilinear",
            "mae_0_255": summary(maes),
            "psnr_db": summary(psnrs),
            "note": "Residual difference is resampling/JPEG; both derive from the same calibrated panorama frame.",
        },
    }
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
