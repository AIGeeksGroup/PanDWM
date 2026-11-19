#!/usr/bin/env python3
"""Prepare an isolated ZeroShot input directory for official GEM sampling."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    if not (0 <= args.start < args.end <= 149):
        raise ValueError(f"invalid common149 range [{args.start}, {args.end})")
    if args.input_dir.exists():
        raise FileExistsError(args.input_dir)
    args.input_dir.mkdir(parents=True)

    expected_names = []
    for index in range(args.start, args.end):
        source = args.common_root / f"clip_{index:04d}" / "00.jpg"
        if not source.is_file():
            raise FileNotFoundError(source)
        name = f"clip_{index:04d}.jpg"
        (args.input_dir / name).symlink_to(source)
        expected_names.append(name)

    # GEM's official ZeroShot reader uses os.listdir without sorting. Record the
    # exact order that maps its local output index back to the common clip id.
    observed_names = os.listdir(args.input_dir)
    if sorted(observed_names) != expected_names:
        raise RuntimeError("prepared GEM inputs do not match the requested range")
    records = [
        {
            "gem_output_index": local_index,
            "input_name": name,
            "common_clip_index": int(name[5:9]),
            "condition_path": str((args.input_dir / name).resolve()),
        }
        for local_index, name in enumerate(observed_names)
    ]
    payload = {
        "protocol": "PanDWM-common149-GEM-official-ZeroShot",
        "range": [args.start, args.end],
        "count": len(records),
        "condition_common_offset": 9,
        "generated_indices_scored": [1, 16],
        "ground_truth_common_offsets": [10, 25],
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
