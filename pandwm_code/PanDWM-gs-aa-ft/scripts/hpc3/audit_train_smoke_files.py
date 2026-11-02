#!/usr/bin/env python3
"""Check an HPC3 train subset against its h20-generated manifest.

Size equality catches interrupted transfers but not a source JPEG whose byte
count is correct and compressed stream is already truncated.  The optional
decode pass therefore performs the same RGB load that the training dataset
uses before a production job is admitted.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import sys


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def decode_image(item):
    root_text, relative = item
    try:
        from PIL import Image

        with Image.open(Path(root_text) / relative) as image:
            image.convert("RGB").load()
        return relative, None
    except Exception as error:  # report corrupt inputs instead of hiding them
        return relative, "{}: {}".format(type(error).__name__, error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument("--decode-images", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    missing = []
    size_mismatch = []
    image_paths = []
    expected_count = 0
    expected_bytes = 0
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            relative, size_text = line.rstrip("\n").split("\t")
            expected_size = int(size_text)
            expected_count += 1
            expected_bytes += expected_size
            path = args.root / relative
            if path.suffix.lower() in IMAGE_SUFFIXES:
                image_paths.append(relative)
            try:
                actual_size = path.stat().st_size
            except FileNotFoundError:
                missing.append(relative)
                continue
            if actual_size != expected_size:
                size_mismatch.append({
                    "path": relative,
                    "expected": expected_size,
                    "actual": actual_size,
                })

    decode_errors = []
    decoded_image_count = 0
    if args.decode_images and not missing and not size_mismatch:
        work = ((str(args.root), relative) for relative in image_paths)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers) as executor:
            for decoded_image_count, (relative, error) in enumerate(
                    executor.map(decode_image, work, chunksize=16), 1):
                if error is not None:
                    decode_errors.append({"path": relative, "error": error})
                if decoded_image_count % 1000 == 0:
                    print("decoded {}/{} images".format(
                        decoded_image_count, len(image_paths)),
                        file=sys.stderr, flush=True)

    payload = {
        "expected_count": expected_count,
        "expected_bytes": expected_bytes,
        "missing_count": len(missing),
        "size_mismatch_count": len(size_mismatch),
        "decode_images_requested": args.decode_images,
        "decoded_image_count": decoded_image_count,
        "decode_error_count": len(decode_errors),
        "valid": not missing and not size_mismatch and not decode_errors,
        "missing": missing,
        "size_mismatch": size_mismatch,
        "decode_errors": decode_errors,
    }
    print(json.dumps({
        **payload,
        "missing": missing[:args.show],
        "size_mismatch": size_mismatch[:args.show],
        "decode_errors": decode_errors[:args.show],
    }, ensure_ascii=False, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    raise SystemExit(0 if payload["valid"] else 1)


if __name__ == "__main__":
    main()
