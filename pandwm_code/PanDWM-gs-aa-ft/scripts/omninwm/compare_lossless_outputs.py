#!/usr/bin/env python3
"""Compare two lossless uint8 OmniNWM outputs from different SP layouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    reference = torch.load(
        args.reference, map_location="cpu", weights_only=True)
    candidate = torch.load(
        args.candidate, map_location="cpu", weights_only=True)
    if reference.shape != candidate.shape:
        raise ValueError((tuple(reference.shape), tuple(candidate.shape)))
    if reference.dtype != torch.uint8 or candidate.dtype != torch.uint8:
        raise TypeError((reference.dtype, candidate.dtype))

    delta = reference.to(torch.int16) - candidate.to(torch.int16)
    squared = delta.to(torch.float64).square()
    mse = squared.mean().item()
    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "shape": list(reference.shape),
        "exact_equal": bool(torch.equal(reference, candidate)),
        "different_values": int(torch.count_nonzero(delta).item()),
        "total_values": int(delta.numel()),
        "mae_uint8": float(delta.abs().to(torch.float64).mean().item()),
        "max_abs_uint8": int(delta.abs().max().item()),
        "mse_uint8": float(mse),
        "psnr_db": math.inf if mse == 0 else 10 * math.log10(255**2 / mse),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
