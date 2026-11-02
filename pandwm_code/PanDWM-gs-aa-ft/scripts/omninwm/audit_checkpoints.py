#!/usr/bin/env python3
"""Compare official OmniNWM model/VAE checkpoint headers with source models."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import torch
from safetensors import safe_open

from omninwm.models.hunyuan_vae.autoencoder_kl_causal_3d import (
    AutoEncoder3DConfig,
    AutoencoderKLCausal3D,
)
from omninwm.models.mmdit.model import MMDiTConfig, MMDiTModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dit", required=True, type=Path,
                        help="directory containing model.safetensors.index.json")
    parser.add_argument("--vae", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def header_shapes(path: Path):
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: list(handle.get_slice(key).get_shape())
                for key in handle.keys()}


def compare(expected, actual):
    expected_keys = set(expected)
    actual_keys = set(actual)
    common = expected_keys & actual_keys
    return {
        "source_keys": len(expected),
        "checkpoint_keys": len(actual),
        "missing_keys": sorted(expected_keys - actual_keys),
        "unexpected_keys": sorted(actual_keys - expected_keys),
        "shape_mismatches": [
            {"key": key, "source": expected[key], "checkpoint": actual[key]}
            for key in sorted(common)
            if expected[key] != actual[key]
        ],
    }


def main():
    args = parse_args()
    cfg = runpy.run_path(str(args.config))
    model_cfg = dict(cfg["model"])
    for key in ("type", "from_pretrained", "strict_load"):
        model_cfg.pop(key, None)
    model_config = MMDiTConfig(
        from_pretrained="", cache_dir=None, **model_cfg)
    vae_cfg = dict(cfg["ae"])
    for key in ("type", "from_pretrained"):
        vae_cfg.pop(key, None)
    vae_config = AutoEncoder3DConfig(from_pretrained="", **vae_cfg)
    with torch.device("meta"):
        model = MMDiTModel(model_config)
        vae = AutoencoderKLCausal3D(vae_config)
    model_shapes = {key: list(value.shape)
                    for key, value in model.state_dict().items()}
    vae_shapes = {key: list(value.shape)
                  for key, value in vae.state_dict().items()}

    index_path = args.dit / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    dit_shapes = {}
    shard_files = sorted(set(weight_map.values()))
    for shard in shard_files:
        shapes = header_shapes(args.dit / shard)
        overlap = set(dit_shapes) & set(shapes)
        if overlap:
            raise ValueError(f"duplicate keys across shards: {sorted(overlap)[:5]}")
        dit_shapes.update(shapes)
    if set(dit_shapes) != set(weight_map):
        raise ValueError("index weight_map and safetensors headers differ")
    if any(weight_map[key] not in shard_files for key in weight_map):
        raise ValueError("index references an unrecognized shard")

    vae_checkpoint_shapes = header_shapes(args.vae)
    payload = {
        "protocol": "OmniNWM-official-source-checkpoint-shape-audit",
        "config": str(args.config),
        "dit_index": str(index_path),
        "dit_index_metadata": index.get("metadata"),
        "dit_shards": [
            {"name": name, "bytes": (args.dit / name).stat().st_size}
            for name in shard_files
        ],
        "dit_official_strict_load": bool(cfg["model"].get("strict_load", False)),
        "dit": compare(model_shapes, dit_shapes),
        "vae_file": str(args.vae),
        "vae_bytes": args.vae.stat().st_size,
        "vae_official_strict_load": True,
        "vae": compare(vae_shapes, vae_checkpoint_shapes),
    }
    payload["dit_exact_key_match"] = all(
        not payload["dit"][field]
        for field in ("missing_keys", "unexpected_keys", "shape_mismatches"))
    payload["dit_load_compatible"] = (
        not payload["dit"]["missing_keys"]
        and not payload["dit"]["shape_mismatches"]
        and (not payload["dit"]["unexpected_keys"]
             or not payload["dit_official_strict_load"]))
    payload["vae_exact_key_match"] = all(
        not payload["vae"][field]
        for field in ("missing_keys", "unexpected_keys", "shape_mismatches"))
    payload["passed"] = (
        payload["dit_load_compatible"] and payload["vae_exact_key_match"])
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
