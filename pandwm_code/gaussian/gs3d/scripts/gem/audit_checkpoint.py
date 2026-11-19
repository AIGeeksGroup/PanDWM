#!/usr/bin/env python3
"""Audit official GEM checkpoint coverage without changing upstream loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from safetensors import safe_open

from gem.util import instantiate_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)
    model_state = model.state_dict()
    model_shapes = {key: list(value.shape) for key, value in model_state.items()}
    with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
        checkpoint_shapes = {
            key: list(checkpoint.get_slice(key).get_shape())
            for key in checkpoint.keys()
        }
    missing = sorted(set(model_shapes) - set(checkpoint_shapes))
    unexpected = sorted(set(checkpoint_shapes) - set(model_shapes))
    shape_mismatches = {
        key: {"model": model_shapes[key], "checkpoint": checkpoint_shapes[key]}
        for key in sorted(set(model_shapes) & set(checkpoint_shapes))
        if model_shapes[key] != checkpoint_shapes[key]
    }
    default_affine_missing = []
    nondefault_missing = []
    for key in missing:
        value = model_state[key]
        expected = 1.0 if key.endswith(".weight") else 0.0
        is_temporal_norm = (
            key.startswith("first_stage_model.decoder.")
            and ".time_stack." in key
            and (".in_layers.0." in key or ".out_layers.0." in key)
            and (key.endswith(".weight") or key.endswith(".bias"))
        )
        if is_temporal_norm and bool((value == expected).all()):
            default_affine_missing.append(key)
        else:
            nondefault_missing.append(key)
    payload = {
        "model_keys": len(model_shapes),
        "checkpoint_keys": len(checkpoint_shapes),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "shape_mismatch_count": len(shape_mismatches),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
        "default_temporal_groupnorm_affine_count": len(default_affine_missing),
        "nondefault_missing_count": len(nondefault_missing),
        "nondefault_missing": nondefault_missing,
        "upstream_strict_false_load_is_deterministic": not (
            unexpected or shape_mismatches or nondefault_missing
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    if unexpected or shape_mismatches or nondefault_missing:
        raise RuntimeError("official GEM checkpoint has unsafe compatibility differences")


if __name__ == "__main__":
    main()
