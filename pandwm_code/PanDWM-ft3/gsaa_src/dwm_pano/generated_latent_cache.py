"""Utilities for an offline Stage-2 latent cache used by GS adaptation.

The cache stores only the frame latents selected by the GS training sampler.
Keeping Stage-2 generation offline avoids co-locating the 13 GiB diffusion
checkpoint with the Gaussian decoder while still exposing the decoder to the
actual four-step student distribution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


CACHE_SCHEMA_VERSION = 1


def scalar_int(value: Any) -> int:
    """Read a scalar from a collated tensor/list/Python value."""
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return scalar_int(value[0])
    return int(value)


def latent_cache_path(
    cache_dir: str | Path,
    dataset_index: int,
    anchor_timestamp_us: int,
) -> Path:
    """Return the collision-resistant path for one training clip."""
    return Path(cache_dir) / (
        f"{int(dataset_index):08d}_{int(anchor_timestamp_us)}.pt"
    )


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    """Write a torch payload without exposing a partially written cache item."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def make_cache_payload(
    *,
    dataset_index: int,
    anchor_timestamp_us: int,
    frame_indices: torch.Tensor,
    frame_timestamps_us: torch.Tensor,
    latents: torch.Tensor,
) -> dict[str, Any]:
    """Build the portable, CPU-only cache record for one GS optimizer step."""
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "latent_source": "stage2_4step_ema",
        "dataset_index": int(dataset_index),
        "anchor_timestamp_us": int(anchor_timestamp_us),
        "frame_indices": frame_indices.detach().cpu().long().contiguous(),
        "frame_timestamps_us": frame_timestamps_us.detach().cpu().long()
        .contiguous(),
        "latents": latents.detach().cpu().half().contiguous(),
    }


def validate_cache_payload(
    payload: dict[str, Any],
    *,
    dataset_index: int,
    anchor_timestamp_us: int,
    frame_indices: torch.Tensor,
    frame_timestamps_us: torch.Tensor,
) -> torch.Tensor:
    """Validate identity, frame selection and shape, then return latents."""
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported generated-latent cache schema: {}".format(
                payload.get("schema_version")
            )
        )
    expected_scalars = {
        "dataset_index": int(dataset_index),
        "anchor_timestamp_us": int(anchor_timestamp_us),
    }
    for key, expected in expected_scalars.items():
        if int(payload.get(key, -1)) != expected:
            raise RuntimeError(
                f"Generated-latent cache {key} mismatch: "
                f"expected {expected}, got {payload.get(key)!r}"
            )

    expected_indices = frame_indices.detach().cpu().long()
    expected_timestamps = frame_timestamps_us.detach().cpu().long()
    cached_indices = payload.get("frame_indices")
    cached_timestamps = payload.get("frame_timestamps_us")
    if not isinstance(cached_indices, torch.Tensor) or not torch.equal(
        cached_indices.long(), expected_indices
    ):
        raise RuntimeError(
            "Generated-latent cache frame indices do not match the active "
            "GS sampler"
        )
    if not isinstance(cached_timestamps, torch.Tensor) or not torch.equal(
        cached_timestamps.long(), expected_timestamps
    ):
        raise RuntimeError(
            "Generated-latent cache timestamps do not match the active clip"
        )

    latents = payload.get("latents")
    expected_prefix = (len(expected_indices), 2)
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5 \
            or tuple(latents.shape[:2]) != expected_prefix:
        raise RuntimeError(
            "Generated-latent cache must contain [F,2,C,H,W], got {}".format(
                None if not isinstance(latents, torch.Tensor)
                else tuple(latents.shape)
            )
        )
    if not latents.is_floating_point() or not torch.isfinite(latents).all():
        raise RuntimeError("Generated-latent cache contains invalid values")
    return latents
