"""Stitching the two panorama halves into one 360-degree canvas.

Split out of the pipeline so it carries no diffusers/transformers import: it
is a pure tensor operation on the frozen evaluation path and wants tests that
run anywhere.
"""
import math
import os

import torch


# Pano dedup width: the two halves cover 201.6 degrees each, so they overlap
# by 43.2 degrees = round(1152 / 5376 * 2304) = 494 columns of the resized
# half. Stitching drops the right half's leftmost 494 columns.
PANO_OVERLAP_W = 494


def _pano_blend_width():
    """Columns of raised-cosine cross-fade to apply inside the overlap.

    0 keeps the historical hard butt-joint, so every frozen number stays
    bit-reproducible unless this is set on purpose. Capped at the overlap
    width because there is no redundancy outside it -- the wrap joint has
    none at all, which is why only the left/right seam can be blended.
    """
    try:
        width = int(os.environ.get("PANDWM_PANO_BLEND_W", "0"))
    except ValueError:
        raise ValueError(
            "PANDWM_PANO_BLEND_W must be an integer column count")
    if width < 0:
        raise ValueError("PANDWM_PANO_BLEND_W must be >= 0")
    return min(width, PANO_OVERLAP_W)


def stitch_pano_halves(x: torch.Tensor, blend_width: int = None):
    """Stitch the V=2 halves into one V=1 360-degree panorama.

    ``x``: ``(B, T, V=2, C, H, W)`` -> ``(B, T, V=1, C, H, 2W - 494)``.

    The output column-to-azimuth mapping is identical for any blend width,
    so FVD patch boundaries, instance box projection and the seam profile all
    stay comparable across arms.

    With ``blend_width = 0`` this is the historical hard cut: the right
    half's first 494 columns are discarded outright even though they image
    the same 43.2 degrees the left half's last 494 columns do. That thrown
    away redundancy is what the cross-fade spends: the joint is spread over
    ``blend_width`` columns instead of landing on one.
    """
    if blend_width is None:
        blend_width = _pano_blend_width()

    left, right = x[:, :, 0], x[:, :, 1]
    if blend_width <= 0:
        return torch.cat(
            [left, right[..., PANO_OVERLAP_W:]], dim=-1).unsqueeze(2)

    # Cross-fade over the last `blend_width` columns of the left half, which
    # image the same azimuths as the right half's columns
    # [PANO_OVERLAP_W - blend_width, PANO_OVERLAP_W).
    keep = left.shape[-1] - blend_width
    ramp = torch.linspace(
        0.0, 1.0, blend_width + 2, device=x.device, dtype=torch.float32)[1:-1]
    ramp = (0.5 - 0.5 * torch.cos(math.pi * ramp)).to(dtype=x.dtype)
    right_overlap = right[
        ..., PANO_OVERLAP_W - blend_width:PANO_OVERLAP_W]
    blended = (
        left[..., keep:] * (1.0 - ramp) + right_overlap * ramp)
    return torch.cat(
        [left[..., :keep], blended, right[..., PANO_OVERLAP_W:]],
        dim=-1).unsqueeze(2)
