"""Run the HorizonDrive evaluator and dump real and generated clips as uint8 tensors.

The upstream evaluator only writes MP4 previews. This entry point wraps the preview
writer and also saves its input tensors, so the metrics see no codec round trip.
Model loading, sampling and rollout are unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from horizondrive.schemas.args import Args
import horizondrive.trainer as trainer_module


def _role_from_path(path: str):
    stem = Path(path).stem
    if stem.endswith("_gt"):
        return "real", stem[:-3]
    ignored = ("_comparison", "_bbox", "_hdmap", "_normal")
    if stem.endswith(ignored):
        return None, None
    return "fake", stem


def _install_lossless_dump():
    dump_root = os.environ.get("HD_COMMON_DUMP_DIR")
    if not dump_root:
        raise RuntimeError("HD_COMMON_DUMP_DIR is required by eval_common.py")
    original = trainer_module.save_multiview_videos_grid

    def wrapped(videos, path, *args, **kwargs):
        result = original(videos, path, *args, **kwargs)
        role, clip_id = _role_from_path(path)
        if role is None:
            return result
        tensor = torch.as_tensor(videos).detach().float()
        if tensor.ndim != 4:
            raise ValueError(
                "common dump expects [T,C,H,W], got {} for {}".format(
                    tuple(tensor.shape), path))
        low, high = float(tensor.min()), float(tensor.max())
        if low < -1e-4 or high > 1.0 + 1e-4:
            raise ValueError(
                "common dump pixels escape [0,1]: min={} max={} ({})"
                .format(low, high, path))
        target_dir = Path(dump_root) / role
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / (clip_id + ".pt")
        temporary = target.with_suffix(".pt.tmp")
        # Quantize once to uint8; all methods are scored from the same quantization.
        torch.save((tensor.clamp(0, 1) * 255).round().to(torch.uint8).cpu(),
                   temporary)
        os.replace(temporary, target)
        print("[common-dump] {} {} {}".format(
            role, clip_id, tuple(tensor.shape)), flush=True)
        return result

    trainer_module.save_multiview_videos_grid = wrapped


def main():
    _install_lossless_dump()
    trainer = trainer_module.WanUnifiedTrainer(Args.parse_args())
    trainer.eval()


if __name__ == "__main__":
    main()
