#!/usr/bin/env python3
"""Run official OmniNWM inference with an additional lossless RGB tap.

The official ``tools/inference.py`` module is loaded unchanged. Its global
``process_and_save`` callback is wrapped so the decoded six-view RGB tensor is
saved before the official visualization/MP4 writer mutates it. Model loading,
dataset construction, denoising, VAE decode, and official output saving remain
the upstream implementation.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import torch


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings(item)


def load_official_driver(repo: Path):
    path = repo / "tools" / "inference.py"
    spec = importlib.util.spec_from_file_location(
        "omninwm_official_inference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official inference driver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkout_commit(repo: Path):
    git_dir = repo / ".git"
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose = git_dir / ref
    if loose.is_file():
        return loose.read_text().strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == ref:
                    return commit
    raise RuntimeError(f"cannot resolve official checkout HEAD ref: {ref}")


def main():
    import os

    repo = Path(os.environ["OMNINWM_OFFICIAL_REPO"]).resolve()
    expected_commit = "8e2333ee0dfb7be7a477e51a97e644bf7ec53039"
    ref = checkout_commit(repo)
    if ref != expected_commit:
        raise RuntimeError(f"official checkout mismatch: {ref}")

    driver = load_official_driver(repo)
    official_process_and_save = driver.process_and_save

    def process_and_save_lossless(
        x, batch, cfg, sub_dir, generate_sampling_option, epoch,
        start_index, dataset, devices=None,
    ):
        rgb = x["rgb_video"].detach().cpu()
        if rgb.ndim != 5 or rgb.shape[0] != 6 or rgb.shape[1] != 3:
            raise ValueError(
                f"official decoded RGB is not [6,3,T,H,W]: {tuple(rgb.shape)}")
        clip_ids = {
            int(match.group(1))
            for value in strings(batch.get("path"))
            for match in [re.search(r"clip_(\d{4})", value)]
            if match is not None
        }
        if len(clip_ids) != 1:
            raise ValueError(f"could not resolve one common clip id: {clip_ids}")
        clip_id = clip_ids.pop()
        lossless = Path(cfg.save_dir) / "lossless"
        lossless.mkdir(parents=True, exist_ok=True)
        target = lossless / f"clip_{clip_id:04d}.pt"
        if target.exists():
            raise FileExistsError(target)
        rgb_uint8 = (
            ((rgb.float().clamp(-1, 1) + 1.0) * 127.5)
            .round().clamp(0, 255).to(torch.uint8)
        )
        torch.save(rgb_uint8, target)
        meta = {
            "common_clip_index": clip_id,
            "shape": list(rgb_uint8.shape),
            "dtype": str(rgb_uint8.dtype),
            "official_epoch": epoch,
            "official_dataloader_index": start_index,
        }
        target.with_suffix(".json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"lossless_rgb_saved {target} {tuple(rgb_uint8.shape)}",
              flush=True)
        return official_process_and_save(
            x, batch, cfg, sub_dir, generate_sampling_option, epoch,
            start_index, dataset, devices=devices)

    driver.process_and_save = process_and_save_lossless
    driver.main()


if __name__ == "__main__":
    main()
