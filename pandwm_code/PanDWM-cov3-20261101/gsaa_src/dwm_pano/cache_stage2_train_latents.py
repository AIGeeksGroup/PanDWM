"""Cache exact four-step Stage-2 latents for GS domain adaptation.

The dataset order and frame sampler intentionally mirror ``dwm_pano.train``
at world size one.  Each cache item therefore corresponds to exactly one of
the first N GS optimizer steps and stores only the selected frame latents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import dwm.common
from dwm.datasets.common import CollateFnIgnoring
from dwm_pano.evaluate_gs import (
    atomic_json_dump,
    build_g1_dataset_config,
    build_stage2_pipeline,
    ensure_single_rank_process_group,
    install_strict_pano_hardseam_stub,
    load_config,
    make_stage2_latent_provider,
)
from dwm_pano.generated_latent_cache import (
    CACHE_SCHEMA_VERSION,
    atomic_torch_save,
    latent_cache_path,
    make_cache_payload,
    scalar_int,
)
from dwm_pano.pipelines.ctsd_pano_gs_decoder import sample_frame_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache Stage-2 EMA latents for the first N GS train steps"
    )
    parser.add_argument("--gs-config", required=True, type=Path)
    parser.add_argument("--stage2-config", required=True, type=Path)
    parser.add_argument("--stage2-run", required=True, type=Path)
    parser.add_argument("--stage2-step", required=True, type=int)
    parser.add_argument("--stage2-ema", action="store_true")
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--start-offset", default=0, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.stage2_step < 1 or args.steps < 1:
        parser.error("--stage2-step and --steps must be positive")
    if args.start_offset < 0:
        parser.error("--start-offset must be non-negative")
    return args


def ordered_training_indices(
    dataset: torch.utils.data.Dataset, seed: int
) -> list[int]:
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=seed,
    )
    sampler.set_epoch(0)
    return [int(value) for value in sampler]


def digest_indices(indices: list[int]) -> str:
    serialized = ",".join(str(value) for value in indices).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def expected_manifest(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset_length: int,
    indices: list[int],
) -> dict[str, Any]:
    training = config["pipeline"]["training_config"]
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_kind": "stage2_4step_selected_train_latents",
        "gs_config": str(args.gs_config.resolve()),
        "stage2_config": str(args.stage2_config.resolve()),
        "stage2_run": str(args.stage2_run.resolve()),
        "stage2_step": args.stage2_step,
        "stage2_ema": args.stage2_ema,
        "requested_steps": args.steps,
        "start_offset": args.start_offset,
        "dataset_length": dataset_length,
        "generator_seed": int(config["generator_seed"]),
        "frames_per_step": int(training["frames_per_step"]),
        "frame_sampling": training.get("frame_sampling", "random"),
        "ordered_indices_sha256": digest_indices(indices),
    }


def assert_resume_compatible(
    actual: dict[str, Any], expected: dict[str, Any]
) -> int:
    mismatches = [
        key for key, value in expected.items() if actual.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "Existing cache manifest is incompatible: " + ", ".join(mismatches)
        )
    completed = int(actual.get("completed_steps", 0))
    if not 0 <= completed <= expected["requested_steps"]:
        raise RuntimeError(f"Invalid completed_steps in cache: {completed}")
    return completed


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.gs_config)
    training = config["pipeline"]["training_config"]
    if not config.get("fix_training_data_order", False):
        raise RuntimeError(
            "Cache generation requires fix_training_data_order=true")
    if config["training_dataloader"].get("batch_size", 1) != 1:
        raise RuntimeError("Cache generation requires GS batch_size=1")
    if args.steps > 0 and args.steps > config.get("train_epochs", 1) * 10**9:
        raise RuntimeError("Unreasonable cache step count")

    device = torch.device("cuda", 0)
    created_process_group = ensure_single_rank_process_group(device)
    try:
        stage2_pipeline = build_stage2_pipeline(
            args.stage2_config,
            args.stage2_run,
            args.stage2_step,
            args.stage2_ema,
            device,
        )
        latent_provider = make_stage2_latent_provider(
            None, stage2_pipeline, "stage2")

        install_strict_pano_hardseam_stub()
        dataset_config = build_g1_dataset_config(config["training_dataset"])
        dataset = dwm.common.create_instance_from_config(dataset_config)
        indices = ordered_training_indices(dataset, int(config["generator_seed"]))
        if args.start_offset + args.steps > len(indices):
            raise RuntimeError(
                "Requested cache range [{}:{}) exceeds dataset length {}"
                .format(
                    args.start_offset,
                    args.start_offset + args.steps,
                    len(indices),
                )
            )
        indices = indices[
            args.start_offset:args.start_offset + args.steps]
        expected = expected_manifest(args, config, len(dataset), indices)

        args.out.mkdir(parents=True, exist_ok=True)
        manifest_path = args.out / "manifest.json"
        rng_path = args.out / "rng_state.pt"
        completed = 0
        frame_generator = torch.Generator().manual_seed(
            int(config["generator_seed"]))
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                completed = assert_resume_compatible(
                    json.load(handle), expected)
            if completed:
                if not rng_path.is_file():
                    raise RuntimeError(
                        "Cache manifest is resumable but rng_state.pt is missing")
                rng_state = torch.load(
                    rng_path, map_location="cpu", weights_only=True)
                stage2_pipeline.generator.set_state(rng_state["stage2"])
                frame_generator.set_state(rng_state["frame_sampler"])
                print(f"Resuming cache after {completed}/{args.steps} items.",
                      flush=True)

        subset = torch.utils.data.Subset(dataset, indices[completed:])
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=CollateFnIgnoring(keys=["clip_text"]),
        )
        for position, batch in enumerate(loader, start=completed):
            frame_count = int(batch["vae_images"].shape[1])
            frame_indices = sample_frame_indices(
                frame_count,
                int(training["frames_per_step"]),
                training.get("frame_sampling", "random"),
                frame_generator,
            )
            clip_latents = latent_provider(batch)
            selected_latents = clip_latents[0, frame_indices]
            selected_timestamps = batch["frame_timestamps_us"][
                0, frame_indices]
            dataset_index = scalar_int(batch["dataset_index"])
            anchor_timestamp_us = scalar_int(batch["anchor_timestamp_us"])
            cache_path = latent_cache_path(
                args.out, dataset_index, anchor_timestamp_us)
            atomic_torch_save(make_cache_payload(
                dataset_index=dataset_index,
                anchor_timestamp_us=anchor_timestamp_us,
                frame_indices=frame_indices,
                frame_timestamps_us=selected_timestamps,
                latents=selected_latents,
            ), cache_path)

            next_completed = position + 1
            atomic_torch_save({
                "stage2": stage2_pipeline.generator.get_state(),
                "frame_sampler": frame_generator.get_state(),
            }, rng_path)
            manifest = dict(expected)
            manifest["completed_steps"] = next_completed
            atomic_json_dump(manifest, manifest_path)
            if next_completed == 1 or next_completed % 10 == 0 \
                    or next_completed == args.steps:
                print(
                    f"cached {next_completed}/{args.steps}: {cache_path.name}",
                    flush=True,
                )
    finally:
        if created_process_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
