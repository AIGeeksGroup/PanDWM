"""Segmented latency benchmark for the 3+16 PanDWM inference path.

The benchmark intentionally separates the two products of the video model:

* pixel-video route: context VAE encode -> denoising -> VAE decode;
* Gaussian route: context VAE encode -> 4-step denoising -> Gaussian head ->
  six-view gsplat rendering.

The 50-step arm uses the same Stage-2 EMA weights as the 4-step arm and only
restores the inherited 50-step sampler.  It is therefore a controlled speed
comparison, not a valid quality evaluation of the distilled checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

import dwm.common


MIB = 1024 ** 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark 50-step, 4-step, VAE, Gaussian head and gsplat")
    parser.add_argument("--gs-config", type=Path, required=True)
    parser.add_argument("--gs-run", type=Path, required=True)
    parser.add_argument("--gs-step", type=int, required=True)
    parser.add_argument("--stage2-config", type=Path, required=True)
    parser.add_argument("--stage2-run", type=Path, required=True)
    parser.add_argument("--stage2-step", type=int, required=True)
    parser.add_argument("--stage2-ema", action="store_true")
    parser.add_argument(
        "--disable-50step-cfg",
        action="store_true",
        help=(
            "Disable classifier-free guidance for the inherited 50-step "
            "sampler. This makes the 50-step and 4-step arms both use one "
            "denoiser input batch and isolates the NFE reduction."),
    )
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def sync() -> None:
    torch.cuda.synchronize()


def memory_snapshot() -> dict[str, float]:
    return {
        "allocated_mib": torch.cuda.memory_allocated() / MIB,
        "reserved_mib": torch.cuda.memory_reserved() / MIB,
    }


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def measure(
    name: str,
    function: Callable[[], Any],
    warmup: int,
    repeats: int,
    before_each: Callable[[], None] | None = None,
    call_counter: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    last_output = None
    for index in range(warmup):
        if before_each is not None:
            before_each()
        last_output = function()
        sync()
        print(f"{name}: warmup {index + 1}/{warmup} done", flush=True)
        del last_output
        last_output = None
        gc.collect()

    wall_values = []
    cuda_values = []
    peak_allocated = []
    peak_reserved = []
    incremental_allocated = []
    forward_calls = []
    forward_batches = []
    for index in range(repeats):
        if before_each is not None:
            before_each()
        if call_counter is not None:
            call_counter["calls"] = 0
            call_counter["batches"] = []
        gc.collect()
        sync()
        baseline_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wall_start = time.perf_counter()
        last_output = function()
        end.record()
        sync()
        wall_seconds = time.perf_counter() - wall_start
        cuda_seconds = start.elapsed_time(end) / 1000.0
        wall_values.append(wall_seconds)
        cuda_values.append(cuda_seconds)
        peak = torch.cuda.max_memory_allocated()
        peak_allocated.append(peak / MIB)
        peak_reserved.append(torch.cuda.max_memory_reserved() / MIB)
        incremental_allocated.append((peak - baseline_allocated) / MIB)
        if call_counter is not None:
            forward_calls.append(int(call_counter["calls"]))
            forward_batches.append(list(call_counter["batches"]))
        print(
            f"{name}: repeat {index + 1}/{repeats} "
            f"wall={wall_seconds:.3f}s cuda={cuda_seconds:.3f}s "
            f"peak={peak / MIB:.1f}MiB",
            flush=True,
        )
        if index + 1 < repeats:
            del last_output
            last_output = None

    result = {
        "name": name,
        "wall_seconds": summarize(wall_values),
        "cuda_seconds": summarize(cuda_values),
        "peak_allocated_mib": summarize(peak_allocated),
        "peak_reserved_mib": summarize(peak_reserved),
        "incremental_peak_allocated_mib": summarize(incremental_allocated),
    }
    if call_counter is not None:
        result["denoiser_forward_calls"] = forward_calls
        result["denoiser_input_batches"] = forward_batches
    return result, last_output


@contextlib.contextmanager
def skip_vae_decode(pipeline: Any):
    """Keep the exact sampler but replace its terminal VAE decode.

    Both the inherited 50-step function and the 4-step helper decode before
    returning their latents.  A tiny stand-in lets us time latent generation
    without duplicating either denoising loop in this benchmark.
    """
    original_decode = pipeline.vae.decode

    def dummy_decode(sample: torch.Tensor, *args: Any, **kwargs: Any):
        image = torch.empty(
            (sample.shape[0], 3, 1, 1),
            dtype=sample.dtype,
            device=sample.device,
        )
        return (image,) if kwargs.get("return_dict") is False else image

    pipeline.vae.decode = dummy_decode
    try:
        yield
    finally:
        pipeline.vae.decode = original_decode


def make_sampler_call(
    pipeline: Any,
    inference: Callable[..., dict[str, torch.Tensor]],
    latent_shape: tuple[int, ...],
    batch: dict[str, Any],
    context_latents: torch.Tensor,
    reference_frame_count: int,
    disable_cfg: bool = False,
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        missing = object()
        guidance_scale = missing
        if disable_cfg:
            guidance_scale = pipeline.inference_config.pop(
                "guidance_scale", missing)
        try:
            with skip_vae_decode(pipeline):
                output = inference(
                    latent_shape,
                    batch,
                    "pt",
                    image_latents=context_latents,
                    reference_frame_count=reference_frame_count,
                )
        finally:
            if disable_cfg and guidance_scale is not missing:
                pipeline.inference_config["guidance_scale"] = guidance_scale
        # Always pin the clean context explicitly.  This matches the G1
        # provider and makes the output layout identical for both samplers.
        return torch.cat([
            context_latents,
            output["latents"][:, reference_frame_count:],
        ], dim=1)

    return run


def measure_gaussian_path(
    pipeline: Any,
    future_latents: torch.Tensor,
    warmup: int,
    repeats: int,
    ray_batch: dict[str, Any] | None = None,
    frame_indices: torch.Tensor | None = None,
) -> dict[str, Any]:
    latent_directions = pipeline.latent_directions.unsqueeze(0)
    pixel_directions = pipeline.pixel_directions.unsqueeze(0)
    if frame_indices is None:
        frame_indices = torch.arange(
            future_latents.shape[1], dtype=torch.long)
    if ray_batch is None:
        if pipeline.decoder.use_camera_origins:
            raise ValueError("ray_batch is required for camera-origin geometry")
        latent_origins, pixel_origins = None, None
    else:
        latent_origins, pixel_origins = pipeline._frame_ray_origins(
            ray_batch, frame_indices)
    use_temporal_context = pipeline.training_config.get(
        "demo_use_temporal_context", True)

    def one_frame(
        frame_latents: torch.Tensor, frame_index: int = 0
    ) -> tuple[torch.Tensor, int]:
        gaussians, _, _ = pipeline.decoder_wrapper(
            frame_latents, latent_directions, pixel_directions, None,
            latent_origins=None if latent_origins is None else
                latent_origins[frame_index:frame_index + 1],
            pixel_origins=None if pixel_origins is None else
                pixel_origins[frame_index:frame_index + 1])
        rendered_rgb, _, _, _ = pipeline._render_own_supervision(
            gaussians[0])
        return rendered_rgb, int(gaussians.shape[1])

    for index in range(warmup):
        rendered, _ = one_frame(future_latents[:, 0])
        sync()
        del rendered
        print(
            f"gaussian_head_and_render: warmup {index + 1}/{warmup} done",
            flush=True,
        )

    wall_values = []
    head_values = []
    render_values = []
    combined_cuda_values = []
    peak_allocated = []
    peak_reserved = []
    incremental_allocated = []
    gaussian_count = None
    for repeat_index in range(repeats):
        gc.collect()
        sync()
        baseline_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        overall_start = torch.cuda.Event(enable_timing=True)
        overall_end = torch.cuda.Event(enable_timing=True)
        head_pairs = []
        render_pairs = []
        overall_start.record()
        wall_start = time.perf_counter()
        previous_latents = None
        for frame_index in range(future_latents.shape[1]):
            head_start = torch.cuda.Event(enable_timing=True)
            head_end = torch.cuda.Event(enable_timing=True)
            render_start = torch.cuda.Event(enable_timing=True)
            render_end = torch.cuda.Event(enable_timing=True)
            head_start.record()
            gaussians, _, _ = pipeline.decoder_wrapper(
                future_latents[:, frame_index],
                latent_directions,
                pixel_directions,
                None,
                previous_latents=(
                    previous_latents if use_temporal_context else None),
                temporal_context_mask=(
                    1.0 if use_temporal_context
                    and previous_latents is not None else 0.0),
                latent_origins=None if latent_origins is None else
                    latent_origins[frame_index:frame_index + 1],
                pixel_origins=None if pixel_origins is None else
                    pixel_origins[frame_index:frame_index + 1],
            )
            previous_latents = (
                future_latents[:, frame_index].detach()
                if use_temporal_context else None)
            head_end.record()
            render_start.record()
            rendered_rgb, _, _, _ = pipeline._render_own_supervision(
                gaussians[0])
            render_end.record()
            head_pairs.append((head_start, head_end))
            render_pairs.append((render_start, render_end))
            gaussian_count = int(gaussians.shape[1])
            del rendered_rgb, gaussians
        overall_end.record()
        sync()
        wall_seconds = time.perf_counter() - wall_start
        head_seconds = sum(a.elapsed_time(b) for a, b in head_pairs) / 1000.0
        render_seconds = sum(a.elapsed_time(b) for a, b in render_pairs) / 1000.0
        combined_cuda = overall_start.elapsed_time(overall_end) / 1000.0
        peak = torch.cuda.max_memory_allocated()
        wall_values.append(wall_seconds)
        head_values.append(head_seconds)
        render_values.append(render_seconds)
        combined_cuda_values.append(combined_cuda)
        peak_allocated.append(peak / MIB)
        peak_reserved.append(torch.cuda.max_memory_reserved() / MIB)
        incremental_allocated.append((peak - baseline_allocated) / MIB)
        print(
            f"gaussian_head_and_render: repeat {repeat_index + 1}/{repeats} "
            f"wall={wall_seconds:.3f}s head={head_seconds:.3f}s "
            f"render={render_seconds:.3f}s peak={peak / MIB:.1f}MiB",
            flush=True,
        )

    frame_count = int(future_latents.shape[1])
    view_count = 6
    return {
        "name": "gaussian_head_and_render",
        "future_frame_count": frame_count,
        "views_per_frame": view_count,
        "render_output": pipeline.supervision_output_label(),
        "temporal_context": use_temporal_context,
        "gaussians_per_frame": gaussian_count,
        "wall_seconds": summarize(wall_values),
        "head_cuda_seconds": summarize(head_values),
        "render_cuda_seconds": summarize(render_values),
        "combined_cuda_seconds": summarize(combined_cuda_values),
        "head_fps": frame_count / statistics.median(head_values),
        "render_frame_sets_per_second": (
            frame_count / statistics.median(render_values)),
        "render_views_per_second": (
            frame_count * view_count / statistics.median(render_values)),
        "peak_allocated_mib": summarize(peak_allocated),
        "peak_reserved_mib": summarize(peak_reserved),
        "incremental_peak_allocated_mib": summarize(incremental_allocated),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from dwm_pano.demo_world_model import collate_single
    from dwm_pano.evaluate_gs import (
        build_g1_dataset_config,
        build_stage2_pipeline,
        decode_latents_to_images,
        encode_context_latents,
        ensure_single_rank_process_group,
    )

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    ensure_single_rank_process_group(device)

    with args.gs_config.open("r", encoding="utf-8") as handle:
        gs_config = json.load(handle)
    torch.manual_seed(gs_config["generator_seed"])

    dataset_config = build_g1_dataset_config(gs_config["validation_dataset"])
    dataset = dwm.common.create_instance_from_config(dataset_config)
    if not 0 <= args.clip_index < len(dataset):
        raise IndexError(
            f"clip {args.clip_index} out of range for {len(dataset)} clips")
    batch = collate_single(dataset[args.clip_index])
    clip_length = int(batch["vae_images"].shape[1])

    stage2_pipeline = build_stage2_pipeline(
        args.stage2_config,
        args.stage2_run,
        args.stage2_step,
        args.stage2_ema,
        device,
    )
    reference_frame_count = int(
        stage2_pipeline.inference_config.get("reference_frame_count", 3))
    if clip_length != 19 or reference_frame_count != 3:
        raise RuntimeError(
            "the publication benchmark is fixed to 3+16 frames; got "
            f"clip_length={clip_length}, reference={reference_frame_count}")

    context_result, context_latents = measure(
        "context_vae_encode",
        lambda: encode_context_latents(
            stage2_pipeline, batch, reference_frame_count),
        args.warmup,
        args.repeats,
    )
    latent_shape = tuple(batch["vae_images"].shape[:3]) + tuple(
        context_latents.shape[3:])

    four_step_inference = stage2_pipeline._g1_4step_inference
    fifty_step_inference = stage2_pipeline.__class__.inference_pipeline.__get__(
        stage2_pipeline, stage2_pipeline.__class__)
    generator_state = stage2_pipeline.generator.get_state()
    reset_generator = lambda: stage2_pipeline.generator.set_state(
        generator_state)

    call_counter = {"calls": 0, "batches": []}

    def count_forward(_module: Any, inputs: tuple[Any, ...]) -> None:
        call_counter["calls"] += 1
        batch_size = int(inputs[0].shape[0]) if inputs else -1
        call_counter["batches"].append(batch_size)

    hook = stage2_pipeline.model_wrapper.register_forward_pre_hook(
        count_forward)
    try:
        fifty_result, fifty_latents = measure(
            "50step_latent_generation",
            make_sampler_call(
                stage2_pipeline,
                fifty_step_inference,
                latent_shape,
                batch,
                context_latents,
                reference_frame_count,
                disable_cfg=args.disable_50step_cfg,
            ),
            args.warmup,
            args.repeats,
            before_each=reset_generator,
            call_counter=call_counter,
        )
        four_result, four_latents = measure(
            "4step_latent_generation",
            make_sampler_call(
                stage2_pipeline,
                four_step_inference,
                latent_shape,
                batch,
                context_latents,
                reference_frame_count,
            ),
            args.warmup,
            args.repeats,
            before_each=reset_generator,
            call_counter=call_counter,
        )
    finally:
        hook.remove()

    decode_50_result, decoded_50 = measure(
        "50step_vae_decode_19frames",
        lambda: decode_latents_to_images(stage2_pipeline, fifty_latents),
        args.warmup,
        args.repeats,
    )
    del decoded_50, fifty_latents
    gc.collect()
    decode_4_result, decoded_4 = measure(
        "4step_vae_decode_19frames",
        lambda: decode_latents_to_images(stage2_pipeline, four_latents),
        args.warmup,
        args.repeats,
    )
    del decoded_4
    gc.collect()

    checkpoint_path = args.gs_run / "checkpoints" / f"{args.gs_step}.pth"
    gs_pipeline = dwm.common.create_instance_from_config(
        gs_config["pipeline"],
        output_path=str(args.gs_run),
        config=gs_config,
        device=device,
        resume_from=None,
    )
    gs_pipeline.decoder.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu"))
    gs_pipeline.decoder.eval()
    gs_pipeline.decoder_wrapper.eval()
    gs_pipeline._prepare_nuscenes6_supervision(batch)
    gs_result = measure_gaussian_path(
        gs_pipeline,
        four_latents[:, reference_frame_count:],
        args.warmup,
        args.repeats,
        ray_batch=batch,
        frame_indices=torch.arange(
            reference_frame_count, clip_length, dtype=torch.long),
    )

    context_median = context_result["wall_seconds"]["median"]
    fifty_median = fifty_result["wall_seconds"]["median"]
    four_median = four_result["wall_seconds"]["median"]
    decode_50_median = decode_50_result["wall_seconds"]["median"]
    decode_4_median = decode_4_result["wall_seconds"]["median"]
    gs_wall_median = gs_result["wall_seconds"]["median"]
    generated_frames = clip_length - reference_frame_count

    summary = {
        "latent_generation_speedup_50_over_4": (
            fifty_median / four_median),
        "50step_pixel_video_seconds": (
            context_median + fifty_median + decode_50_median),
        "4step_pixel_video_seconds": (
            context_median + four_median + decode_4_median),
        "pixel_video_speedup_50_over_4": (
            (context_median + fifty_median + decode_50_median)
            / (context_median + four_median + decode_4_median)),
        "50step_pixel_video_fps": generated_frames / (
            context_median + fifty_median + decode_50_median),
        "4step_pixel_video_fps": generated_frames / (
            context_median + four_median + decode_4_median),
        "4step_gaussian_world_seconds": (
            context_median + four_median + gs_wall_median),
        "4step_gaussian_world_fps": generated_frames / (
            context_median + four_median + gs_wall_median),
    }

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "comparison_type": (
                "pure_nfe_same_cfg"
                if args.disable_50step_cfg
                else "deployed_paths_asymmetric_cfg"),
            "batch_size": 1,
            "clip_index": args.clip_index,
            "clip_length": clip_length,
            "reference_frame_count": reference_frame_count,
            "generated_frame_count": generated_frames,
            "supervision_geometry": gs_pipeline.supervision_geometry,
            "views_per_frame": 6,
            "render_width": (
                gs_pipeline.nuscenes_render_width
                if gs_pipeline.uses_nuscenes6_stitch else
                gs_pipeline.training_config["target_width"]),
            "render_height": (
                gs_pipeline.nuscenes_render_height
                if gs_pipeline.uses_nuscenes6_stitch else
                gs_pipeline.training_config["target_height"]),
            "render_output": gs_pipeline.supervision_output_label(),
            "temporal_context": gs_pipeline.training_config.get(
                "demo_use_temporal_context", True),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "50step_nfe": 50,
            "4step_nfe": 4,
            "50step_note": (
                "Stage-2 EMA weights through the inherited 50-step sampler; "
                "speed-only, not a valid distilled-model quality path"),
            "50step_cfg": not args.disable_50step_cfg,
            "4step_cfg": False,
            "50step_denoiser_input_batch_size": (
                1 if args.disable_50step_cfg else 2),
            "4step_denoiser_input_batch_size": 1,
        },
        "checkpoints": {
            "stage2": str(
                args.stage2_run / "checkpoints" /
                f"{args.stage2_step}{'_ema' if args.stage2_ema else ''}.pth"),
            "gaussian": str(checkpoint_path),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_device_index_visible": 0,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "hostname": platform.node(),
            "initial_memory": memory_snapshot(),
        },
        "segments": {
            "context_vae_encode": context_result,
            "50step_latent_generation": fifty_result,
            "4step_latent_generation": four_result,
            "50step_vae_decode_19frames": decode_50_result,
            "4step_vae_decode_19frames": decode_4_result,
            "gaussian_head_and_render": gs_result,
        },
        "summary": summary,
    }
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote: {args.out}", flush=True)

    del four_latents, gs_pipeline, stage2_pipeline
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
