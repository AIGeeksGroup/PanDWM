"""Paired temporal evaluation for two panoramic Gaussian decoders.

Both checkpoints consume the exact same clip latents.  The headline
``XRenderGT-L1`` is reduced on candidate-independent target-static-valid
support.  Per-model and common rendered-visibility reductions remain explicit
diagnostics, but cannot shrink the support of the headline metric.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any

import torch
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure

import dwm.common
from dwm_pano.evaluate_gs import (
    atomic_json_dump,
    build_dataset_config,
    build_g1_dataset_config,
    build_stage2_pipeline,
    ensure_single_rank_process_group,
    install_strict_pano_hardseam_stub,
    make_stage2_latent_provider,
)
from dwm_pano.pipelines.ctsd_pano_gs_decoder import (
    cross_render_visibility_mask,
    cross_target_validity_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fair paired temporal evaluation of two GS checkpoints")
    for name in ("baseline", "candidate"):
        parser.add_argument(f"--{name}-config", required=True, type=Path)
        parser.add_argument(f"--{name}-run", required=True, type=Path)
        parser.add_argument(f"--{name}-step", required=True, type=int)
    parser.add_argument("--clip-limit", type=int, default=4)
    parser.add_argument("--frame-count", type=int, default=19)
    parser.add_argument(
        "--dataset-source", choices=("training", "validation"),
        default="validation",
        help="Dataset config supplying RGB/seg paths (use training + --pkl "
             "for the held-out P2 split)")
    parser.add_argument(
        "--pkl", type=Path,
        help="Optional clip pkl override; uses fixed [[10,20]] enumeration")
    parser.add_argument(
        "--latent-source", choices=("gt", "stage2"), default="gt")
    parser.add_argument("--stage2-config", type=Path)
    parser.add_argument("--stage2-run", type=Path)
    parser.add_argument("--stage2-step", type=int)
    parser.add_argument("--stage2-ema", action="store_true")
    parser.add_argument(
        "--range-dir", type=Path,
        help="Depth directory; default is validation rgb_dir/../depth")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.clip_limit < 1 or args.frame_count < 2:
        parser.error("--clip-limit must be >=1 and --frame-count must be >=2")
    for name in ("baseline_step", "candidate_step"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.latent_source == "stage2":
        missing = [
            name for name in ("stage2_config", "stage2_run", "stage2_step")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "--latent-source stage2 requires " + ", ".join(
                    "--" + name.replace("_", "-") for name in missing))
    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_eval_paths(
    config: dict[str, Any], range_dir: Path | None, dataset_key: str
) -> Path:
    base = config[dataset_key]["base_dataset"]
    rgb_dir = Path(base["rgb_dir"])
    resolved_range_dir = range_dir or rgb_dir.parent / "depth"
    training = config["pipeline"]["training_config"]
    training["range_dir"] = str(resolved_range_dir)
    training["segment_dir"] = str(base["seg_dir"])
    return resolved_range_dir


def build_gs_pipeline(
    config: dict[str, Any], run_dir: Path, step: int,
    device: torch.device,
) -> Any:
    pipeline_config = copy.deepcopy(config["pipeline"])
    # Evaluation loads the requested checkpoint directly.  Avoid loading its
    # training parent first, and avoid allocating a VGG LPIPS copy per model.
    pipeline_config.pop("initial_decoder_checkpoint_path", None)
    pipeline_config["training_config"]["weight_perceptual"] = 0.0
    pipeline = dwm.common.create_instance_from_config(
        pipeline_config,
        output_path=str(run_dir),
        config=config,
        device=device,
        resume_from=None,
    )
    checkpoint = run_dir / "checkpoints" / f"{step}.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    state, migrated = pipeline.decoder.migrate_input_checkpoint(state)
    pipeline.decoder.load_state_dict(state)
    pipeline.decoder.eval()
    pipeline.decoder_wrapper.eval()
    pipeline.vae.eval()
    # Checkpoint N is written after train_step(global_step=N-1).  Reproduce
    # the origin ramp used by that exact step; checkpoints after the warmup
    # naturally evaluate at the final scale 1.0.
    pipeline.evaluation_camera_origin_scale = \
        pipeline._camera_origin_scale(max(0, step - 1))
    if migrated:
        print(f"{checkpoint}: migrated {migrated}", flush=True)
    return pipeline


def new_accumulator() -> dict[str, float]:
    return {
        "frames": 0,
        "directions": 0,
        "direction_pixels": 0,
        "target_static_valid_pixels": 0,
        "baseline_visible_pixels": 0,
        "candidate_visible_pixels": 0,
        "common_visible_pixels": 0,
        "baseline_alpha_covered_pixels": 0,
        "candidate_alpha_covered_pixels": 0,
        "baseline_own_l1_sum": 0.0,
        "candidate_own_l1_sum": 0.0,
        "baseline_psnr_sum": 0.0,
        "candidate_psnr_sum": 0.0,
        "baseline_ssim_sum": 0.0,
        "candidate_ssim_sum": 0.0,
        "baseline_xrender_gt_target_sum": 0.0,
        "candidate_xrender_gt_target_sum": 0.0,
        "baseline_xrender_gt_visible_sum": 0.0,
        "candidate_xrender_gt_visible_sum": 0.0,
        "baseline_xrender_gt_common_sum": 0.0,
        "candidate_xrender_gt_common_sum": 0.0,
        "baseline_xrender_self_sum": 0.0,
        "candidate_xrender_self_sum": 0.0,
    }


def add_accumulator(target: dict[str, float], source: dict[str, float]) -> None:
    for key in target:
        target[key] += source[key]


def source_static_gaussians(
    gaussians: torch.Tensor, source_segments: torch.Tensor
) -> torch.Tensor:
    source_static = 1.0 - source_segments.reshape(
        source_segments.shape[0], -1, 1).to(gaussians.dtype)
    return torch.cat([
        gaussians[..., :6],
        gaussians[..., 6:7] * source_static,
        gaussians[..., 7:],
    ], dim=-1)


def update_own_metrics(
    accumulator: dict[str, float], model_name: str,
    rendered: torch.Tensor, target: torch.Tensor,
) -> None:
    rendered = rendered.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    accumulator[f"{model_name}_own_l1_sum"] += float(
        (rendered - target).abs().mean())
    accumulator[f"{model_name}_psnr_sum"] += float(
        peak_signal_noise_ratio(rendered, target, data_range=1.0))
    accumulator[f"{model_name}_ssim_sum"] += float(
        structural_similarity_index_measure(
            rendered, target, data_range=1.0))


def update_cross_render_reductions(
    accumulator: dict[str, float],
    rendered: dict[str, torch.Tensor],
    alphas: dict[str, torch.Tensor],
    visibility: dict[str, torch.Tensor],
    target_support: torch.Tensor,
    target_rgb: torch.Tensor,
    own_rgb: dict[str, torch.Tensor],
    alpha_threshold: float,
) -> None:
    """Accumulate one temporal direction without model-selected support.

    ``target_support`` depends only on target depth validity/static semantics.
    Rendered visibility is still useful for diagnosing occlusion agreement,
    but it never changes the headline XRenderGT denominator.
    """
    common = visibility["baseline"] & visibility["candidate"]
    total_count = target_support.numel()
    target_count = int(target_support.sum())
    common_count = int(common.sum())
    accumulator["directions"] += 1
    accumulator["direction_pixels"] += total_count
    accumulator["target_static_valid_pixels"] += target_count
    accumulator["common_visible_pixels"] += common_count

    target_float = target_support.to(rendered["baseline"].dtype)
    common_float = common.to(rendered["baseline"].dtype)
    for name in ("baseline", "candidate"):
        model_visibility = visibility[name]
        visible_count = int(model_visibility.sum())
        accumulator[f"{name}_visible_pixels"] += visible_count
        alpha_covered = target_support & (
            alphas[name] >= alpha_threshold)
        accumulator[f"{name}_alpha_covered_pixels"] += int(
            alpha_covered.sum())

        gt_error = (rendered[name] - target_rgb).abs().mean(dim=1)
        accumulator[f"{name}_xrender_gt_target_sum"] += float(
            (gt_error * target_float).sum())
        accumulator[f"{name}_xrender_gt_visible_sum"] += float(
            (gt_error * model_visibility.to(gt_error.dtype)).sum())
        accumulator[f"{name}_xrender_gt_common_sum"] += float(
            (gt_error * common_float).sum())

        self_error = (rendered[name] - own_rgb[name]).abs().mean(dim=1)
        accumulator[f"{name}_xrender_self_sum"] += float(
            (self_error * common_float).sum())


@torch.no_grad()
def update_direction_metrics(
    accumulator: dict[str, float],
    baseline: Any,
    candidate: Any,
    batch: dict[str, Any],
    source_record: dict[str, Any],
    target_record: dict[str, Any],
    gt_ranges: torch.Tensor,
    gt_valid: torch.Tensor,
    gt_segments: torch.Tensor,
    canonical_pixel_origins: torch.Tensor | None,
) -> None:
    source_index = source_record["sample_index"]
    target_index = target_record["sample_index"]
    source_frame = source_record["frame_index"]
    target_frame = target_record["frame_index"]
    cross_c2w = candidate._target_cameras_in_source_ego(
        batch, 0, source_frame, target_frame)

    target_segments = gt_segments[target_index:target_index + 1]
    target_dynamic = candidate._resample_targets(
        candidate._stitch_canvas(
            target_segments.unsqueeze(2).float()),
        mode="nearest",
    )[0, :, 0]
    target_depth, target_depth_valid = candidate._target_view_depths(
        gt_ranges[target_index:target_index + 1],
        gt_valid[target_index:target_index + 1],
        None if canonical_pixel_origins is None else
            canonical_pixel_origins[target_index:target_index + 1],
    )
    target_depth = target_depth[0]
    target_depth_valid = target_depth_valid[0]

    rendered = {}
    alphas = {}
    visibility = {}
    visibility_config = candidate.training_config
    alpha_threshold = visibility_config.get("cross_min_alpha", 0.05)
    target_support = cross_target_validity_mask(
        target_depth,
        target_dynamic < 0.5,
        target_depth_valid,
        near_plane=visibility_config["near_plane"],
    )
    for name, pipeline in (("baseline", baseline), ("candidate", candidate)):
        static_gaussians = source_static_gaussians(
            source_record[f"{name}_gaussians"],
            gt_segments[source_index:source_index + 1],
        )
        rgb, depth, alpha = pipeline.render_targets(
            static_gaussians[0], target_c2w=cross_c2w)
        rendered[name] = rgb
        alphas[name] = alpha
        visibility[name] = cross_render_visibility_mask(
            depth,
            target_depth,
            alpha,
            target_dynamic < 0.5,
            target_depth_valid,
            min_alpha=alpha_threshold,
            absolute_tolerance_m=visibility_config.get(
                "cross_depth_abs_tolerance_m", 2.0),
            relative_tolerance=visibility_config.get(
                "cross_depth_rel_tolerance", 0.1),
            near_plane=visibility_config["near_plane"],
        )

    update_cross_render_reductions(
        accumulator,
        rendered,
        alphas,
        visibility,
        target_support,
        target_record["target_rgb"],
        {
            name: target_record[f"{name}_own_rgb"]
            for name in ("baseline", "candidate")
        },
        alpha_threshold,
    )


def finalize(accumulator: dict[str, float]) -> dict[str, Any]:
    frames = max(1.0, accumulator["frames"])
    common = max(1.0, accumulator["common_visible_pixels"])
    direction_pixels = max(1.0, accumulator["direction_pixels"])
    target_support = max(1.0, accumulator["target_static_valid_pixels"])
    baseline_visible = max(1.0, accumulator["baseline_visible_pixels"])
    candidate_visible = max(1.0, accumulator["candidate_visible_pixels"])
    result = {
        "counts": {
            "frames": int(accumulator["frames"]),
            "directions": int(accumulator["directions"]),
            "direction_pixels": int(accumulator["direction_pixels"]),
            "target_static_valid_pixels": int(
                accumulator["target_static_valid_pixels"]),
            "baseline_visible_pixels": int(
                accumulator["baseline_visible_pixels"]),
            "candidate_visible_pixels": int(
                accumulator["candidate_visible_pixels"]),
            "common_visible_pixels": int(
                accumulator["common_visible_pixels"]),
        },
        "coverage": {
            "baseline": accumulator["baseline_visible_pixels"]
            / direction_pixels,
            "candidate": accumulator["candidate_visible_pixels"]
            / direction_pixels,
            "common": accumulator["common_visible_pixels"]
            / direction_pixels,
            "target_static_valid":
                accumulator["target_static_valid_pixels"] / direction_pixels,
        },
        "alpha_coverage_on_target": {
            "baseline": accumulator["baseline_alpha_covered_pixels"]
            / target_support,
            "candidate": accumulator["candidate_alpha_covered_pixels"]
            / target_support,
        },
        "visibility_recall_on_target": {
            "baseline": accumulator["baseline_visible_pixels"]
            / target_support,
            "candidate": accumulator["candidate_visible_pixels"]
            / target_support,
            "common": accumulator["common_visible_pixels"] / target_support,
        },
        "baseline": {
            "Own-L1": accumulator["baseline_own_l1_sum"] / frames,
            "Own-PSNR": accumulator["baseline_psnr_sum"] / frames,
            "Own-SSIM": accumulator["baseline_ssim_sum"] / frames,
            "XRenderGT-L1":
                accumulator["baseline_xrender_gt_target_sum"]
                / target_support,
            "XRenderGT-L1-render-visible":
                accumulator["baseline_xrender_gt_visible_sum"]
                / baseline_visible,
            "XRenderGT-L1-common":
                accumulator["baseline_xrender_gt_common_sum"] / common,
            "XRenderSelf-L1-common":
                accumulator["baseline_xrender_self_sum"] / common,
        },
        "candidate": {
            "Own-L1": accumulator["candidate_own_l1_sum"] / frames,
            "Own-PSNR": accumulator["candidate_psnr_sum"] / frames,
            "Own-SSIM": accumulator["candidate_ssim_sum"] / frames,
            "XRenderGT-L1":
                accumulator["candidate_xrender_gt_target_sum"]
                / target_support,
            "XRenderGT-L1-render-visible":
                accumulator["candidate_xrender_gt_visible_sum"]
                / candidate_visible,
            "XRenderGT-L1-common":
                accumulator["candidate_xrender_gt_common_sum"] / common,
            "XRenderSelf-L1-common":
                accumulator["candidate_xrender_self_sum"] / common,
        },
    }
    result["candidate_minus_baseline"] = {
        key: result["candidate"][key] - result["baseline"][key]
        for key in result["baseline"]
    }
    return result


@torch.no_grad()
def evaluate_clip(
    baseline: Any,
    candidate: Any,
    batch: dict[str, Any],
    frame_count: int,
    latent_provider: Any,
) -> dict[str, float]:
    frame_indices = torch.arange(frame_count, dtype=torch.long)
    half_images = batch["vae_images"][:, :frame_count].flatten(0, 1)
    half_images = half_images.to(candidate.device)
    gt_ranges, gt_valid = candidate._load_frame_ranges(
        batch, frame_indices, return_valid=True)
    gt_segments = candidate._load_frame_segments(batch, frame_indices)
    baseline_latent_origins, baseline_pixel_origins = \
        baseline._frame_ray_origins(batch, frame_indices)
    candidate_latent_origins, candidate_pixel_origins = \
        candidate._frame_ray_origins(batch, frame_indices)
    canonical_pixel_origins = candidate_pixel_origins \
        if candidate_pixel_origins is not None else baseline_pixel_origins

    clip_latents = latent_provider(batch) if latent_provider is not None else None
    previous_latents = None
    previous_record = None
    accumulator = new_accumulator()
    for sample_index in range(frame_count):
        frame_images = half_images[sample_index:sample_index + 1]
        if clip_latents is None:
            latents = baseline.encode_half_latents(frame_images)
        else:
            latents = clip_latents[:, sample_index]
        target_rgb = candidate._resample_targets(
            candidate._stitch_canvas(frame_images))[0]

        record = {
            "sample_index": sample_index,
            "frame_index": sample_index,
            "target_rgb": target_rgb,
        }
        for name, pipeline, latent_origins, pixel_origins in (
            ("baseline", baseline, baseline_latent_origins,
             baseline_pixel_origins),
            ("candidate", candidate, candidate_latent_origins,
             candidate_pixel_origins),
        ):
            gaussians, _, _ = pipeline.decoder_wrapper(
                latents,
                pipeline.latent_directions.unsqueeze(0),
                pipeline.pixel_directions.unsqueeze(0),
                None,
                previous_latents=previous_latents,
                temporal_context_mask=0.0 if previous_latents is None else 1.0,
                latent_origins=None if latent_origins is None else
                    latent_origins[sample_index:sample_index + 1],
                pixel_origins=None if pixel_origins is None else
                    pixel_origins[sample_index:sample_index + 1],
                camera_origin_scale=pipeline.evaluation_camera_origin_scale,
            )
            own_rgb, _, _ = pipeline.render_targets(gaussians[0])
            record[f"{name}_gaussians"] = gaussians
            record[f"{name}_own_rgb"] = own_rgb
            update_own_metrics(
                accumulator, name, own_rgb, target_rgb)
        accumulator["frames"] += 1

        if previous_record is not None:
            update_direction_metrics(
                accumulator, baseline, candidate, batch,
                previous_record, record, gt_ranges, gt_valid, gt_segments,
                canonical_pixel_origins)
            update_direction_metrics(
                accumulator, baseline, candidate, batch,
                record, previous_record, gt_ranges, gt_valid, gt_segments,
                canonical_pixel_origins)
        previous_record = record
        previous_latents = latents
    return accumulator


def main() -> None:
    args = parse_args()
    baseline_config = load_json(args.baseline_config)
    candidate_config = load_json(args.candidate_config)
    dataset_key = f"{args.dataset_source}_dataset"
    baseline_range_dir = configure_eval_paths(
        baseline_config, args.range_dir, dataset_key)
    candidate_range_dir = configure_eval_paths(
        candidate_config, args.range_dir, dataset_key)
    for path in (baseline_range_dir, candidate_range_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Depth directory is missing: {path}")

    baseline_training = baseline_config["pipeline"]["training_config"]
    candidate_training = candidate_config["pipeline"]["training_config"]
    non_virtual = [
        label for label, training in (
            ("baseline", baseline_training),
            ("candidate", candidate_training),
        )
        if training.get("supervision_geometry", "virtual_center")
        != "virtual_center"
    ]
    if non_virtual:
        raise NotImplementedError(
            "paired temporal evaluation currently implements only the "
            "legacy virtual rig; {} uses real-six-camera stitch geometry. "
            "Do not compare it until the source-depth z-buffer co-visible "
            "protocol is implemented.".format(", ".join(non_virtual)))
    geometry_keys = (
        "half_height", "half_width", "target_height", "target_width",
        "target_fov_deg", "target_yaws_deg",
    )
    mismatches = [
        key for key in geometry_keys
        if baseline_training[key] != candidate_training[key]
    ]
    if mismatches:
        raise ValueError(
            "Paired evaluation requires matching render geometry: "
            + ", ".join(mismatches))

    torch.manual_seed(candidate_config["generator_seed"])
    device = torch.device(candidate_config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    process_group_started = False
    stage2_pipeline = None
    if args.latent_source == "stage2":
        process_group_started = ensure_single_rank_process_group(device)
        device = torch.device("cuda", 0)
        stage2_pipeline = build_stage2_pipeline(
            args.stage2_config, args.stage2_run, args.stage2_step,
            args.stage2_ema, device)

    baseline = build_gs_pipeline(
        baseline_config, args.baseline_run, args.baseline_step, device)
    candidate = build_gs_pipeline(
        candidate_config, args.candidate_run, args.candidate_step, device)
    # Both configs use the same SD3.5 VAE.  Release the duplicate after
    # construction; GT latents are encoded once and fed to both decoders.
    candidate.vae = baseline.vae
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fps_stride_tuples = [[10, 20]] if args.pkl is not None else None
    dataset_config = build_dataset_config(
        candidate_config[dataset_key],
        pkl_path=args.pkl,
        fps_stride_tuples=fps_stride_tuples)
    collate_fn = None
    latent_provider = None
    if args.latent_source == "stage2":
        install_strict_pano_hardseam_stub()
        dataset_config = build_g1_dataset_config(
            candidate_config[dataset_key],
            pkl_path=args.pkl,
            fps_stride_tuples=fps_stride_tuples)
        from dwm.datasets.common import CollateFnIgnoring
        collate_fn = CollateFnIgnoring(keys=["clip_text"])
        latent_provider = make_stage2_latent_provider(
            baseline, stage2_pipeline, "stage2")

    dataset = dwm.common.create_instance_from_config(dataset_config)
    clip_count = min(args.clip_limit, len(dataset))
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 0,
    }
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(clip_count)), **loader_kwargs)

    total = new_accumulator()
    per_clip = []
    for clip_index, batch in enumerate(loader):
        available_frames = int(batch["vae_images"].shape[1])
        frame_count = min(args.frame_count, available_frames)
        clip_accumulator = evaluate_clip(
            baseline, candidate, batch, frame_count, latent_provider)
        add_accumulator(total, clip_accumulator)
        clip_result = finalize(clip_accumulator)
        clip_result["clip_index"] = clip_index
        clip_result["frame_count"] = frame_count
        clip_result["frame_timestamps_us"] = [
            int(value) for value in
            batch["frame_timestamps_us"][0, :frame_count]
        ]
        per_clip.append(clip_result)
        print(
            "clip {}/{}: target support {:.3f}, XRenderGT L1 "
            "{:.4f}->{:.4f}, visibility recall {:.3f}->{:.3f}".format(
                clip_index + 1, clip_count,
                clip_result["coverage"]["target_static_valid"],
                clip_result["baseline"]["XRenderGT-L1"],
                clip_result["candidate"]["XRenderGT-L1"],
                clip_result["visibility_recall_on_target"]["baseline"],
                clip_result["visibility_recall_on_target"]["candidate"],
            ),
            flush=True,
        )

    result = {
        "protocol": {
            "name": "paired-adjacent-bidirectional-target-support-v2",
            "latent_source": args.latent_source,
            "dataset_source": args.dataset_source,
            "pkl": None if args.pkl is None else str(args.pkl),
            "clip_count": clip_count,
            "requested_frame_count": args.frame_count,
            "visibility": {
                "alpha_threshold": candidate_training.get(
                    "cross_min_alpha", 0.05),
                "absolute_depth_tolerance_m": candidate_training.get(
                    "cross_depth_abs_tolerance_m", 2.0),
                "relative_depth_tolerance": candidate_training.get(
                    "cross_depth_rel_tolerance", 0.1),
                "headline_reduction":
                    "candidate-independent target-static-valid support",
                "headline_limitation":
                    "target support is not a GT co-visible z-buffer and may "
                    "include true disocclusions",
                "diagnostics": [
                    "baseline rendered visibility",
                    "candidate rendered visibility",
                    "intersection of baseline/candidate rendered visibility",
                ],
            },
            "camera_origin_scale": {
                "baseline": baseline.evaluation_camera_origin_scale,
                "candidate": candidate.evaluation_camera_origin_scale,
            },
            "metric_direction": {
                "Own-L1": "lower",
                "Own-PSNR": "higher",
                "Own-SSIM": "higher",
                "XRenderGT-L1": "lower",
                "XRenderGT-L1-render-visible": "lower (diagnostic only)",
                "XRenderGT-L1-common": "lower",
                "XRenderSelf-L1-common": "lower",
            },
        },
        "checkpoints": {
            "baseline": str(
                args.baseline_run / "checkpoints" /
                f"{args.baseline_step}.pth"),
            "candidate": str(
                args.candidate_run / "checkpoints" /
                f"{args.candidate_step}.pth"),
        },
        "aggregate": finalize(total),
        "per_clip": per_clip,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(result, args.out)
    print(f"Wrote paired report: {args.out}", flush=True)
    if process_group_started:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
