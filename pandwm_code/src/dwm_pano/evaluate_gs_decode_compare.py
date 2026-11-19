"""Matched comparison of Stage-2 VAE RGB and Gaussian-rendered RGB.

The evaluator generates each clip exactly once, then sends the same Stage-2
latent through two output paths:

* SD3.5 VAE decode (the video generator without the Gaussian head), and
* the feed-forward Gaussian decoder followed by the configured own-view render.

Both outputs are compared with the same GT panorama on the same fixed geometry
support.  A GT-latent Gaussian arm is evaluated in the same process as the
decoder ceiling, and the Gaussian RGB is also compared directly with the VAE
RGB to isolate the change introduced downstream of the shared latent.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import einops
import torch
import torch.nn.functional as F
from PIL import Image

import dwm.common
from dwm_pano.evaluate_gs import (
    DepthMetricAccumulator,
    RGBMetricAccumulator,
    assert_dataset_frame_manifest,
    atomic_json_dump,
    build_dataset_config,
    build_g1_dataset_config,
    build_stage2_pipeline,
    decode_latents_to_images,
    ensure_single_rank_process_group,
    install_strict_pano_hardseam_stub,
    load_config,
    make_stage2_latent_provider,
    scalar_from_batch,
    select_middle_frame_indices,
    string_from_batch,
    validate_protocol,
)


REFERENCE_FRAME_COUNT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Stage-2 VAE RGB, Stage-2+GS RGB, and a GT-latent GS "
            "ceiling under one matched evaluation protocol"
        )
    )
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("-r", "--run-dir", required=True, type=Path)
    parser.add_argument("-s", "--step", required=True, type=int)
    parser.add_argument("--holdout-pkl", required=True, type=Path)
    parser.add_argument("--stage2-config", required=True, type=Path)
    parser.add_argument("--stage2-run", required=True, type=Path)
    parser.add_argument("--stage2-step", required=True, type=int)
    parser.add_argument("--stage2-ema", action="store_true")
    parser.add_argument(
        "--clip-limit",
        type=int,
        default=0,
        help="0 evaluates the complete fixed holdout; positive values cap it",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=19,
        help=(
            "Clip length used to enumerate the evaluation dataset.  The "
            "default reproduces the G2 holdout protocol; pass 35 for the "
            "frozen common149 anchors."
        ),
    )
    parser.add_argument(
        "--expected-clip-count",
        type=int,
        help=(
            "Fail unless the untruncated enumerated dataset has exactly this "
            "many clips.  This prevents a changed sampling protocol from "
            "being mislabeled as a fixed evaluation set."
        ),
    )
    parser.add_argument(
        "--anchor-reference-sequence-length",
        type=int,
        help=(
            "Restrict shorter inference windows to anchors enumerated with "
            "this reference length.  For example, use 19-frame inference "
            "with a 35-frame reference to preserve the frozen common149 "
            "scene/anchor identities without a 35-frame DiT attention pass."
        ),
    )
    parser.add_argument(
        "--gaussian-rgb-blend",
        type=float,
        default=0.0,
        help=(
            "Diagnostic only: replace each predicted Gaussian color by "
            "(1-blend)*predicted + blend*same-latent VAE RGB. Geometry, "
            "opacity, scale and rotation stay unchanged."
        ),
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help=(
            "Development-only RGB evaluation: do not load or report depth "
            "ground truth. This must not be used as a depth comparison."
        ),
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for name in ("step", "stage2_step"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.clip_limit < 0:
        parser.error("--clip-limit must be non-negative")
    if args.sequence_length < 4:
        parser.error("--sequence-length must be at least 4")
    if (args.expected_clip_count is not None and
            args.expected_clip_count < 1):
        parser.error("--expected-clip-count must be positive")
    if (args.anchor_reference_sequence_length is not None and
            args.anchor_reference_sequence_length < args.sequence_length):
        parser.error(
            "--anchor-reference-sequence-length must be at least "
            "--sequence-length"
        )
    if not 0.0 <= args.gaussian_rgb_blend <= 1.0:
        parser.error("--gaussian-rgb-blend must be in [0, 1]")
    return args


def blend_gaussian_rgb(
    gaussians: torch.Tensor,
    base_halves: torch.Tensor,
    blend: float,
) -> torch.Tensor:
    """Blend only Gaussian RGB with aligned full-resolution panorama halves."""
    if blend == 0.0:
        return gaussians
    base_rgb = einops.rearrange(
        base_halves, "b v c h w -> b (v h w) c"
    ).to(device=gaussians.device, dtype=gaussians.dtype)
    if base_rgb.shape != gaussians[..., 3:6].shape:
        raise ValueError(
            "aligned RGB shape {} does not match Gaussian RGB {}".format(
                tuple(base_rgb.shape), tuple(gaussians[..., 3:6].shape)
            )
        )
    output = gaussians.clone()
    output[..., 3:6] = (
        (1.0 - blend) * output[..., 3:6] + blend * base_rgb
    ).clamp(0.0, 1.0)
    return output


def simple_rgb_result(
    accumulator: RGBMetricAccumulator,
) -> dict[str, float]:
    raw = accumulator.result("matched")
    return {key.split("-", 1)[0]: value for key, value in raw.items()}


def simple_rgb_per_frame(
    accumulator: RGBMetricAccumulator,
) -> dict[str, dict[str, float]]:
    raw = accumulator.per_frame_result("matched")
    return {
        frame: {key.split("-", 1)[0]: value for key, value in values.items()}
        for frame, values in raw.items()
    }


def simple_depth_result(
    accumulator: DepthMetricAccumulator,
) -> dict[str, float]:
    raw = accumulator.result("matched")
    return {
        ("DepthL1" if key.startswith("DepthL1-") else "AbsRel"): value
        for key, value in raw.items()
    }


def simple_depth_per_frame(
    accumulator: DepthMetricAccumulator,
) -> dict[str, dict[str, float]]:
    raw = accumulator.per_frame_result("matched")
    return {
        frame: {
            ("DepthL1" if key.startswith("DepthL1-") else "AbsRel"): value
            for key, value in values.items()
        }
        for frame, values in raw.items()
    }


def average_selected_frames(
    per_frame: dict[str, dict[str, float]], frame_indices: list[int]
) -> dict[str, float]:
    selected = [per_frame[str(index)] for index in frame_indices]
    if not selected:
        raise RuntimeError("Cannot average an empty frame selection")
    keys = selected[0].keys()
    return {
        key: sum(values[key] for values in selected) / len(selected)
        for key in keys
    }


def metric_delta(
    minuend: dict[str, float], subtrahend: dict[str, float]
) -> dict[str, float]:
    return {key: minuend[key] - subtrahend[key] for key in minuend}


def resize_preview(image: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        image,
        size=(64, 1028),
        mode="bilinear",
        antialias=True,
    ).clamp(0.0, 1.0).cpu()


def save_preview(path: Path, panels: list[torch.Tensor]) -> None:
    if not panels:
        raise RuntimeError("No preview panels were collected")
    # Each panel contains four rows in this fixed order: GT, no-GS, GS, and
    # GT-latent GS.  Selected temporal frames are stacked vertically.
    grid = torch.cat(panels, dim=-2)[0]
    array = grid.permute(1, 2, 0).mul(255.0).round().byte().numpy()
    Image.fromarray(array).save(path)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    required_files = (
        args.config,
        args.holdout_pkl,
        args.stage2_config,
        args.run_dir / "checkpoints" / f"{args.step}.pth",
        args.stage2_run / "checkpoints" /
        f"{args.stage2_step}{'_ema' if args.stage2_ema else ''}.pth",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    config = load_config(args.config)
    torch.manual_seed(config["generator_seed"])
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    process_group_started = ensure_single_rank_process_group(device)
    device = torch.device("cuda", 0)
    stage2_pipeline = build_stage2_pipeline(
        args.stage2_config,
        args.stage2_run,
        args.stage2_step,
        args.stage2_ema,
        device,
    )

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"],
        output_path=str(args.run_dir),
        config=config,
        device=device,
        resume_from=None,
    )
    checkpoint_path = args.run_dir / "checkpoints" / f"{args.step}.pth"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError("GS checkpoint is not a raw decoder state dict")
    pipeline.decoder.load_state_dict(state_dict)
    pipeline.decoder.eval()
    pipeline.decoder_wrapper.eval()
    pipeline.vae.eval()
    if pipeline.perceptual_metric is not None:
        pipeline.perceptual_metric.eval()
    validate_protocol(config, pipeline)
    if not pipeline.uses_nuscenes6_stitch:
        raise RuntimeError(
            "The matched direct-RGB comparison currently requires the G2 "
            "real-six-camera stitch geometry"
        )
    if config["pipeline"]["training_config"].get(
        "evaluation_use_temporal_context", True
    ):
        raise RuntimeError(
            "This comparison is pinned to the current independent-frame G2 "
            "protocol; disable evaluation temporal context"
        )
    evaluation_camera_origin_scale = pipeline._camera_origin_scale(args.step)
    if evaluation_camera_origin_scale != 1.0:
        raise RuntimeError(
            "The current G2 comparison requires full real camera origins; "
            f"got scale {evaluation_camera_origin_scale}"
        )

    latent_provider = make_stage2_latent_provider(
        pipeline, stage2_pipeline, "stage2"
    )
    install_strict_pano_hardseam_stub()
    conditioned_config = build_g1_dataset_config(
        config["training_dataset"],
        pkl_path=args.holdout_pkl,
        fps_stride_tuples=[[10, 20]],
    )
    canonical_config = build_dataset_config(
        config["training_dataset"],
        pkl_path=args.holdout_pkl,
        fps_stride_tuples=[[10, 20]],
    )
    # The ordinary G2 reconstruction protocol uses 19-frame windows.  For
    # common149 we must instead retain the canonical 35-frame enumeration:
    # that is what fixes the 149 scene/anchor identities across methods.
    for dataset_config in (conditioned_config, canonical_config):
        dataset_config["base_dataset"]["sequence_length"] = \
            args.sequence_length
    dataset = dwm.common.create_instance_from_config(conditioned_config)
    canonical_dataset = dwm.common.create_instance_from_config(
        canonical_config
    )
    assert_dataset_frame_manifest(canonical_dataset, dataset, "holdout")
    anchor_reference_count = None
    if args.anchor_reference_sequence_length is not None:
        reference_config = copy.deepcopy(canonical_config)
        reference_config["base_dataset"]["sequence_length"] = \
            args.anchor_reference_sequence_length
        reference_dataset = dwm.common.create_instance_from_config(
            reference_config
        )

        def anchor_key(window: list[dict[str, Any]]) -> tuple[str, int]:
            return str(window[0]["scene_token"]), int(window[0]["timestamp"])

        candidate_keys = [
            anchor_key(window)
            for window in canonical_dataset.base_dataset.clips
        ]
        candidate_index = {key: index for index, key in enumerate(candidate_keys)}
        if len(candidate_index) != len(candidate_keys):
            raise RuntimeError("Duplicate short-window scene/anchor keys")
        reference_keys = [
            anchor_key(window)
            for window in reference_dataset.base_dataset.clips
        ]
        missing_keys = [key for key in reference_keys if key not in candidate_index]
        if missing_keys:
            raise RuntimeError(
                "Reference anchors were not found among inference windows: "
                f"first missing key {missing_keys[0]}"
            )
        dataset = torch.utils.data.Subset(
            dataset, [candidate_index[key] for key in reference_keys]
        )
        anchor_reference_count = len(reference_keys)
    clip_count = len(dataset)
    if (args.expected_clip_count is not None and
            clip_count != args.expected_clip_count):
        raise RuntimeError(
            "Evaluation clip count differs from the frozen protocol: "
            f"got {clip_count}, expected {args.expected_clip_count}"
        )
    if args.clip_limit:
        clip_count = min(clip_count, args.clip_limit)
        dataset = torch.utils.data.Subset(dataset, range(clip_count))
    if clip_count < 1:
        raise RuntimeError("The requested holdout contains no clips")

    from dwm.datasets.common import CollateFnIgnoring

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=CollateFnIgnoring(keys=["clip_text"]),
    )

    label = pipeline.supervision_output_label()
    rgb_metrics = {
        name: RGBMetricAccumulator(
            pipeline.perceptual_metric, protocol_label=label
        )
        for name in (
            "no_gs_vs_gt",
            "stage2_gs_vs_gt",
            "gt_latent_gs_vs_gt",
            "stage2_gs_vs_no_gs",
        )
    }
    depth_metrics = None if args.rgb_only else {
        "stage2_gs_vs_gt": DepthMetricAccumulator(),
        "gt_latent_gs_vs_gt": DepthMetricAccumulator(),
    }
    preview_panels: list[torch.Tensor] = []
    manifest: list[dict[str, Any]] = []
    selected_indices: list[int] | None = None

    for clip_index, batch in enumerate(loader):
        pipeline._prepare_nuscenes6_supervision(batch)
        frame_count = int(batch["vae_images"].shape[1])
        frame_indices = select_middle_frame_indices(
            frame_count, pipeline.training_config["frames_per_step"]
        )
        current_indices = [int(value) for value in frame_indices]
        if selected_indices is None:
            selected_indices = current_indices
        elif selected_indices != current_indices:
            raise RuntimeError("Selected frame indices changed across clips")

        half_images = batch["vae_images"][:, frame_indices].flatten(0, 1)
        half_images = half_images.to(device)
        if args.rgb_only:
            gt_ranges, gt_valid = None, None
        else:
            gt_ranges, gt_valid = pipeline._load_frame_ranges(
                batch, frame_indices, return_valid=True
            )
            gt_valid = gt_valid & pipeline.pixel_content_ray_valid.unsqueeze(0)
        latent_origins, pixel_origins = pipeline._frame_ray_origins(
            batch, frame_indices
        )

        clip_latents = latent_provider(batch)
        stage2_latents = clip_latents[:, frame_indices]
        decoded_halves = decode_latents_to_images(
            stage2_pipeline, stage2_latents
        )
        manifest.append(
            {
                "dataset_index": scalar_from_batch(batch["dataset_index"]),
                "scene_token": string_from_batch(batch["scene_token"]),
                "anchor_timestamp_us": scalar_from_batch(
                    batch["anchor_timestamp_us"]
                ),
                "selected_frame_indices": current_indices,
                "frame_timestamps_us": [
                    int(batch["frame_timestamps_us"][0, index])
                    for index in current_indices
                ],
            }
        )

        for local_index, frame_index in enumerate(current_indices):
            frame_images = half_images[local_index:local_index + 1]
            target_rgb = pipeline._target_rgb_for_supervision(frame_images)
            no_gs_rgb = pipeline._target_rgb_for_supervision(
                decoded_halves[:, local_index]
            )
            frame_latent_origins = None if latent_origins is None else \
                latent_origins[local_index:local_index + 1]
            frame_pixel_origins = None if pixel_origins is None else \
                pixel_origins[local_index:local_index + 1]

            def decode_gaussians(
                latents: torch.Tensor,
                base_halves: torch.Tensor,
            ) -> tuple[
                torch.Tensor, torch.Tensor, torch.Tensor
            ]:
                gaussians, predicted_range, _ = pipeline.decoder_wrapper(
                    latents,
                    pipeline.latent_directions.unsqueeze(0),
                    pipeline.pixel_directions.unsqueeze(0),
                    None,
                    previous_latents=None,
                    temporal_context_mask=0.0,
                    latent_origins=frame_latent_origins,
                    pixel_origins=frame_pixel_origins,
                    camera_origin_scale=evaluation_camera_origin_scale,
                )
                gaussians = blend_gaussian_rgb(
                    gaussians,
                    base_halves,
                    args.gaussian_rgb_blend,
                )
                rendered_rgb, _, _, render_valid = \
                    pipeline._render_own_supervision(gaussians[0])
                return rendered_rgb, predicted_range, render_valid

            stage2_gs_rgb, stage2_range, render_valid = decode_gaussians(
                stage2_latents[:, local_index],
                decoded_halves[:, local_index],
            )
            # GT posterior.sample() is kept for compatibility with the
            # established G1 "gt" arm.  fork_rng makes this diagnostic arm
            # unable to perturb the random stream used by later Stage-2 clips.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(
                    int(config["generator_seed"])
                    + clip_index * 1000
                    + frame_index
                )
                gt_latents = pipeline.encode_half_latents(frame_images)
            gt_gs_rgb, gt_range, gt_render_valid = decode_gaussians(
                gt_latents,
                frame_images,
            )
            if not torch.equal(render_valid, gt_render_valid):
                raise RuntimeError(
                    "Fixed stitch support changed between Gaussian arms"
                )

            rgb_metrics["no_gs_vs_gt"].update(
                no_gs_rgb, target_rgb, frame_index, render_valid
            )
            rgb_metrics["stage2_gs_vs_gt"].update(
                stage2_gs_rgb, target_rgb, frame_index, render_valid
            )
            rgb_metrics["gt_latent_gs_vs_gt"].update(
                gt_gs_rgb, target_rgb, frame_index, render_valid
            )
            rgb_metrics["stage2_gs_vs_no_gs"].update(
                stage2_gs_rgb, no_gs_rgb, frame_index, render_valid
            )
            if depth_metrics is not None:
                frame_range = gt_ranges[local_index:local_index + 1]
                frame_valid = gt_valid[local_index:local_index + 1]
                depth_metrics["stage2_gs_vs_gt"].update(
                    stage2_range, frame_range, frame_valid, frame_index
                )
                depth_metrics["gt_latent_gs_vs_gt"].update(
                    gt_range, frame_range, frame_valid, frame_index
                )

            if clip_index == 0:
                rows = [
                    pipeline.mask_supervision_for_display(value)
                    for value in (
                        target_rgb,
                        no_gs_rgb,
                        stage2_gs_rgb,
                        gt_gs_rgb,
                    )
                ]
                preview_panels.append(
                    torch.cat([resize_preview(value) for value in rows], dim=-2)
                )

        print(
            f"matched comparison: evaluated {clip_index + 1}/{clip_count} clips",
            flush=True,
        )

    if selected_indices is None:
        raise RuntimeError("No frames were evaluated")
    future_indices = [
        index for index in selected_indices if index >= REFERENCE_FRAME_COUNT
    ]
    rgb_per_frame = {
        name: simple_rgb_per_frame(accumulator)
        for name, accumulator in rgb_metrics.items()
    }
    depth_per_frame = {} if depth_metrics is None else {
        name: simple_depth_per_frame(accumulator)
        for name, accumulator in depth_metrics.items()
    }
    rgb_all = {
        name: simple_rgb_result(accumulator)
        for name, accumulator in rgb_metrics.items()
    }
    rgb_future = {
        name: average_selected_frames(rgb_per_frame[name], future_indices)
        for name in rgb_metrics
    }
    depth_all = {} if depth_metrics is None else {
        name: simple_depth_result(accumulator)
        for name, accumulator in depth_metrics.items()
    }
    depth_future = {} if depth_metrics is None else {
        name: average_selected_frames(depth_per_frame[name], future_indices)
        for name in depth_metrics
    }

    result = {
        "protocol": {
            "name": "matched-no-gs-vs-gs-vs-gt-v1",
            "geometry": pipeline.supervision_geometry,
            "output_label": label,
            "clip_count": clip_count,
            "clip_length": args.sequence_length,
            "anchor_reference_sequence_length": \
                args.anchor_reference_sequence_length,
            "anchor_reference_clip_count": anchor_reference_count,
            "selected_frame_indices": selected_indices,
            "gaussian_rgb_blend": args.gaussian_rgb_blend,
            "future_selected_frame_indices": future_indices,
            "reference_frame_count": REFERENCE_FRAME_COUNT,
            "same_stage2_latent_for_no_gs_and_gs": True,
            "depth_evaluated": not args.rgb_only,
            "temporal_context": False,
            "valid_support": "fixed calibrated real-six-camera stitch mask",
            "gt_latent_semantics": "SD3.5 VAE posterior.sample()",
            "no_gs_semantics": "same Stage-2 latent decoded by SD3.5 VAE",
            "gs_semantics": "same Stage-2 latent decoded by G2 and rendered",
            "preview_row_order": [
                "GT RGB",
                "Stage-2 VAE RGB (no GS)",
                "Stage-2 + G2 RGB",
                "GT latent + G2 RGB",
            ],
        },
        "checkpoints": {
            "stage2": str(
                args.stage2_run / "checkpoints" /
                f"{args.stage2_step}{'_ema' if args.stage2_ema else ''}.pth"
            ),
            "gaussian": str(checkpoint_path),
        },
        "all_selected_frames": {
            "rgb": rgb_all,
            "depth": depth_all,
            "gs_minus_no_gs_against_gt": metric_delta(
                rgb_all["stage2_gs_vs_gt"], rgb_all["no_gs_vs_gt"]
            ),
        },
        "future_selected_frames": {
            "rgb": rgb_future,
            "depth": depth_future,
            "gs_minus_no_gs_against_gt": metric_delta(
                rgb_future["stage2_gs_vs_gt"],
                rgb_future["no_gs_vs_gt"],
            ),
        },
        "per_frame": {
            "rgb": rgb_per_frame,
            "depth": depth_per_frame,
        },
        "manifest": manifest,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(result, args.out / "metrics.json")
    save_preview(args.out / "comparison_grid.png", preview_panels)
    print(json.dumps(result["all_selected_frames"], indent=2), flush=True)
    print(json.dumps(result["future_selected_frames"], indent=2), flush=True)
    print(f"Wrote matched comparison to: {args.out}", flush=True)

    if process_group_started:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
