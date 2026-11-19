"""Export a controlled before/after antialiased-GS video for one panorama clip.

The Stage-2 4-step latents are sampled once, then reused unchanged by both
arms.  The ``classic`` arm loads the G2 checkpoint immediately before the AA
adaptation and uses classic rasterization; the ``aa`` arm loads the final
AA-compatible checkpoint and uses antialiased rasterization with eps2d=0.1.
This is a qualitative diagnostic, not an FID/FVD evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import dwm.common
from dwm.utils.preview import save_tensor_to_video
from dwm_pano.evaluate_gs import (
    build_g1_dataset_config,
    build_stage2_pipeline,
    ensure_single_rank_process_group,
    install_strict_pano_hardseam_stub,
    load_config,
    make_stage2_latent_provider,
    validate_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export matched classic-GS versus AA-GS qualitative video"
    )
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--data-pkl", required=True, type=Path)
    parser.add_argument("--stage2-config", required=True, type=Path)
    parser.add_argument("--stage2-run", required=True, type=Path)
    parser.add_argument("--stage2-step", required=True, type=int)
    parser.add_argument("--stage2-ema", action="store_true")
    parser.add_argument("--classic-checkpoint", required=True, type=Path)
    parser.add_argument("--aa-checkpoint", required=True, type=Path)
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.clip_index < 0:
        parser.error("--clip-index must be non-negative")
    if args.stage2_step < 1:
        parser.error("--stage2-step must be positive")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_decoder(pipeline: Any, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"{checkpoint} is not a raw GS decoder state dict")
    pipeline.decoder.load_state_dict(state, strict=True)
    pipeline.decoder.eval()
    pipeline.decoder_wrapper.eval()


def render_arm(
    pipeline: Any,
    batch: dict[str, Any],
    clip_latents: torch.Tensor,
    checkpoint: Path,
    rasterize_mode: str,
    eps2d: float,
) -> list[torch.Tensor]:
    load_decoder(pipeline, checkpoint)
    pipeline.rasterize_mode = rasterize_mode
    pipeline.active_rasterize_eps2d = eps2d
    rows: list[torch.Tensor] = []
    previous_latents = None
    frame_count = int(clip_latents.shape[1])
    for frame_index in range(frame_count):
        selector = torch.tensor([frame_index], dtype=torch.long)
        latent_origins, pixel_origins = pipeline._frame_ray_origins(
            batch, selector)
        latents = clip_latents[:, frame_index]
        gaussians, _, _ = pipeline.decoder_wrapper(
            latents,
            pipeline.latent_directions.unsqueeze(0),
            pipeline.pixel_directions.unsqueeze(0),
            None,
            previous_latents=previous_latents,
            temporal_context_mask=0.0,
            latent_origins=latent_origins,
            pixel_origins=pixel_origins,
            camera_origin_scale=1.0,
        )
        # The G2 evaluation protocol is independent-frame; do not carry a
        # decoder temporal state between frames in either arm.
        previous_latents = None
        rendered_rgb, _, _, _ = pipeline._render_own_supervision(gaussians[0])
        rendered = pipeline.mask_supervision_for_display(rendered_rgb)
        rows.append(torch.cat(list(rendered), dim=-1).clamp(0.0, 1.0).cpu())
    return rows


@torch.no_grad()
def main() -> None:
    args = parse_args()
    required = (
        args.config, args.data_pkl, args.stage2_config,
        args.stage2_run / "checkpoints" /
        f"{args.stage2_step}{'_ema' if args.stage2_ema else ''}.pth",
        args.classic_checkpoint, args.aa_checkpoint,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)

    config = load_config(args.config)
    torch.manual_seed(int(config["generator_seed"]))
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    process_group_started = ensure_single_rank_process_group(device)
    device = torch.device("cuda", 0)
    try:
        stage2_pipeline = build_stage2_pipeline(
            args.stage2_config, args.stage2_run, args.stage2_step,
            args.stage2_ema, device)
        pipeline = dwm.common.create_instance_from_config(
            config["pipeline"], output_path=str(args.out), config=config,
            device=device, resume_from=None)
        pipeline.vae.eval()
        validate_protocol(config, pipeline)
        if pipeline.training_config.get("evaluation_use_temporal_context", True):
            raise RuntimeError("A/B video is pinned to independent-frame G2")
        latent_provider = make_stage2_latent_provider(
            pipeline, stage2_pipeline, "stage2")
        install_strict_pano_hardseam_stub()
        dataset_config = build_g1_dataset_config(
            config["training_dataset"], pkl_path=args.data_pkl,
            fps_stride_tuples=[[10, 20]])
        dataset_config["base_dataset"]["sequence_length"] = 19
        dataset = dwm.common.create_instance_from_config(dataset_config)
        if args.clip_index >= len(dataset):
            raise IndexError(
                f"clip index {args.clip_index} >= dataset length {len(dataset)}")
        from dwm.datasets.common import CollateFnIgnoring
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, [args.clip_index]), batch_size=1,
            shuffle=False, num_workers=0,
            collate_fn=CollateFnIgnoring(keys=["clip_text"]))
        batch = next(iter(loader))
        pipeline._prepare_nuscenes6_supervision(batch)
        clip_latents = latent_provider(batch)
        if int(clip_latents.shape[1]) != 19:
            raise RuntimeError("Expected a 19-frame Stage-2 window")

        target_rows = []
        for frame_index in range(19):
            # Keep an explicit temporal dimension before flattening B×T.  A
            # direct integer index would collapse T and make the downstream
            # real-six-camera stitch mistake views for batch elements.
            selector = torch.tensor([frame_index], dtype=torch.long)
            half_images = batch["vae_images"][:, selector].flatten(0, 1)
            target_rgb = pipeline._target_rgb_for_supervision(
                half_images.to(device))
            target_rows.append(torch.cat(list(
                pipeline.mask_supervision_for_display(target_rgb)), dim=-1
            ).clamp(0.0, 1.0).cpu())

        classic_rows = render_arm(
            pipeline, batch, clip_latents, args.classic_checkpoint,
            rasterize_mode="classic", eps2d=0.0)
        aa_rows = render_arm(
            pipeline, batch, clip_latents, args.aa_checkpoint,
            rasterize_mode="antialiased", eps2d=0.1)
        grids = [torch.cat((target, classic, aa), dim=-2)
                 for target, classic, aa in zip(
                     target_rows, classic_rows, aa_rows)]
        save_tensor_to_video(
            str(args.out / "gs_before_after_triplet.mp4"), "libx264",
            float(batch["fps"][0]), grids)
        # These two videos retain the target/output two-row form used by the
        # existing GS previews, while the triplet supports direct comparison.
        save_tensor_to_video(
            str(args.out / "gs_classic_two_row.mp4"), "libx264",
            float(batch["fps"][0]),
            [torch.cat((target, classic), dim=-2)
             for target, classic in zip(target_rows, classic_rows)])
        save_tensor_to_video(
            str(args.out / "gs_aa_two_row.mp4"), "libx264",
            float(batch["fps"][0]),
            [torch.cat((target, aa), dim=-2)
             for target, aa in zip(target_rows, aa_rows)])

        record = {
            "purpose": "qualitative GS anti-aliasing before/after video",
            "data_pkl": str(args.data_pkl),
            "clip_index": args.clip_index,
            "dataset_index": int(batch["dataset_index"][0]),
            "scene_token": batch["scene_token"][0],
            "anchor_timestamp_us": int(batch["anchor_timestamp_us"][0]),
            "frames": 19,
            "fps": float(batch["fps"][0]),
            "stage2": {
                "checkpoint": str(required[3]),
                "checkpoint_sha256": sha256(required[3]),
                "sampler": "4-step stochastic [1000, 625, 250, 42]",
            },
            "classic_arm": {
                "checkpoint": str(args.classic_checkpoint),
                "checkpoint_sha256": sha256(args.classic_checkpoint),
                "rasterize_mode": "classic",
                "eps2d": 0.0,
            },
            "aa_arm": {
                "checkpoint": str(args.aa_checkpoint),
                "checkpoint_sha256": sha256(args.aa_checkpoint),
                "rasterize_mode": "antialiased",
                "eps2d": 0.1,
            },
            "triplet_row_order": ["Ground truth", "GS classic", "GS AA"],
            "note": (
                "The two arms reuse identical Stage-2 latents.  This is a "
                "qualitative video, not a FID/FVD measurement."
            ),
        }
        (args.out / "metadata.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2), flush=True)
    finally:
        if process_group_started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
