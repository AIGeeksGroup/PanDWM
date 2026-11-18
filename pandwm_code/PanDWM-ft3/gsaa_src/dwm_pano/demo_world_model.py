"""End-to-end world model demo: generate the future, then look around in it.

This is the join between the two halves that already existed separately:

- `evaluate_gs.py` runs the real Stage-2 4-step autoregressive pipeline and
  hands the generated latents straight to the Gaussian decoder, but it only
  ever renders the supervised camera rig and reports metrics.
- `infer_pano_gs_decoder.py` renders novel viewpoints (lateral ego shifts)
  through `build_virtual_camera`, but it feeds the decoder ground-truth VAE
  latents, so nothing is ever generated.

Running both in one command is what makes it a world model demo rather than
two diagnostics: 3 context frames go in, Stage 2 imagines the future, the
Gaussian decoder lifts each imagined frame to 3D, and the result is
rasterized from viewpoints the model was never shown.

Render targets
--------------
``--render-target supervised`` uses the checkpoint's own training protocol.
For G2 this means real six-camera rendering followed by the differentiable
OneBEV-compatible hard seam; it is the only pixel-aligned choice for
``--save-gt-views`` under G2.

``--render-target virtual`` keeps the supervised rig: equi-spaced yaws from
`target_yaws_deg`, optionally displaced sideways by ``--lateral-shifts-m``.
It is a novel-view diagnostic rather than the G2 supervised rig.

``--render-target nuscenes6`` rasterizes at the **real** nuScenes six-camera
calibration instead -- per-camera intrinsics and `sensor2ego` pose read from
the pkl and matched to each frame by timestamp. This is the render target an
`FVD-I3D-camera-6-T` number needs, because that metric compares against the
original camera images.

  Caveat worth stating up front: the panorama the model generates is a 33
  degree elevation band (+/- 16.67 deg), while a real nuScenes camera sees
  far more vertically. A nuscenes6 render is therefore only partly
  populated. That is the model's actual field of view, not a bug -- but any
  metric computed against full camera frames is dominated by the missing
  sky and road.

  Do NOT "crop to a fixed row band" to fix that. The band is defined around
  rays leaving the EGO ORIGIN (z = 0), while a real nuScenes camera sits
  ~1.5 m above it, so the band's image of the world lands on a
  range-dependent set of rows: for the upper edge, elevation -12.8 deg at
  3 m, +8.1 deg at 10 m, +15.0 deg at 50 m. A fixed crop therefore both
  includes empty regions and excludes populated ones. Deciding the right
  comparison region is unfinished work, and is one of the reasons this mode
  is not yet an `FVD-I3D-camera-6-T`.

Usage::

    python3 -m dwm_pano.demo_world_model \
        --gs-config configs/ctsd/pandwm/h20/gs_decoder_prod_rd0p01_rl0p01.json \
        --gs-run /path/to/pandwm_root/runs/gs_prod_rd0p01_rl0p01 --gs-step 15000 \
        --stage2-config configs/ctsd/pandwm/h20/cd_eval_h20.json \
        --stage2-run /path/to/pandwm_root/runs/cd_baseline --stage2-step 10000 \
        --stage2-ema --clip-index 0 --lateral-shifts-m 0.0 1.0 2.0 \
        --out-dir /path/to/pandwm_root/demos/clip0
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import dwm.common
from dwm_pano.calibrated_stitch import (
    camera_records_to_arrays,
    prepare_gsplat_intrinsics,
)
from dwm_pano.gs_geometry import build_virtual_camera


# Same order `PanoPatchFrechetVideoDistance` emits its six Voronoi patches in
# (see metrics/fvd_pano.py). Keeping the two identical means the panels of a
# nuscenes6 render can be paired positionally with the FVD patch stack; a
# different order here would silently pair CAM_FRONT with a CAM_FRONT_RIGHT
# patch when someone slices the 6-panel row.
NUSCENES_CAM_ORDER = [
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
]

# Every nuScenes camera frame is 1600x900. This cannot be derived from the
# intrinsics: the principal point is not the image center (CAM_FRONT sits at
# 816.27, 491.51), so doubling it would give 1632x983 and silently shift
# every projection.
NUSCENES_IMAGE_WIDTH = 1600
NUSCENES_IMAGE_HEIGHT = 900


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate future frames and render them from new views")
    parser.add_argument("--gs-config", type=Path, required=True)
    parser.add_argument("--gs-run", type=Path, required=True)
    parser.add_argument("--gs-step", type=int, required=True)
    parser.add_argument("--stage2-config", type=Path, default=None)
    parser.add_argument("--stage2-run", type=Path, default=None)
    parser.add_argument("--stage2-step", type=int, default=None)
    parser.add_argument("--stage2-ema", action="store_true")
    parser.add_argument(
        "--latent-source", choices=["stage2", "gt"], default="stage2",
        help="stage2 generates the future; gt is the upper-bound control")
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument(
        "--frame-count", type=int, default=None,
        help="frames to render; default is the whole clip")
    parser.add_argument(
        "--lateral-shifts-m", type=float, nargs="+", default=[0.0, 1.0, 2.0],
        help="ego +Y offsets, virtual render target only")
    parser.add_argument(
        "--render-target", choices=["supervised", "virtual", "nuscenes6"],
        default="virtual")
    parser.add_argument(
        "--calibration-pkl", type=Path, default=None,
        help="pkl holding real camera calibration; required for nuscenes6")
    parser.add_argument(
        "--nuscenes-width", type=int, default=NUSCENES_IMAGE_WIDTH)
    parser.add_argument(
        "--nuscenes-height", type=int, default=NUSCENES_IMAGE_HEIGHT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save-panos", action="store_true",
        help="also decode the latents back to panoramas for reference")
    parser.add_argument(
        "--save-gt-views", action="store_true",
        help=(
            "also save the held-out RGB frame resampled at the unshifted "
            "virtual six-camera rig for pixel-aligned GT-vs-render videos"))
    args = parser.parse_args()

    if args.latent_source == "stage2":
        missing = [
            name for name, value in (
                ("--stage2-config", args.stage2_config),
                ("--stage2-run", args.stage2_run),
                ("--stage2-step", args.stage2_step),
            ) if value is None
        ]
        if missing:
            parser.error(
                "--latent-source stage2 requires " + ", ".join(missing))
    if args.render_target == "nuscenes6" and args.calibration_pkl is None:
        parser.error("--render-target nuscenes6 requires --calibration-pkl")
    if args.save_gt_views and args.render_target == "nuscenes6":
        parser.error(
            "--save-gt-views is not aligned to six raw camera panels; use "
            "--render-target supervised")
    return args


def collate_single(item):
    batch = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = [value]
    return batch


def virtual_camera_stack(pipeline, lateral_shift_m, device):
    """The supervised rig, optionally displaced sideways."""
    training_config = pipeline.training_config
    c2w_list = []
    for yaw_deg in training_config["target_yaws_deg"]:
        c2w, _ = build_virtual_camera(
            yaw_deg, training_config["target_height"],
            training_config["target_width"],
            training_config["target_fov_deg"], lateral_shift_m)
        c2w_list.append(c2w.astype(np.float32))
    return torch.from_numpy(np.stack(c2w_list)).to(device)


def load_calibration_index(pkl_path):
    """Map frame timestamp (us) -> that frame's six-camera calibration."""
    with open(pkl_path, "rb") as handle:
        payload = pickle.load(handle)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    index = {}
    for info in infos:
        cams = info.get("cams")
        if cams:
            index[int(info["timestamp"])] = cams
    if not index:
        raise ValueError("no camera calibration found in {}".format(pkl_path))
    return index


def nuscenes_camera_stack(cams, device, target_size):
    """Real six-camera c2w and gsplat K at ``target_size=(H,W)``.

    nuScenes camera axes are already x-right / y-down / z-forward, the same
    convention `build_virtual_camera` emits, so `sensor2ego` drops straight
    into the c2w rotation with no remap. Dataset K is OpenCV pixel-index K;
    convert its principal point before resizing it for gsplat.
    """
    transforms, raw_intrinsics, image_sizes = camera_records_to_arrays(
        cams, camera_order=NUSCENES_CAM_ORDER)
    if not np.all(image_sizes == image_sizes[0]):
        raise ValueError("all six nuScenes cameras must share image size")
    native_width, native_height = [int(value) for value in image_sizes[0]]
    intrinsics = prepare_gsplat_intrinsics(
        torch.from_numpy(raw_intrinsics).to(device),
        (native_height, native_width), target_size)
    return torch.from_numpy(transforms).to(device), intrinsics


def to_uint8_image(tensor):
    """(C, H, W) float in [0, 1] -> (H, W, C) uint8, rounded.

    Rounded, not truncated: `evaluate_gs.save_grid` uses
    `.mul(255).round().byte()`, and a PSNR computed off a truncated PNG
    carries a systematic -0.5 LSB offset against the in-tensor value, so the
    two never reconcile.
    """
    array = tensor.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def save_view_row(rendered_rgb, path):
    row = torch.cat(list(rendered_rgb), dim=-1)
    Image.fromarray(to_uint8_image(row)).save(path)


def write_manifest(
    out_dir, args, reference_frame_count, frame_count, fps, horizon_s,
    checkpoint_path, renders
):
    with open(
        os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "clip_index": args.clip_index,
            "latent_source": args.latent_source,
            "render_target": args.render_target,
            "reference_frame_count": reference_frame_count,
            "frame_count": frame_count,
            "fps": fps,
            "generated_horizon_s": horizon_s,
            "gs_checkpoint": str(checkpoint_path),
            "stage2_step": args.stage2_step,
            "stage2_ema": args.stage2_ema,
            "renders": renders,
        }, handle, indent=2)


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from dwm_pano.evaluate_gs import (
        build_dataset_config,
        build_g1_dataset_config,
        build_stage2_pipeline,
        decode_latents_to_images,
        ensure_single_rank_process_group,
        install_strict_pano_hardseam_stub,
        make_stage2_latent_provider,
    )

    with open(args.gs_config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    torch.manual_seed(config["generator_seed"])
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    stage2_pipeline = None
    if args.latent_source == "stage2":
        # The Stage-2 stack is built exactly as evaluate_gs.py builds it, so
        # the demo and the G1 numbers come from the same inference path.
        ensure_single_rank_process_group(device)
        device = torch.device("cuda", 0)
        suffix = "_ema" if args.stage2_ema else ""
        stage2_checkpoint = args.stage2_run / "checkpoints" / \
            "{}{}.pth".format(args.stage2_step, suffix)
        if not stage2_checkpoint.is_file():
            raise FileNotFoundError(
                "Stage-2 checkpoint is missing: {}".format(stage2_checkpoint))
        stage2_pipeline = build_stage2_pipeline(
            args.stage2_config, args.stage2_run, args.stage2_step,
            args.stage2_ema, device)

    checkpoint_path = args.gs_run / "checkpoints" / "{}.pth".format(
        args.gs_step)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "GS checkpoint is missing: {}".format(checkpoint_path))

    # The same two protocol guards evaluate_gs.py applies before making the
    # identical decoder call. Without them a config with `use_range_prior`
    # crashes minutes in (the demo passes `None` as the prior), and one with
    # `latent_refinement.enabled` loads a whole frozen CD DiT onto the GPU
    # that the demo never calls -- so the render silently omits the
    # refinement the checkpoint expects, and wastes several GB doing it.
    decoder_config = config["pipeline"]["decoder"]
    if decoder_config.get("use_range_prior") is not False:
        raise RuntimeError(
            "this demo passes no range prior; the config must set "
            "decoder.use_range_prior to false")
    refiner_config = config["pipeline"].get("latent_refinement")
    if refiner_config is not None and refiner_config.get("enabled"):
        raise RuntimeError(
            "latent_refinement is enabled in the config but this demo never "
            "calls refine_clip_latents; the render would silently skip it")

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=str(args.gs_run), config=config,
        device=device, resume_from=None)
    pipeline.decoder.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu"))
    pipeline.decoder.eval()
    pipeline.decoder_wrapper.eval()
    pipeline.vae.eval()
    if args.save_gt_views and args.render_target == "virtual" \
            and pipeline.uses_nuscenes6_stitch:
        raise RuntimeError(
            "G2 stitched GT is not aligned to the virtual central rig; use "
            "--render-target supervised for a pixel-aligned comparison")
    print("loaded GS decoder: {}".format(checkpoint_path), flush=True)

    latent_provider = None
    if args.latent_source == "stage2":
        latent_provider = make_stage2_latent_provider(
            pipeline, stage2_pipeline, "stage2")
        install_strict_pano_hardseam_stub()
        dataset_config = build_g1_dataset_config(config["validation_dataset"])
    else:
        dataset_config = build_dataset_config(config["validation_dataset"])
    dataset = dwm.common.create_instance_from_config(dataset_config)
    if not 0 <= args.clip_index < len(dataset):
        raise IndexError(
            "clip {} out of range for {} clips".format(
                args.clip_index, len(dataset)))
    batch = collate_single(dataset[args.clip_index])
    pipeline._prepare_nuscenes6_supervision(batch)

    clip_length = batch["vae_images"].shape[1]
    frame_count = clip_length if args.frame_count is None \
        else min(args.frame_count, clip_length)
    # Read off the provider rather than keeping a second copy of the
    # constant: the value that actually decides which frames are generated
    # lives in make_stage2_latent_provider.
    reference_frame_count = (
        getattr(latent_provider, "reference_frame_count", 3)
        if latent_provider is not None else 0)
    if frame_count <= reference_frame_count:
        raise ValueError(
            "--frame-count {} renders only context frames ({} of them); a "
            "world model demo that generates nothing is not a demo".format(
                frame_count, reference_frame_count))
    fps = float(batch["fps"][0]) if "fps" in batch else float("nan")
    horizon_s = (
        (frame_count - reference_frame_count) / fps if fps == fps and fps > 0
        else float("nan"))
    print(
        "clip {}: {} frames, {} context + {} generated, fps {:.2f} "
        "(~{:.2f}s of imagined future)".format(
            args.clip_index, clip_length, reference_frame_count,
            frame_count - reference_frame_count, fps, horizon_s),
        flush=True)

    if latent_provider is not None:
        clip_latents = latent_provider(batch)
    else:
        # One frame at a time. encode_half_latents flattens its leading dim
        # into a single vae.encode with no chunking, so handing it the whole
        # clip would put 19x2 = 38 halves through at once -- an ~11 GB
        # activation at this geometry. Every other call site in the repo
        # encodes per frame for exactly this reason.
        vae_images = batch["vae_images"]
        frame_latents = [
            # (B, 2, 3, H, W) -> (B, 2, C, h, w)
            pipeline.encode_half_latents(vae_images[:, t].to(device))
            for t in range(clip_length)
        ]
        clip_latents = torch.stack(frame_latents, dim=1)

    calibration_index = None
    if args.render_target == "nuscenes6":
        if (args.nuscenes_width, args.nuscenes_height) != (
            NUSCENES_IMAGE_WIDTH, NUSCENES_IMAGE_HEIGHT
        ):
            print(
                "[info] rendering at {}x{} instead of native nuScenes "
                "{}x{}; calibrated intrinsics will be converted and scaled"
                .format(
                    args.nuscenes_width, args.nuscenes_height,
                    NUSCENES_IMAGE_WIDTH, NUSCENES_IMAGE_HEIGHT),
                flush=True)
        calibration_index = load_calibration_index(args.calibration_pkl)
        print(
            "calibration frames indexed: {}".format(len(calibration_index)),
            flush=True)

    virtual_stacks = None
    if args.render_target == "virtual":
        virtual_stacks = {
            shift: virtual_camera_stack(pipeline, shift, device)
            for shift in args.lateral_shifts_m
        }

    manifest = []
    previous_latents = None
    use_temporal_context = pipeline.training_config.get(
        "demo_use_temporal_context", True)
    ray_frame_indices = torch.arange(frame_count, dtype=torch.long)
    latent_origins, pixel_origins = pipeline._frame_ray_origins(
        batch, ray_frame_indices)
    for frame_index in range(frame_count):
        latents = clip_latents[:, frame_index]
        gaussians, _, _ = pipeline.decoder_wrapper(
            latents,
            pipeline.latent_directions.unsqueeze(0),
            pipeline.pixel_directions.unsqueeze(0),
            None,
            previous_latents=(
                previous_latents if use_temporal_context else None),
            temporal_context_mask=(
                1.0 if use_temporal_context
                and previous_latents is not None else 0.0),
            latent_origins=None if latent_origins is None else
                latent_origins[frame_index:frame_index + 1],
            pixel_origins=None if pixel_origins is None else
                pixel_origins[frame_index:frame_index + 1])
        previous_latents = (
            latents.detach() if use_temporal_context else None)
        role = "context" if frame_index < reference_frame_count \
            else "generated"

        gt_views_name = None
        if args.save_gt_views:
            # Use the pipeline's active supervision protocol.  In G2 this is
            # the 4114-wide stitched panorama; in legacy runs it is the six
            # virtual views.  A decoded latent is not ground truth and must
            # never be labelled as such in a presentation video.
            frame_images = batch["vae_images"][:, frame_index].to(device)
            gt_views = pipeline._target_rgb_for_supervision(frame_images)
            gt_views = pipeline.mask_supervision_for_display(gt_views)
            gt_geometry = (
                "stitch" if pipeline.uses_nuscenes6_stitch else "virtual")
            gt_views_name = "f{:02d}_{}_gt_{}.png".format(
                frame_index, role, gt_geometry)
            save_view_row(
                gt_views, os.path.join(args.out_dir, gt_views_name))

        if args.save_panos:
            pano = decode_latents_to_images(
                pipeline, clip_latents[:, frame_index:frame_index + 1])
            # (B, T, V, C, H, W) -> the two 2304-wide halves stacked
            # vertically. Named "halves", not "pano", because it is NOT the
            # project's canonical panorama: that is the V=1, 4114-wide
            # de-overlapped stitch that fvd_pano.py consumes. Feeding this
            # file to PanoPatchFrechetVideoDistance would slice a 2304-wide
            # double-decker at boundaries computed for 4114.
            halves = pano[0, 0]
            stacked = torch.cat(list(halves), dim=-2)
            Image.fromarray(to_uint8_image(stacked)).save(
                os.path.join(
                    args.out_dir,
                    "halves_f{:02d}_{}.png".format(frame_index, role)))

        if args.render_target == "supervised":
            rendered_rgb, _, _, _ = \
                pipeline._render_own_supervision(gaussians[0])
            rendered_rgb = pipeline.mask_supervision_for_display(rendered_rgb)
            name = "f{:02d}_{}_supervised.png".format(frame_index, role)
            save_view_row(rendered_rgb, os.path.join(args.out_dir, name))
            manifest.append({
                "frame_index": frame_index,
                "role": role,
                "render_target": "supervised",
                "supervision_geometry": pipeline.supervision_geometry,
                "invalid_stitch_pixels": "masked_black_on_both_sides",
                "file": name,
                "ground_truth_file": gt_views_name,
            })
        elif args.render_target == "virtual":
            for shift, camera_stack in virtual_stacks.items():
                rendered_rgb, _, _ = pipeline.render_targets(
                    gaussians[0], target_c2w=camera_stack)
                name = "f{:02d}_{}_shift{:+.1f}m.png".format(
                    frame_index, role, shift)
                save_view_row(rendered_rgb, os.path.join(args.out_dir, name))
                manifest.append({
                    "frame_index": frame_index, "role": role,
                    "render_target": "virtual", "lateral_shift_m": shift,
                    "file": name,
                    "ground_truth_file": (
                        gt_views_name if float(shift) == 0.0 else None),
                })
        else:
            timestamps = batch["frame_timestamps_us"][0][frame_index]
            timestamp = int(
                timestamps.flatten()[0] if torch.is_tensor(timestamps)
                else timestamps)
            if timestamp not in calibration_index:
                raise KeyError(
                    "frame timestamp {} has no calibration in {}; the demo "
                    "clip and the calibration pkl are not the same "
                    "split".format(timestamp, args.calibration_pkl))
            cams = calibration_index[timestamp]
            width = args.nuscenes_width
            height = args.nuscenes_height
            camera_stack, camera_intrinsics = nuscenes_camera_stack(
                cams, device, (height, width))
            rendered_rgb, _, _ = pipeline.render_targets(
                gaussians[0], target_c2w=camera_stack,
                target_intrinsics=camera_intrinsics,
                width=width, height=height)
            name = "f{:02d}_{}_nuscenes6.png".format(frame_index, role)
            save_view_row(rendered_rgb, os.path.join(args.out_dir, name))
            manifest.append({
                "frame_index": frame_index, "role": role,
                "render_target": "nuscenes6", "timestamp_us": timestamp,
                "cam_order": NUSCENES_CAM_ORDER,
                "width": width, "height": height, "file": name})

        # Rewritten after every frame, not once at the end: an OOM at frame
        # 15 would otherwise leave 15 valid PNGs with no manifest, after a
        # multi-minute Stage-2 generation that cannot be resumed.
        write_manifest(
            args.out_dir, args, reference_frame_count, frame_count, fps,
            horizon_s, checkpoint_path, manifest)

        print("  frame {:02d} ({}) rendered".format(frame_index, role),
              flush=True)

    print("wrote {} renders and {}".format(
        len(manifest), os.path.join(args.out_dir, "manifest.json")))

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
