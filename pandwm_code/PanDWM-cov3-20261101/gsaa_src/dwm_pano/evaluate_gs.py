"""Standalone P2 and G1 evaluation for the panoramic Gaussian decoder."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

try:
    import h5py
except ModuleNotFoundError:  # Keep --help usable on the CPU-only laptop.
    h5py = None

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure

import dwm.common
from dwm_pano.datasets.pano_projection import (
    PANO_HALF_LEFT_RANGE,
    PANO_HALF_RIGHT_RANGE,
)


HOLDOUT_SUFFIX = "holdout5"
VALIDATION_SUFFIX = "val32"
GRID_VIEW_HEIGHT = 75
GRID_VIEW_WIDTH = 200
ARM_SUFFIXES = {
    "gt": "gt",
    "stage2": "stage2ref3",
    "stage2_reencode": "stage2ref3reencode",
    "gt_reencode": "gtreencode",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a P2 GS decoder checkpoint on holdout and val."
    )
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("-r", "--run-dir", required=True, type=Path)
    parser.add_argument("-s", "--step", required=True, type=int)
    parser.add_argument("--holdout-pkl", required=True, type=Path)
    parser.add_argument("--val-limit", default=32, type=int)
    parser.add_argument(
        "--latent-source",
        choices=("gt", "stage2", "stage2_reencode", "gt_reencode"),
        default="gt",
        help="Latent arm used by the G1 domain-gap evaluation.",
    )
    parser.add_argument("--stage2-config", type=Path)
    parser.add_argument("--stage2-run", type=Path)
    parser.add_argument("--stage2-step", type=int)
    parser.add_argument(
        "--stage2-ema",
        action="store_true",
        help="Load the Stage-2 EMA checkpoint instead of the raw checkpoint.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Evaluate one holdout and one validation clip for a smoke run.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output directory (default: RUN_DIR/eval_gs/STEP).",
    )
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step must be positive")
    minimum_val_limit = 1 if args.smoke else 2
    if args.val_limit < minimum_val_limit:
        parser.error(
            f"--val-limit must be at least {minimum_val_limit} for the grid")
    if args.stage2_step is not None and args.stage2_step < 1:
        parser.error("--stage2-step must be positive")
    if args.latent_source in ("stage2", "stage2_reencode"):
        missing = [
            name for name in ("stage2_config", "stage2_run", "stage2_step")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "non-GT latent sources require: " + ", ".join(
                    "--" + name.replace("_", "-") for name in missing))
    if args.out is None:
        if args.latent_source == "gt":
            args.out = args.run_dir / "eval_gs" / str(args.step)
        else:
            args.out = args.run_dir / "eval_gs_g1" / str(args.step) / \
                args.latent_source
    return args


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def build_dataset_config(
    source: dict[str, Any],
    pkl_path: Path | None = None,
    fps_stride_tuples: list[list[float]] | None = None,
) -> dict[str, Any]:
    dataset_config = copy.deepcopy(source)
    base_config = dataset_config["base_dataset"]
    if pkl_path is not None:
        base_config["pkl_path"] = str(pkl_path)
    if fps_stride_tuples is not None:
        base_config["fps_stride_tuples"] = fps_stride_tuples
    return dataset_config


def build_g1_dataset_config(
    source: dict[str, Any],
    pkl_path: Path | None = None,
    fps_stride_tuples: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Add the Stage-2 conditioning fields to a GS dataset config.

    The GS training and validation configs are the source of truth for clip
    enumeration.  Only fields consumed by the CD condition builder are added,
    so the RGB, timestamps, and frame selection remain identical to the GT arm.
    """
    dataset_config = build_dataset_config(
        source, pkl_path=pkl_path, fps_stride_tuples=fps_stride_tuples)
    base_config = dataset_config["base_dataset"]
    base_config.update({
        # Defaults preserve the original H20 evaluation protocol.  The
        # explicit overrides make a byte-identical sparse holdout replica
        # usable on another cluster without changing clip enumeration.
        "bbox_dir": os.environ.get(
            "PANDWM_BBOX_DIR", "/path/to/pandwm_root/data/bbox"),
        "hdmap_dir": os.environ.get(
            "PANDWM_HDMAP_DIR", "/path/to/pandwm_root/data/hdmap"),
        "enable_3dbox": True,
        "enable_hdmap": True,
        "enable_description": True,
    })
    stub = copy.deepcopy(base_config.get("stub_key_data_dict") or {})
    stub["crossview_mask"] = [
        "content", torch.tensor([[True, True], [True, True]])
    ]
    base_config["stub_key_data_dict"] = stub

    transforms = dataset_config.get("transform_list") or []
    dataset_config["transform_list"] = transforms

    def add_image_transform(old_key: str) -> None:
        if any(item.get("old_key") == old_key for item in transforms):
            return
        transforms.append({
            "old_key": old_key,
            "new_key": old_key,
            "transform": {
                "_class_name": "torchvision.transforms.Compose",
                "transforms": [
                    {
                        "_class_name": "torchvision.transforms.Resize",
                        "size": [256, 2304],
                    },
                    {"_class_name": "torchvision.transforms.ToTensor"},
                ],
            },
        })

    add_image_transform("3dbox_images")
    add_image_transform("hdmap_images")
    description_transform = next(
        (
            item for item in transforms
            if item.get("old_key") == "image_description" and
            item.get("new_key") == "clip_text"
        ),
        None,
    )
    if description_transform is None:
        transforms.append({
            "old_key": "image_description",
            "new_key": "clip_text",
            "transform": {"_class_name": "dwm.datasets.common.Copy"},
            "stack": False,
        })
    else:
        description_transform["stack"] = False

    pop_list = dataset_config.get("pop_list") or []
    dataset_config["pop_list"] = pop_list
    if "images" not in pop_list:
        pop_list.append("images")
    if "lidar_points" not in pop_list:
        pop_list.append("lidar_points")
    if "image_description" not in pop_list:
        pop_list.append("image_description")
    return dataset_config


def arm_suffix(latent_source: str) -> str:
    try:
        return ARM_SUFFIXES[latent_source]
    except KeyError as exc:
        raise ValueError(f"Unknown latent source: {latent_source}") from exc


def select_middle_frame_indices(
    frame_count: int, frames_per_step: int
) -> torch.Tensor:
    if frames_per_step > frame_count:
        raise ValueError(
            f"frames_per_step={frames_per_step} exceeds clip length "
            f"{frame_count}"
        )
    # Select the midpoint of each equal-width temporal bin. This avoids clip
    # endpoints while keeping the frame subset deterministic and evenly spaced.
    positions = (
        (torch.arange(frames_per_step, dtype=torch.float64) + 0.5)
        * frame_count
        / frames_per_step
    )
    return positions.floor().long()


def load_frame_valid_masks(
    pipeline: Any,
    batch: dict[str, Any],
    frame_indices: torch.Tensor,
) -> torch.Tensor:
    """Resize raw h5 depth>0 masks to the two decoder pano halves."""
    if h5py is None:
        raise ModuleNotFoundError(
            "h5py is required for holdout depth evaluation")
    range_dir = pipeline.training_config["range_dir"]
    half_height = pipeline.training_config["half_height"]
    half_width = pipeline.training_config["half_width"]
    batch_size = batch["vae_images"].shape[0]
    half_list = []
    for sample_index in range(batch_size):
        for frame_index in frame_indices.tolist():
            timestamp = int(
                batch["frame_timestamps_us"][sample_index, frame_index]
            )
            with h5py.File(
                os.path.join(range_dir, f"{timestamp}.h5"), "r"
            ) as handle:
                raw_valid = np.asarray(handle["depth"]) > 0.0
            valid = torch.from_numpy(raw_valid.astype(np.float32))[None, None]
            for column_range in (
                PANO_HALF_LEFT_RANGE,
                PANO_HALF_RIGHT_RANGE,
            ):
                half = valid[:, :, :, column_range[0]:column_range[1]]
                half = F.interpolate(
                    half,
                    size=(half_height, half_width),
                    mode="nearest",
                )
                half_list.append(half[0, 0] > 0.5)
    stacked = torch.stack(half_list).to(pipeline.device)
    return stacked.unflatten(0, (batch_size * len(frame_indices), 2))


class RGBMetricAccumulator:
    def __init__(
        self,
        lpips_module: torch.nn.Module,
        protocol_label: str = "6view-800x300",
    ):
        self.lpips_module = lpips_module
        self.protocol_label = protocol_label
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.lpips_sum = 0.0
        self.count = 0
        self.per_frame: dict[int, dict[str, float]] = {}

    def update(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        frame_index: int | None = None,
        valid: torch.Tensor | None = None,
    ) -> None:
        rendered = rendered.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        if valid is None:
            psnr = float(
                peak_signal_noise_ratio(
                    rendered, target, data_range=1.0).item()
            )
        else:
            valid = valid.to(device=rendered.device, dtype=torch.bool)
            if valid.ndim != 3 or valid.shape != rendered.shape[:1] \
                    + rendered.shape[-2:]:
                raise ValueError(
                    "RGB validity mask must be [V,H,W], got {} for {}"
                    .format(tuple(valid.shape), tuple(rendered.shape)))
            valid_channels = valid.unsqueeze(1).expand_as(rendered)
            mse = (rendered - target).square()[valid_channels].mean()
            psnr = float((10.0 * torch.log10(1.0 / mse)).item())
            # SSIM and LPIPS do not accept an external validity mask.  Make
            # the fixed stitch-invalid cells equal on both sides.  They are
            # calibration-dependent, never candidate-dependent.
            rendered = torch.where(
                valid.unsqueeze(1), rendered, target)
        ssim = float(
            structural_similarity_index_measure(
                rendered, target, data_range=1.0
            ).item()
        )
        self.lpips_module.reset()
        lpips = float(self.lpips_module(rendered, target).item())
        self.lpips_module.reset()
        self.psnr_sum += psnr
        self.ssim_sum += ssim
        self.lpips_sum += lpips
        self.count += 1
        if frame_index is not None:
            values = self.per_frame.setdefault(
                int(frame_index),
                {"psnr_sum": 0.0, "ssim_sum": 0.0, "lpips_sum": 0.0,
                 "count": 0},
            )
            values["psnr_sum"] += psnr
            values["ssim_sum"] += ssim
            values["lpips_sum"] += lpips
            values["count"] += 1

    def result(self, suffix: str) -> dict[str, float]:
        if self.count == 0:
            raise RuntimeError("No RGB frames were evaluated")
        return {
            f"PSNR-{self.protocol_label}-{suffix}":
                self.psnr_sum / self.count,
            f"SSIM-{self.protocol_label}-{suffix}":
                self.ssim_sum / self.count,
            f"LPIPS-{self.protocol_label}-{suffix}":
                self.lpips_sum / self.count,
        }

    def per_frame_result(self, suffix: str) -> dict[str, dict[str, float]]:
        return {
            str(frame_index): {
                f"PSNR-{self.protocol_label}-{suffix}": values["psnr_sum"] /
                values["count"],
                f"SSIM-{self.protocol_label}-{suffix}": values["ssim_sum"] /
                values["count"],
                f"LPIPS-{self.protocol_label}-{suffix}": values["lpips_sum"] /
                values["count"],
            }
            for frame_index, values in sorted(self.per_frame.items())
        }


class DepthMetricAccumulator:
    def __init__(self) -> None:
        self.absolute_error_sum = 0.0
        self.absolute_relative_error_sum = 0.0
        self.valid_count = 0
        self.per_frame: dict[int, dict[str, float]] = {}

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        frame_index: int | None = None,
    ) -> None:
        if prediction.shape != target.shape or valid.shape != target.shape:
            raise ValueError("Depth prediction, target, and mask shapes differ")
        error = (prediction - target).abs()
        absolute_error = float(error[valid].sum().item())
        absolute_relative_error = float(
            (error[valid] / target[valid]).sum().item()
        )
        valid_count = int(valid.sum().item())
        self.absolute_error_sum += absolute_error
        self.absolute_relative_error_sum += absolute_relative_error
        self.valid_count += valid_count
        if frame_index is not None:
            values = self.per_frame.setdefault(
                int(frame_index),
                {"absolute_error_sum": 0.0,
                 "absolute_relative_error_sum": 0.0, "valid_count": 0},
            )
            values["absolute_error_sum"] += absolute_error
            values["absolute_relative_error_sum"] += absolute_relative_error
            values["valid_count"] += valid_count

    def result(self, suffix: str) -> dict[str, float]:
        if self.valid_count == 0:
            raise RuntimeError("No valid holdout depth pixels were evaluated")
        return {
            f"DepthL1-pano-{suffix}":
                self.absolute_error_sum / self.valid_count,
            f"AbsRel-pano-{suffix}":
                self.absolute_relative_error_sum / self.valid_count,
        }

    def per_frame_result(self, suffix: str) -> dict[str, dict[str, float]]:
        return {
            str(frame_index): {
                f"DepthL1-pano-{suffix}":
                    values["absolute_error_sum"] / values["valid_count"],
                f"AbsRel-pano-{suffix}":
                    values["absolute_relative_error_sum"] /
                    values["valid_count"],
            }
            for frame_index, values in sorted(self.per_frame.items())
            if values["valid_count"] > 0
        }


def make_grid_pair(
    target: torch.Tensor, rendered: torch.Tensor
) -> torch.Tensor:
    view_width = GRID_VIEW_WIDTH * 6 // target.shape[0]
    target_small = F.interpolate(
        target,
        size=(GRID_VIEW_HEIGHT, view_width),
        mode="bilinear",
        antialias=True,
    ).clamp(0.0, 1.0)
    rendered_small = F.interpolate(
        rendered,
        size=(GRID_VIEW_HEIGHT, view_width),
        mode="bilinear",
        antialias=True,
    ).clamp(0.0, 1.0)
    target_row = torch.cat(list(target_small), dim=-1)
    rendered_row = torch.cat(list(rendered_small), dim=-1)
    return torch.cat([target_row, rendered_row], dim=-2).cpu()


def scalar_from_batch(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def string_from_batch(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def decode_latents_to_images(
    vae_pipeline: Any, latents: torch.Tensor
) -> torch.Tensor:
    """Decode (B,T,V,C,H,W) latents into (B,T,V,3,H_img,W_img)."""
    import dwm.functional

    shift_factor = vae_pipeline.vae.config.shift_factor
    if shift_factor is None:
        shift_factor = 0
    flat_latents = latents.flatten(0, 2).to(dtype=vae_pipeline.vae.dtype)
    decode_batch = (
        vae_pipeline.common_config.get("memory_efficient_batch", -1)
        if hasattr(vae_pipeline, "common_config")
        else 12
    )
    # Shared-GPU escape hatch: the whole-clip VAE decode is the peak
    # allocation of the re-encode arms; a smaller batch trades time for
    # headroom when a co-tenant process grows mid-run.
    env_batch = os.environ.get("PANDWM_G1_DECODE_BATCH")
    if env_batch:
        decode_batch = int(env_batch)
    image_tensor = dwm.functional.memory_efficient_split_call(
        vae_pipeline.vae,
        flat_latents,
        lambda block, tensor: block.decode(
            tensor / block.config.scaling_factor + shift_factor,
            return_dict=False,
        )[0],
        decode_batch,
    )
    image_processor = getattr(vae_pipeline, "image_processor", None)
    if image_processor is None:
        try:
            import diffusers
        except ModuleNotFoundError:
            # The GS pipeline has no image_processor of its own.  This is the
            # same [-1, 1] -> [0, 1] operation used by VaeImageProcessor and
            # keeps parser/unit smokes usable on the CPU-only laptop.
            images = (image_tensor / 2.0 + 0.5).clamp(0.0, 1.0)
        else:
            image_processor = diffusers.image_processor.VaeImageProcessor(
                vae_scale_factor=2 ** (
                    len(vae_pipeline.vae.config.block_out_channels) - 1))
            images = image_processor.postprocess(
                image_tensor, output_type="pt")
    else:
        images = image_processor.postprocess(image_tensor, output_type="pt")
    return images.unflatten(0, latents.shape[:3])


def encode_context_latents(
    stage2_pipeline: Any, batch: dict[str, Any], reference_frame_count: int
) -> torch.Tensor:
    """Encode CD reference frames with the AR path's VAE mode() operation."""
    import dwm.functional

    raw_images = batch["vae_images"][:, :reference_frame_count]
    image_tensor = stage2_pipeline.image_processor.preprocess(
        raw_images.flatten(0, 2).to(stage2_pipeline.device))
    shift_factor = stage2_pipeline.vae.config.shift_factor
    if shift_factor is None:
        shift_factor = 0
    image_latents = dwm.functional.memory_efficient_split_call(
        stage2_pipeline.vae,
        image_tensor,
        lambda block, tensor: (
            block.encode(tensor).latent_dist.mode() - shift_factor
        ) * block.config.scaling_factor,
        stage2_pipeline.common_config.get("memory_efficient_batch", -1),
    )
    return image_latents.unflatten(
        0, raw_images.shape[:3]).to(stage2_pipeline.device)


def make_stage2_latent_provider(
    gs_pipeline: Any,
    stage2_pipeline: Any,
    latent_source: str,
) -> Any:
    if latent_source not in ("stage2", "stage2_reencode", "gt_reencode"):
        raise ValueError(f"Unsupported generated latent source: {latent_source}")
    infer_fn = getattr(stage2_pipeline, "_g1_4step_inference", None)
    if infer_fn is None and stage2_pipeline is not None:
        candidate = getattr(stage2_pipeline, "inference_pipeline", None)
        # A callable installed directly on an instance (as the 4-step helper
        # does) is safe to reuse.  A bound class method may still be the
        # original 50-step path, so rebuild that one below.
        if candidate is not None and not hasattr(candidate, "__self__"):
            infer_fn = candidate
    if latent_source != "gt_reencode":
        if infer_fn is None:
            from dwm_pano.evaluate_4step import (
                compute_denoising_sigmas,
                make_4step_inference,
            )

            sigmas = compute_denoising_sigmas(stage2_pipeline)
            infer_fn = make_4step_inference(stage2_pipeline, sigmas)
    reference_frame_count = 3

    def roundtrip_frame_step() -> int:
        step = int(os.environ.get("PANDWM_G1_ROUNDTRIP_FRAMES", "2"))
        if step < 1:
            raise ValueError(
                "PANDWM_G1_ROUNDTRIP_FRAMES must be at least 1, got "
                f"{step}"
            )
        return step

    def encode_clip_images(
        encode_pipeline: Any, clip_images: torch.Tensor
    ) -> torch.Tensor:
        """Encode a clip without materializing a full-clip VAE activation."""
        step = roundtrip_frame_step()
        chunks = []
        for start in range(0, clip_images.shape[1], step):
            part = clip_images[:, start:start + step]
            chunks.append(encode_pipeline.encode_half_latents(
                part.flatten(0, 1).to(encode_pipeline.device)
            ).unflatten(0, part.shape[:2]))
        return torch.cat(chunks, dim=1)

    def roundtrip_latents(
        decode_pipeline: Any, clip_latents: torch.Tensor
    ) -> torch.Tensor:
        # Decode->postprocess->encode a few frames at a time: a whole 19-frame
        # clip through the VAE in one call is a >10 GiB activation spike, which
        # does not fit next to the co-resident Stage-2 stack on a shared GPU.
        step = roundtrip_frame_step()
        chunks = []
        for start in range(0, clip_latents.shape[1], step):
            part = clip_latents[:, start:start + step]
            decoded = decode_latents_to_images(decode_pipeline, part)
            chunks.append(gs_pipeline.encode_half_latents(
                decoded.flatten(0, 1).to(gs_pipeline.device)
            ).unflatten(0, decoded.shape[:2]))
        return torch.cat(chunks, dim=1)

    @torch.no_grad()
    def provider(batch: dict[str, Any]) -> torch.Tensor:
        if latent_source == "gt_reencode":
            vae_images = batch["vae_images"].to(gs_pipeline.device)
            # The original path encoded all 19 frames (38 panorama halves)
            # in one VAE call before applying the round-trip microbatch.  On
            # the shared H20 that left a >10 GiB activation spike outside the
            # memory guard and made this arm OOM even with
            # PANDWM_G1_ROUNDTRIP_FRAMES=1.  Apply the same fixed microbatch
            # to the initial GT encoding as to the decode/re-encode pass.
            gt_latents = encode_clip_images(gs_pipeline, vae_images)
            roundtrip = roundtrip_latents(gs_pipeline, gt_latents)
            if tuple(roundtrip.shape[:3]) != tuple(vae_images.shape[:3]):
                raise RuntimeError(
                    "GT re-encode latent provider returned an unexpected shape"
                )
            return roundtrip

        image_latents = encode_context_latents(
            stage2_pipeline, batch, reference_frame_count)
        latent_shape = tuple(batch["vae_images"].shape[:3]) + tuple(
            image_latents.shape[3:])
        if infer_fn is None:
            raise RuntimeError("Stage-2 inference callable was not built")
        result = infer_fn(
            latent_shape,
            batch,
            "latent",
            image_latents=image_latents,
            reference_frame_count=reference_frame_count,
        )
        clip_latents = torch.cat([
            image_latents,
            result["latents"][:, reference_frame_count:],
        ], dim=1)
        expected_prefix = tuple(batch["vae_images"].shape[:3])
        if tuple(clip_latents.shape[:3]) != expected_prefix:
            raise RuntimeError(
                "Stage-2 latent provider returned an unexpected clip shape: "
                f"expected prefix {expected_prefix}, got {tuple(clip_latents.shape[:3])}"
            )
        if latent_source == "stage2":
            return clip_latents
        roundtrip = roundtrip_latents(stage2_pipeline, clip_latents)
        if tuple(roundtrip.shape[:3]) != expected_prefix:
            raise RuntimeError(
                "Stage-2 re-encode latent provider returned an unexpected "
                "clip shape"
            )
        return roundtrip

    # Published so callers can label which frames are context instead of
    # keeping their own copy of this constant, which would silently disagree
    # the moment this one changes.
    provider.reference_frame_count = reference_frame_count
    return provider


def install_strict_pano_hardseam_stub() -> None:
    """Install the P0 hard-seam stub before any conditioned dataset import."""
    def unavailable_stitcher(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Online pano stitching is unavailable; verify every bbox and "
            "hdmap timestamp has a pre-rendered PNG"
        )

    omninwm = types.ModuleType("omninwm")
    omninwm.__path__ = []
    omninwm_utils = types.ModuleType("omninwm.utils")
    omninwm_utils.__path__ = []
    pano_hardseam = types.ModuleType("omninwm.utils.pano_hardseam")
    pano_hardseam.stitch_hardseam = unavailable_stitcher
    omninwm.utils = omninwm_utils
    omninwm_utils.pano_hardseam = pano_hardseam
    sys.modules["omninwm"] = omninwm
    sys.modules["omninwm.utils"] = omninwm_utils
    sys.modules["omninwm.utils.pano_hardseam"] = pano_hardseam

    module = sys.modules.get("dwm_pano.datasets.pano_via_stitch")
    if module is not None:
        module.stitch_hardseam = unavailable_stitcher

    import importlib

    dataset_module = importlib.import_module(
        "dwm_pano.datasets.nuscenes_pano")
    if not getattr(dataset_module, "_g1_strict_png_guard", False):
        original_bbox = dataset_module._render_bboxes_on_full_pano
        original_hdmap = dataset_module._render_hdmap_on_full_pano

        def strict_bbox(info: dict, pen_width: int = 3,
                        bbox_dir: str | None = None) -> Any:
            if not bbox_dir:
                raise RuntimeError(
                    "G1 requires a pre-rendered bbox_dir with timestamp PNGs")
            path = os.path.join(bbox_dir, f"{int(info['timestamp'])}.png")
            if not os.path.isfile(path):
                raise RuntimeError(f"Missing bbox PNG: {path}")
            return original_bbox(info, pen_width=pen_width, bbox_dir=bbox_dir)

        def strict_hdmap(info: dict, pen_width: int = 2,
                         hdmap_dir: str | None = None) -> Any:
            if not hdmap_dir:
                raise RuntimeError(
                    "G1 requires a pre-rendered hdmap_dir with timestamp PNGs")
            path = os.path.join(hdmap_dir, f"{int(info['timestamp'])}.png")
            if not os.path.isfile(path):
                raise RuntimeError(f"Missing hdmap PNG: {path}")
            return original_hdmap(info, pen_width=pen_width, hdmap_dir=hdmap_dir)

        dataset_module._render_bboxes_on_full_pano = strict_bbox
        dataset_module._render_hdmap_on_full_pano = strict_hdmap
        dataset_module._g1_strict_png_guard = True


def ensure_single_rank_process_group(device: torch.device) -> bool:
    if torch.distributed.is_initialized():
        if torch.distributed.get_world_size() != 1:
            raise RuntimeError(
                "Stage-2 G1 evaluation requires a one-rank process group"
            )
        return False
    if device.type != "cuda":
        raise RuntimeError("Stage-2 G1 evaluation requires a CUDA device")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError(
            "Stage-2 G1 evaluation must be launched with WORLD_SIZE=1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29505")
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend="nccl")
    return True


def build_stage2_pipeline(
    stage2_config_path: Path,
    stage2_run: Path,
    stage2_step: int,
    stage2_ema: bool,
    device: torch.device,
) -> Any:
    install_strict_pano_hardseam_stub()
    from dwm_pano.pipelines import (
        ctsd_pano_consistency_distillation as cd_module,
    )
    import dwm_pano.evaluate_4step as evaluate_4step

    config = load_config(stage2_config_path)
    global_state_config = config.setdefault("global_state", {})
    device_mesh_config = global_state_config.get("device_mesh")
    if device_mesh_config is not None:
        device_mesh_config["mesh_shape"] = [1, 1]
    config["pipeline"]["metrics"] = {}

    for key, value in global_state_config.items():
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(
            value)

    pipeline_class = (
        cd_module.PanoCrossviewTemporalConsistencyDistillation)
    original_build_auxiliary_model = pipeline_class._build_auxiliary_model

    def build_auxiliary_model(
        self: Any,
        checkpoint_path: str,
        model_load_state_args: dict,
        requires_grad: bool,
        role: str,
    ) -> Any:
        if role in ("teacher", "ema") and hasattr(self, "model_wrapper"):
            print(
                f"G1 eval shim: aliasing consistency {role} to the main model",
                flush=True,
            )
            return self.model, self.model_wrapper
        return original_build_auxiliary_model(
            self, checkpoint_path, model_load_state_args, requires_grad, role)

    pipeline_class._build_auxiliary_model = build_auxiliary_model

    pipeline_config = config["pipeline"]
    try:
        if stage2_ema:
            checkpoint_path = stage2_run / "checkpoints" / \
                f"{stage2_step}_ema.pth"
            pipeline_config["model_checkpoint_path"] = str(checkpoint_path)
            pipeline_config["consistency_ema_checkpoint_path"] = str(
                checkpoint_path)
            pipeline = dwm.common.create_instance_from_config(
                pipeline_config,
                output_path=str(stage2_run),
                config=config,
                device=device,
            )
        else:
            pipeline = dwm.common.create_instance_from_config(
                pipeline_config,
                output_path=str(stage2_run),
                config=config,
                device=device,
                resume_from=stage2_step,
            )
    finally:
        # The shim is needed only while the constructor requests teacher/EMA
        # copies.  Restore the class method so importing this module does not
        # alter later pipeline construction in the same interpreter.
        pipeline_class._build_auxiliary_model = original_build_auxiliary_model

    # The 4-step callable decodes the whole clip with this batch size, and so
    # do the re-encode arms; one override covers every decode on this pipeline.
    env_batch = os.environ.get("PANDWM_G1_DECODE_BATCH")
    if env_batch:
        pipeline.common_config["memory_efficient_batch"] = int(env_batch)
    sigmas = evaluate_4step.compute_denoising_sigmas(pipeline)
    infer_fn = evaluate_4step.make_4step_inference(pipeline, sigmas)
    pipeline.inference_pipeline = infer_fn
    pipeline._g1_4step_inference = infer_fn
    pipeline.model_wrapper.eval()
    pipeline.vae.eval()
    return pipeline


@torch.no_grad()
def evaluate_dataset(
    pipeline: Any,
    dataset: torch.utils.data.Dataset,
    split_name: str,
    include_depth: bool,
    manifest: list[dict[str, Any]],
    latent_provider: Any = None,
    collate_fn: Any = None,
    rng_seed: int | None = None,
) -> tuple[
    RGBMetricAccumulator,
    DepthMetricAccumulator | None,
    DepthMetricAccumulator | None,
    list[torch.Tensor],
]:
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 0,
    }
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    # ``PanoGaussianDecoderPipeline.encode_half_latents`` uses the global
    # torch RNG for posterior.sample().  Reset once at the start of each split
    # so a re-encode arm is reproducible regardless of which split ran first.
    # The GT arm passes None and therefore keeps its original RNG stream.
    if rng_seed is not None:
        torch.manual_seed(rng_seed)
    rgb_protocol_label = pipeline.supervision_output_label()
    rgb_metrics = RGBMetricAccumulator(
        pipeline.perceptual_metric, protocol_label=rgb_protocol_label)
    depth_metrics = DepthMetricAccumulator() if include_depth else None
    trivial_metrics = DepthMetricAccumulator() if include_depth else None
    grid_pairs = []
    frames_per_step = pipeline.training_config["frames_per_step"]
    hole_fill = float(pipeline.training_config["range_hole_fill_m"])
    use_temporal_context = pipeline.training_config.get(
        "evaluation_use_temporal_context", True)

    for clip_index, batch in enumerate(loader):
        if batch["vae_images"].shape[0] != 1:
            raise RuntimeError("G1 evaluation requires batch_size=1")
        pipeline._prepare_nuscenes6_supervision(batch)
        frame_count = int(batch["vae_images"].shape[1])
        frame_indices = select_middle_frame_indices(
            frame_count, frames_per_step
        )
        half_images = batch["vae_images"][:, frame_indices].flatten(0, 1)
        half_images = half_images.to(pipeline.device)
        latent_origins, pixel_origins = pipeline._frame_ray_origins(
            batch, frame_indices)

        gt_ranges = None
        valid_masks = None
        if include_depth:
            gt_ranges, valid_masks = pipeline._load_frame_ranges(
                batch, frame_indices, return_valid=True)
            valid_masks = valid_masks \
                & pipeline.pixel_content_ray_valid.unsqueeze(0)

        selected_timestamps = [
            int(batch["frame_timestamps_us"][0, index])
            for index in frame_indices.tolist()
        ]
        manifest.append(
            {
                "split": split_name,
                "dataset_index": scalar_from_batch(batch["dataset_index"]),
                "scene_token": string_from_batch(batch["scene_token"]),
                "anchor_timestamp_us": scalar_from_batch(
                    batch["anchor_timestamp_us"]
                ),
                "frame_timestamps_us": selected_timestamps,
                "in_grid": clip_index < 2,
            }
        )
        if "clip_text" in batch:
            manifest[-1]["captions"] = json_safe(batch["clip_text"][0])

        clip_latents = None
        if latent_provider is not None:
            clip_latents = latent_provider(batch)

        visual_frame_index = len(frame_indices) // 2
        previous_latents = None
        for sample_index in range(half_images.shape[0]):
            frame_images = half_images[sample_index:sample_index + 1]
            frame_index_in_clip = int(frame_indices[sample_index])
            if latent_provider is None:
                latents = pipeline.encode_half_latents(frame_images)
            else:
                if clip_latents is None:
                    raise RuntimeError(
                        "Latent provider returned no clip latents")
                latents = clip_latents[:, frame_index_in_clip]
            gaussians, predicted_range, _ = pipeline.decoder_wrapper(
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
                    latent_origins[sample_index:sample_index + 1],
                pixel_origins=None if pixel_origins is None else
                    pixel_origins[sample_index:sample_index + 1],
            )
            previous_latents = (
                latents.detach() if use_temporal_context else None)
            rendered_rgb, _, _, render_valid = \
                pipeline._render_own_supervision(gaussians[0])
            target_images = pipeline._target_rgb_for_supervision(frame_images)
            rgb_metrics.update(
                rendered_rgb, target_images,
                frame_index=frame_index_in_clip,
                valid=render_valid)

            if include_depth:
                assert gt_ranges is not None
                assert valid_masks is not None
                assert depth_metrics is not None
                assert trivial_metrics is not None
                frame_gt = gt_ranges[sample_index:sample_index + 1]
                frame_valid = valid_masks[
                    sample_index:sample_index + 1
                ]
                depth_metrics.update(
                    predicted_range, frame_gt, frame_valid,
                    frame_index=frame_index_in_clip,
                )
                trivial_metrics.update(
                    torch.full_like(frame_gt, hole_fill),
                    frame_gt,
                    frame_valid,
                    frame_index=frame_index_in_clip,
                )

            if clip_index < 2 and sample_index == visual_frame_index:
                display_target = pipeline.mask_supervision_for_display(
                    target_images)
                display_render = pipeline.mask_supervision_for_display(
                    rendered_rgb)
                grid_pairs.append(
                    make_grid_pair(display_target, display_render)
                )

        if (clip_index + 1) % 10 == 0 or clip_index + 1 == len(dataset):
            print(
                f"{split_name}: evaluated {clip_index + 1}/{len(dataset)} clips",
                flush=True,
            )

    return rgb_metrics, depth_metrics, trivial_metrics, grid_pairs


def validate_protocol(config: dict[str, Any], pipeline: Any) -> None:
    training_config = pipeline.training_config
    decoder_config = config["pipeline"]["decoder"]
    refiner_config = config["pipeline"].get("latent_refinement")
    common_required = {
        "range_dir is set": bool(training_config.get("range_dir")),
        "range_hole_fill_m is 100.0":
            float(training_config["range_hole_fill_m"]) == 100.0,
        "use_range_prior is false":
            decoder_config.get("use_range_prior") is False,
        "latent refinement is disabled":
            refiner_config is None or refiner_config.get("enabled") is False,
        "LPIPS module is available": pipeline.perceptual_metric is not None,
    }
    if pipeline.uses_nuscenes6_stitch:
        geometry_required = {
            "real-six-camera stitch supervision":
                training_config.get("supervision_geometry")
                == "nuscenes6_onebev_stitch",
            "public OneBEV-compatible projection":
                training_config.get("stitch_projection_model")
                == "onebev_legacy",
            "source render height is 360":
                training_config.get("source_camera_render_height") == 360,
            "source render width is 640":
                training_config.get("source_camera_render_width") == 640,
            "camera origins enabled":
                decoder_config.get("use_camera_origins") is True,
            "Plucker rays enabled":
                decoder_config.get("ray_embedding_mode") == "plucker",
            "independent-frame G2 evaluation":
                training_config.get("evaluation_use_temporal_context")
                is False,
        }
    else:
        geometry_required = {
            "six target views":
                len(training_config["target_yaws_deg"]) == 6,
            "target height is 300":
                training_config["target_height"] == 300,
            "target width is 800":
                training_config["target_width"] == 800,
        }
    required = {**common_required, **geometry_required}
    failed = [name for name, valid in required.items() if not valid]
    if failed:
        raise RuntimeError(
            "P2 GS evaluation protocol guard failed: " + "; ".join(failed)
        )


def save_grid(
    path: Path, pairs: list[torch.Tensor], expected_pairs: int = 4
) -> None:
    if len(pairs) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} grid pairs, got {len(pairs)}"
        )
    grid = torch.cat(pairs, dim=-2).clamp(0.0, 1.0)
    array = (
        grid.permute(1, 2, 0).mul(255.0).round().byte().numpy()
    )
    Image.fromarray(array).save(path)


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def dataset_frame_manifest(dataset: torch.utils.data.Dataset) -> list[list[int]]:
    """Return every clip's frame timestamps using the dataset enumeration."""
    seen = set()
    while hasattr(dataset, "base_dataset") or hasattr(dataset, "dataset"):
        object_id = id(dataset)
        if object_id in seen:
            raise RuntimeError("G1 manifest guard found a dataset wrapper cycle")
        seen.add(object_id)
        if hasattr(dataset, "base_dataset"):
            dataset = dataset.base_dataset
        else:
            dataset = dataset.dataset
    if not hasattr(dataset, "clips"):
        raise RuntimeError("G1 manifest guard requires a PanoNuScenesDataset")
    return [
        [int(info["timestamp"]) for info in clip]
        for clip in dataset.clips
    ]


def assert_dataset_frame_manifest(
    expected: torch.utils.data.Dataset,
    actual: torch.utils.data.Dataset,
    label: str,
) -> None:
    expected_manifest = dataset_frame_manifest(expected)
    actual_manifest = dataset_frame_manifest(actual)
    if expected_manifest != actual_manifest:
        first_mismatch = next(
            (
                index for index, (expected_clip, actual_clip) in enumerate(
                    zip(expected_manifest, actual_manifest))
                if expected_clip != actual_clip
            ),
            min(len(expected_manifest), len(actual_manifest)),
        )
        expected_clip = (
            expected_manifest[first_mismatch]
            if first_mismatch < len(expected_manifest) else "missing")
        actual_clip = (
            actual_manifest[first_mismatch]
            if first_mismatch < len(actual_manifest) else "missing")
        raise RuntimeError(
            f"G1 frame_timestamps manifest mismatch for {label}; "
            "latent arms must enumerate identical clips and frames; "
            f"first mismatch at clip {first_mismatch} "
            f"(expected={expected_clip}, actual={actual_clip})"
        )


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"Config is missing: {args.config}")
    if not args.holdout_pkl.is_file():
        raise FileNotFoundError(f"Holdout pkl is missing: {args.holdout_pkl}")
    checkpoint_path = args.run_dir / "checkpoints" / f"{args.step}.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint_path}")

    stage2_checkpoint_path = None
    if args.latent_source in ("stage2", "stage2_reencode"):
        assert args.stage2_config is not None
        assert args.stage2_run is not None
        assert args.stage2_step is not None
        if not args.stage2_config.is_file():
            raise FileNotFoundError(
                f"Stage-2 config is missing: {args.stage2_config}")
        suffix = "_ema" if args.stage2_ema else ""
        stage2_checkpoint_path = args.stage2_run / "checkpoints" / \
            f"{args.stage2_step}{suffix}.pth"
        if not stage2_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Stage-2 checkpoint is missing: {stage2_checkpoint_path}")

    config = load_config(args.config)
    torch.manual_seed(config["generator_seed"])
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    process_group_started = False
    stage2_pipeline = None
    latent_provider = None
    collate_fn = None
    if args.latent_source in ("stage2", "stage2_reencode"):
        process_group_started = ensure_single_rank_process_group(device)
        device = torch.device("cuda", 0)
        stage2_pipeline = build_stage2_pipeline(
            args.stage2_config,
            args.stage2_run,
            args.stage2_step,
            args.stage2_ema,
            device,
        )

    # Match train.py construction, but do not pass resume_from: eval loads only
    # the raw decoder state dict and never loads the optimizer checkpoint.
    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"],
        output_path=str(args.run_dir),
        config=config,
        device=device,
        resume_from=None,
    )
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
    print(f"Loaded decoder checkpoint: {checkpoint_path}", flush=True)

    if args.latent_source in ("stage2", "stage2_reencode"):
        latent_provider = make_stage2_latent_provider(
            pipeline, stage2_pipeline, args.latent_source)
        from dwm.datasets.common import CollateFnIgnoring

        collate_fn = CollateFnIgnoring(keys=["clip_text"])
    elif args.latent_source == "gt_reencode":
        latent_provider = make_stage2_latent_provider(
            pipeline, None, args.latent_source)

    holdout_config = build_dataset_config(
        config["training_dataset"],
        pkl_path=args.holdout_pkl,
        fps_stride_tuples=[[10, 20]],
    )
    validation_config = build_dataset_config(config["validation_dataset"])
    if args.latent_source in ("stage2", "stage2_reencode"):
        install_strict_pano_hardseam_stub()
        holdout_eval_config = build_g1_dataset_config(
            config["training_dataset"],
            pkl_path=args.holdout_pkl,
            fps_stride_tuples=[[10, 20]],
        )
        validation_eval_config = build_g1_dataset_config(
            config["validation_dataset"])
        holdout_dataset = dwm.common.create_instance_from_config(
            holdout_eval_config)
        validation_dataset = dwm.common.create_instance_from_config(
            validation_eval_config)
        expected_holdout = dwm.common.create_instance_from_config(
            holdout_config)
        expected_validation = dwm.common.create_instance_from_config(
            validation_config)
        # These untouched GS-config datasets are the GT arm's canonical clip
        # enumeration.  The conditioned copies must match frame-for-frame.
        assert_dataset_frame_manifest(
            expected_holdout, holdout_dataset, "holdout5")
        assert_dataset_frame_manifest(
            expected_validation, validation_dataset, "val32")
    else:
        # This construction is the canonical GT clip enumeration.  The
        # generated arms above compare against a fresh copy of these configs;
        # GT itself is the reference and must keep its original path.
        holdout_dataset = dwm.common.create_instance_from_config(holdout_config)
        validation_dataset = dwm.common.create_instance_from_config(
            validation_config
        )
        if args.latent_source == "gt_reencode":
            expected_holdout = dwm.common.create_instance_from_config(
                holdout_config)
            expected_validation = dwm.common.create_instance_from_config(
                validation_config)
            assert_dataset_frame_manifest(
                expected_holdout, holdout_dataset, "holdout5")
            assert_dataset_frame_manifest(
                expected_validation, validation_dataset, "val32")
    validation_count = min(args.val_limit, len(validation_dataset))
    minimum_clips = 1 if args.smoke else 2
    if len(holdout_dataset) < minimum_clips:
        raise RuntimeError(
            f"Holdout dataset needs at least {minimum_clips} clips")
    if validation_count < minimum_clips:
        raise RuntimeError(
            f"Validation dataset needs at least {minimum_clips} clips")
    holdout_eval_dataset: torch.utils.data.Dataset = holdout_dataset
    if args.smoke:
        holdout_eval_dataset = torch.utils.data.Subset(
            holdout_dataset, range(1))
    validation_subset = torch.utils.data.Subset(
        validation_dataset, range(validation_count)
    )
    print(
        f"Evaluation clips: holdout={len(holdout_eval_dataset)}, "
        f"val={validation_count}",
        flush=True,
    )

    manifest: list[dict[str, Any]] = []
    holdout_rgb, holdout_depth, trivial_depth, holdout_pairs = \
        evaluate_dataset(
            pipeline,
            holdout_eval_dataset,
            HOLDOUT_SUFFIX,
            True,
            manifest,
            latent_provider=latent_provider,
            collate_fn=collate_fn,
            rng_seed=(config["generator_seed"]
                      if latent_provider is not None else None),
        )
    validation_rgb, _, _, validation_pairs = evaluate_dataset(
        pipeline,
        validation_subset,
        VALIDATION_SUFFIX,
        False,
        manifest,
        latent_provider=latent_provider,
        collate_fn=collate_fn,
        rng_seed=(config["generator_seed"]
                  if latent_provider is not None else None),
    )
    assert holdout_depth is not None
    assert trivial_depth is not None

    arm = arm_suffix(args.latent_source)
    holdout_metric_suffix = f"{HOLDOUT_SUFFIX}-{arm}"
    validation_metric_suffix = f"{VALIDATION_SUFFIX}-{arm}"
    holdout_result = holdout_rgb.result(holdout_metric_suffix)
    holdout_result.update(holdout_depth.result(holdout_metric_suffix))
    validation_result = validation_rgb.result(validation_metric_suffix)
    trivial_result = trivial_depth.result(holdout_metric_suffix)
    per_frame = {
        HOLDOUT_SUFFIX: holdout_rgb.per_frame_result(holdout_metric_suffix),
        VALIDATION_SUFFIX: validation_rgb.per_frame_result(
            validation_metric_suffix),
        "trivial_baseline": trivial_depth.per_frame_result(
            holdout_metric_suffix),
    }
    for frame_key, depth_values in holdout_depth.per_frame_result(
            holdout_metric_suffix).items():
        per_frame[HOLDOUT_SUFFIX].setdefault(frame_key, {}).update(depth_values)
    protocol = {
        "name": (
            "G2 real6-OneBEV-stitch GS holdout5/val32"
            if pipeline.uses_nuscenes6_stitch
            else "G1 GS holdout5/val32"),
        "supervision_geometry": pipeline.supervision_geometry,
        "temporal_context": config["pipeline"]["training_config"].get(
            "evaluation_use_temporal_context", True),
        "clip_length": 19,
        "reference_frame_count": 3,
        "selected_frame_indices": [2, 7, 11, 16],
        "p0_stage2_clip_length": 35,
    }
    if args.latent_source in ("stage2", "stage2_reencode"):
        protocol["stage2_reference_encoding"] = (
            "CD VAE posterior.mode(); frame 2 may differ from GS "
            "posterior.sample()")
    result = {
        "step": args.step,
        "latent_source": args.latent_source,
        "protocol": protocol,
        "holdout5": holdout_result,
        "val32": validation_result,
        "trivial_baseline": trivial_result,
        "per_frame": per_frame,
        "manifest": manifest,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "metrics.json"
    grid_path = args.out / "grid.png"
    atomic_json_dump(result, json_path)
    save_grid(
        grid_path,
        holdout_pairs + validation_pairs,
        expected_pairs=2 if args.smoke else 4,
    )
    print(f"Wrote: {json_path}", flush=True)
    print(f"Wrote: {grid_path}", flush=True)
    if process_group_started:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
