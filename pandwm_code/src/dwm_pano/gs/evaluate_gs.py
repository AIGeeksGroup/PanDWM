"""Shared helpers for the panoramic Gaussian decoder scripts.

Config and dataset-config loading (load_config, build_dataset_config, build_g1_dataset_config), the Stage-2 video
model stack (build_stage2_pipeline, make_stage2_latent_provider, encode_context_latents, decode_latents_to_images),
the strict pano hard-seam stub and the single-rank process group, imported by the export, evaluation and demo scripts.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch

import dwm.common


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
    """Add the conditioning fields used by the generative model to a GS dataset config.

    The GS training and validation configs define the clip enumeration; only fields consumed by the condition builder
    are added, so RGB, timestamps and frame selection stay identical to the GT arm.
    """
    dataset_config = build_dataset_config(
        source, pkl_path=pkl_path, fps_stride_tuples=fps_stride_tuples)
    base_config = dataset_config["base_dataset"]
    base_config.update({
        # directories can be overridden through the environment
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
    # the whole-clip VAE decode is the peak allocation; a smaller batch trades time for memory
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
            # no diffusers: same [-1, 1] to [0, 1] mapping as VaeImageProcessor
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
        # a callable set on the instance (as the 4-step helper does) is reused; a bound class method
        # may still be the 50-step path, so that one is rebuilt below
        if candidate is not None and not hasattr(candidate, "__self__"):
            infer_fn = candidate
    if latent_source != "gt_reencode":
        if infer_fn is None:
            from dwm_pano.eval.evaluate_4step import (
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
        # decode, postprocess and encode a few frames at a time to bound VAE activation memory
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
            # the initial GT encoding uses the same microbatch as the decode / re-encode pass
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

    # exposed so callers can label which frames are context
    provider.reference_frame_count = reference_frame_count
    return provider


def install_strict_pano_hardseam_stub() -> None:
    """Install a hard-seam stub that disables online panorama stitching, before any conditioned dataset import."""
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
    import dwm_pano.eval.evaluate_4step as evaluate_4step

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
        # the shim is only needed while the constructor requests teacher/EMA copies; restore the class method
        pipeline_class._build_auxiliary_model = original_build_auxiliary_model

    # one override covers every decode on this pipeline
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
