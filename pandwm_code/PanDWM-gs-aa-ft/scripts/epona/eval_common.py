"""Evaluate the official Epona checkpoint on PanDWM's frozen nuScenes clips.

This is deliberately a data/output adapter, not a reimplementation of Epona.
The model, VAE, checkpoint loader, sampler, and rollout step are imported from
the pinned upstream Epona checkout.  The adapter only:

* enumerates the same 149 nuScenes clips used by the PanDWM/HorizonDrive run;
* reconstructs their CAM_FRONT band from PanDWM's immutable panorama cache;
* feeds the official 10-frame Epona context at its native 512x1024 size; and
* writes lossless uint8 tensors understood by the shared PanDWM evaluator.

The target interval for the comparison is frames 10..25 of each 35-frame
clip (16 future frames).  PanDWM and HorizonDrive must be re-scored on exactly
the same interval.  Epona has ten context frames whereas those methods use
three; this conditioning difference must be disclosed with every result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import runpy
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision.transforms import functional as tv_functional

from dwm_pano.datasets.nuscenes_pano import (
    _enumerate_pano_clips,
    _group_infos_by_scene,
    _slice_full_pano_to_halves,
)
from dwm_pano.nuscenes6_bridge import (
    PANO_CANVAS_OVERLAP_COLUMNS,
    apply_camera_sampling_grid,
    build_camera_sampling_grid,
    canvas_source_axes,
)

# These imports must resolve to the pinned official Epona checkout.
from models.model import TrainTransformersDiT
from models.modules.tokenizer import VAETokenizer
from utils.preprocess import get_rel_pose


OFFICIAL_IMAGE_SIZE = (512, 1024)
OFFICIAL_CONTEXT_FRAMES = 10
OFFICIAL_SAMPLING_STEPS = 100
OFFICIAL_TEMPORAL_DECODE_PATCH = 6
COMMON_SEQUENCE_LENGTH = 35
COMMON_FPS = 10.0
COMMON_STRIDE_SECONDS = 20.0
COMMON_CLIP_COUNT = 149
PANO_HALF_SIZE = (2304, 256)  # width, height
NATIVE_CAMERA_SIZE = (1600, 900)  # width, height


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--pano-rgb", required=True)
    parser.add_argument("--canonical-real-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=COMMON_CLIP_COUNT)
    parser.add_argument("--prediction-frames", type=int, default=16)
    parser.add_argument("--dump-height", type=int, default=384)
    parser.add_argument("--dump-width", type=int, default=768)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--vae-sha256", required=True)
    return parser.parse_args()


def load_official_config(path: str, vae_path: str):
    # The released Config helper misses one imported symbol.  Loading this
    # plain assignment-only config via runpy avoids patching upstream code.
    values = {
        key: value for key, value in runpy.run_path(path).items()
        if not key.startswith("__")
    }
    config = SimpleNamespace(**values)
    config.batch_size = 1
    config.vae_ckpt = vae_path
    if tuple(config.image_size) != OFFICIAL_IMAGE_SIZE:
        raise ValueError(
            "official Epona nuScenes image_size changed: {}".format(
                config.image_size))
    if int(config.condition_frames) != OFFICIAL_CONTEXT_FRAMES:
        raise ValueError(
            "official Epona context changed: {}".format(
                config.condition_frames))
    if int(config.num_sampling_steps) != OFFICIAL_SAMPLING_STEPS:
        raise ValueError(
            "official Epona sampling steps changed: {}".format(
                config.num_sampling_steps))
    # The released GitHub config retains the image-DCAE defaults, while the
    # official Hugging Face model card identifies dcae_td_20000.pkl as the
    # temporal-aware decoder with temporal patch size 6.  The checkpoint has
    # 90 temporal-layer keys and cannot load strictly with the image default.
    config.add_encoder_temporal = False
    config.add_decoder_temporal = True
    config.temporal_patch_size = OFFICIAL_TEMPORAL_DECODE_PATCH
    return config


def load_common_clips(annotation_path: str):
    with open(annotation_path, "rb") as handle:
        payload = pickle.load(handle)
    # Match HorizonDrive's official annotation loader before applying the
    # shared PanDWM timestamp sampler; this fixes both scene and clip order.
    infos = list(sorted(payload["infos"], key=lambda item: item["timestamp"]))
    scene_groups = payload["scene_tokens"]
    token_to_scene = {}
    for scene_index, tokens in enumerate(scene_groups):
        for token in tokens:
            if token in token_to_scene:
                raise ValueError("duplicate token in scene groups: {}".format(token))
            token_to_scene[token] = "scene_group_{:04d}".format(scene_index)
    grouped = _group_infos_by_scene(
        infos, token2scene=token_to_scene, strict=True)
    clips = [
        clip for clip, _audit in _enumerate_pano_clips(
            grouped,
            COMMON_SEQUENCE_LENGTH,
            [(COMMON_FPS, COMMON_STRIDE_SECONDS)],
        )
    ]
    if len(clips) != COMMON_CLIP_COUNT:
        raise ValueError(
            "expected {} common clips, found {}".format(
                COMMON_CLIP_COUNT, len(clips)))
    return clips


def quaternion_wxyz_to_matrix(rotation):
    w, x, y, z = [float(value) for value in rotation]
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
         2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
         2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x),
         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def homogeneous_transform(rotation, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_matrix(rotation)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


class CommonBandLoader:
    def __init__(self, pano_rgb_dir: str):
        self.pano_rgb_dir = pano_rgb_dir
        self.source_axes = canvas_source_axes(
            half_size=(PANO_HALF_SIZE[1], PANO_HALF_SIZE[0]),
            right_trim_px=PANO_CANVAS_OVERLAP_COLUMNS,
        )
        self.grid_cache = {}

    def _canvas(self, info):
        path = os.path.join(
            self.pano_rgb_dir, "{}.jpg".format(int(info["timestamp"])))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        image = Image.open(path).convert("RGB")
        left, right = _slice_full_pano_to_halves(
            image, PANO_HALF_SIZE[0], PANO_HALF_SIZE[1])
        return torch.cat([
            tv_functional.to_tensor(left),
            tv_functional.to_tensor(right)[..., PANO_CANVAS_OVERLAP_COLUMNS:],
        ], dim=-1)

    @staticmethod
    def _camera_to_ego(camera):
        return homogeneous_transform(
            camera["sensor2ego_rotation"],
            camera["sensor2ego_translation"],
        )

    @staticmethod
    def ego_to_global(info):
        camera = info["cams"]["CAM_FRONT"]
        return homogeneous_transform(
            camera["ego2global_rotation"],
            camera["ego2global_translation"],
        )

    def _grid(self, info, source_size, output_size):
        camera = info["cams"]["CAM_FRONT"]
        camera_to_ego = self._camera_to_ego(camera)
        intrinsics = np.asarray(
            camera["camera_intrinsics"], dtype=np.float64)[:3, :3]
        key = (
            tuple(np.round(camera_to_ego, 7).ravel()),
            tuple(np.round(intrinsics, 5).ravel()),
            tuple(source_size),
            tuple(output_size),
        )
        if key not in self.grid_cache:
            self.grid_cache[key] = build_camera_sampling_grid(
                camera_to_ego,
                intrinsics,
                NATIVE_CAMERA_SIZE,
                source_size=source_size,
                output_size=output_size,
                band_limit=True,
                camera_index=3,
                source_axes=self.source_axes,
            )[:2]
        return self.grid_cache[key]

    def load(self, info, output_size):
        canvas = self._canvas(info)
        grid, valid = self._grid(
            info, tuple(canvas.shape[-2:]), tuple(output_size))
        return apply_camera_sampling_grid(canvas[None], grid, valid)[0]


def canonical_real_files(directory: str):
    result = {}
    for path in Path(directory).glob("*.pt"):
        prefix, separator, _token = path.stem.partition("_")
        if not separator or not prefix.isdigit():
            raise ValueError("unexpected canonical real filename: {}".format(path))
        index = int(prefix)
        if index in result:
            raise ValueError("duplicate canonical real index: {}".format(index))
        result[index] = path
    if len(result) != COMMON_CLIP_COUNT:
        raise ValueError(
            "expected {} canonical real files, found {}".format(
                COMMON_CLIP_COUNT, len(result)))
    return result


def validate_existing(path: Path, expected_frames: int, height: int, width: int):
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    expected_shape = (expected_frames, 3, height, width)
    if tensor.dtype != torch.uint8 or tuple(tensor.shape) != expected_shape:
        raise ValueError(
            "invalid existing output {}: shape={} dtype={}".format(
                path, tuple(tensor.shape), tensor.dtype))


def atomic_torch_save(tensor: torch.Tensor, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def atomic_json_dump(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def temporal_rollout_frames(scored_frames: int):
    if scored_frames <= 0:
        raise ValueError("prediction-frames must be positive")
    return int(math.ceil(
        scored_frames / OFFICIAL_TEMPORAL_DECODE_PATCH
    ) * OFFICIAL_TEMPORAL_DECODE_PATCH)


@torch.inference_mode()
def generate_clip(model, tokenizer, loader, clip, args, config):
    context_images = []
    absolute_poses = []
    input_start = time.monotonic()
    for frame_index, info in enumerate(clip):
        if frame_index < OFFICIAL_CONTEXT_FRAMES:
            context_images.append(loader.load(info, OFFICIAL_IMAGE_SIZE))
        absolute_poses.append(loader.ego_to_global(info))
    input_seconds = time.monotonic() - input_start

    context = torch.stack(context_images).mul_(2).sub_(1).unsqueeze(0).cuda()
    poses = torch.from_numpy(np.stack(absolute_poses)).float().unsqueeze(0).cuda()
    relative_pose, relative_yaw = get_rel_pose(poses)
    start_latents = tokenizer.encode_to_z(context)
    del context

    # The released temporal DCAE is defined for six-frame patches.  When the
    # metric wants 16 frames, roll the causal world model out to 18, decode
    # 6+6+6 exactly as the official video path does, then retain the first 16.
    # The DCAE uses a lower-triangular temporal mask, so the two padding-tail
    # frames cannot leak into the 16 retained frames.
    rollout_frames = temporal_rollout_frames(args.prediction_frames)
    predicted_latents = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    inference_start = time.monotonic()
    for offset in range(rollout_frames):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _predicted_trajectory, predicted = model.step_eval(
                start_latents,
                relative_pose[:, offset:offset + OFFICIAL_CONTEXT_FRAMES + 1],
                relative_yaw[:, offset:offset + OFFICIAL_CONTEXT_FRAMES + 1],
                self_pred_traj=False,
            )
        predicted_latents.append(predicted.detach().cpu())
        predicted_sequence = predicted.reshape(
            1, 1, config.image_size[0] // config.downsample_size,
            config.image_size[1] // config.downsample_size,
            config.vae_embed_dim,
        ).flatten(2, 3)
        start_latents = torch.cat(
            [start_latents[:, 1:OFFICIAL_CONTEXT_FRAMES], predicted_sequence],
            dim=1,
        )
    torch.cuda.synchronize()
    inference_seconds = time.monotonic() - inference_start
    peak_memory_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    decoded = []
    decode_start = time.monotonic()
    for start in range(0, len(predicted_latents),
                       OFFICIAL_TEMPORAL_DECODE_PATCH):
        group = predicted_latents[
            start:start + OFFICIAL_TEMPORAL_DECODE_PATCH]
        if len(group) != OFFICIAL_TEMPORAL_DECODE_PATCH:
            raise RuntimeError("temporal DCAE received a partial patch")
        images = tokenizer.z_to_image(
            torch.cat(group, dim=0).cuda(),
            is_video=True,
        )
        images = functional.interpolate(
            images,
            size=(args.dump_height, args.dump_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        decoded.extend(
            image.clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
            for image in images)
    torch.cuda.synchronize()
    decode_seconds = time.monotonic() - decode_start
    return torch.stack(decoded[:args.prediction_frames]), {
        "input_seconds": input_seconds,
        "inference_seconds": inference_seconds,
        "decode_seconds": decode_seconds,
        "peak_memory_gib": peak_memory_gib,
        "rollout_frames": rollout_frames,
        "scored_prediction_frames": args.prediction_frames,
        "discarded_causal_tail_frames": (
            rollout_frames - args.prediction_frames),
    }


def main():
    args = parse_args()
    if not (0 <= args.start_index <= args.end_index <= COMMON_CLIP_COUNT):
        raise ValueError("invalid clip range")
    if temporal_rollout_frames(args.prediction_frames) > (
            COMMON_SEQUENCE_LENGTH - OFFICIAL_CONTEXT_FRAMES):
        raise ValueError("prediction interval escapes the common clip")
    for path in (args.config, args.checkpoint, args.vae, args.ann,
                 args.pano_rgb, args.canonical_real_dir):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    output = Path(args.output)
    fake_dir = output / "fake"
    manifest_dir = output / "manifests"
    fake_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    config = load_official_config(args.config, args.vae)
    clips = load_common_clips(args.ann)
    real_files = canonical_real_files(args.canonical_real_dir)
    loader = CommonBandLoader(args.pano_rgb)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    model = TrainTransformersDiT(
        config,
        load_path=args.checkpoint,
        local_rank=0,
        condition_frames=OFFICIAL_CONTEXT_FRAMES,
    ).eval()
    tokenizer = VAETokenizer(config, 0)

    records = []
    expected_frames = OFFICIAL_CONTEXT_FRAMES + args.prediction_frames
    for clip_index in range(args.start_index, args.end_index):
        clip = clips[clip_index]
        canonical_path = real_files[clip_index]
        canonical_token = canonical_path.stem.split("_", 1)[1]
        if canonical_token != clip[0]["token"]:
            raise ValueError(
                "clip {} token mismatch: canonical={} enumerated={}".format(
                    clip_index, canonical_token, clip[0]["token"]))
        target = fake_dir / canonical_path.name
        if target.exists():
            validate_existing(
                target, expected_frames, args.dump_height, args.dump_width)
            print("skip validated {}".format(target), flush=True)
            continue

        # Make random output independent of job-array sharding and resume order.
        clip_seed = int(args.seed) * 1_000_003 + clip_index
        random.seed(clip_seed)
        np.random.seed(clip_seed % (2 ** 32))
        torch.manual_seed(clip_seed)
        torch.cuda.manual_seed_all(clip_seed)
        predictions, timing = generate_clip(
            model, tokenizer, loader, clip, args, config)
        canonical = torch.load(
            canonical_path, map_location="cpu", weights_only=True)
        if canonical.dtype != torch.uint8 or tuple(canonical.shape) != (
                COMMON_SEQUENCE_LENGTH, 3, args.dump_height, args.dump_width):
            raise ValueError(
                "unexpected canonical tensor {} {}".format(
                    tuple(canonical.shape), canonical.dtype))
        result = torch.cat([
            canonical[:OFFICIAL_CONTEXT_FRAMES], predictions,
        ], dim=0)
        atomic_torch_save(result, target)
        record = {
            "clip_index": clip_index,
            "clip_token": clip[0]["token"],
            "clip_seed": clip_seed,
            "timestamps_us": [int(info["timestamp"]) for info in clip],
            **timing,
        }
        records.append(record)
        print(
            "completed clip={} inference_s={:.3f} peak_GiB={:.3f}".format(
                clip_index, timing["inference_seconds"],
                timing["peak_memory_gib"]),
            flush=True,
        )

    manifest = {
        "method": "Epona",
        "upstream_commit": args.upstream_commit,
        "official_checkpoint": os.path.abspath(args.checkpoint),
        "official_checkpoint_sha256": args.checkpoint_sha256,
        "official_vae": os.path.abspath(args.vae),
        "official_vae_sha256": args.vae_sha256,
        "official_config": os.path.abspath(args.config),
        "official_context_frames": OFFICIAL_CONTEXT_FRAMES,
        "official_sampling_steps_per_generated_frame": OFFICIAL_SAMPLING_STEPS,
        "official_temporal_vae": {
            "enabled": True,
            "temporal_patch_size": OFFICIAL_TEMPORAL_DECODE_PATCH,
            "source": (
                "Kevin-thu/Epona Hugging Face model card; this corrects the "
                "stale image-DCAE defaults in the released GitHub config"
            ),
        },
        "common_protocol": {
            "clips": COMMON_CLIP_COUNT,
            "sequence_frames": COMMON_SEQUENCE_LENGTH,
            "fps": COMMON_FPS,
            "target_start": OFFICIAL_CONTEXT_FRAMES,
            "target_frames": args.prediction_frames,
            "rollout_frames": temporal_rollout_frames(
                args.prediction_frames),
            "discarded_causal_tail_frames": (
                temporal_rollout_frames(args.prediction_frames)
                - args.prediction_frames),
            "native_generation_resolution": list(OFFICIAL_IMAGE_SIZE),
            "metric_resolution": [args.dump_height, args.dump_width],
            "camera": "CAM_FRONT band reconstructed from frozen panorama cache",
            "quantization": "round(clamp(x,0,1)*255)",
            "conditioning_disclosure": (
                "Epona uses 10 history frames; PanDWM/HorizonDrive use 3. "
                "All methods are scored on identical target frames 10..25."
            ),
        },
        "generation_seed": args.seed,
        "range": [args.start_index, args.end_index],
        "records": records,
    }
    manifest_path = manifest_dir / "{:04d}_{:04d}.json".format(
        args.start_index, args.end_index)
    atomic_json_dump(manifest, manifest_path)
    print("wrote {}".format(manifest_path), flush=True)


if __name__ == "__main__":
    main()
