#!/usr/bin/env python3
"""Validate the immutable inputs needed by HorizonDrive evaluation.

This preflight mirrors the paths used by the upstream dataset instead of only
checking that a top-level directory exists.  It deliberately does not create,
download, or rewrite any data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_COMMIT = "6a8ab2bb6daf5e1179e125ee1531c3ba5219142d"
EXPECTED_SCENES = 150
EXPECTED_INFOS = 35_364
OLD_NUSCENES_PREFIX = "../data/nuscenes"

MODEL_FILES = {
    "horizondrive-dit.safetensors": 2_888_847_048,
    "horizondrive-vae.pkl": 1_169_123_082,
    "Wan2.1-T2V-1.3B/config.json": 249,
    "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors": 5_676_070_424,
    "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth": 11_361_920_418,
    "Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json": 6_623,
    "Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model": 4_548_313,
    "Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json": 16_837_417,
    "Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json": 61_728,
}

METRIC_FILES = (
    "fvd/styleganv/i3d_torchscript.pt",
    "fid/weights-inception-2015-12-05-6726825d.pth",
)

MAP_NAMES = (
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-onenorth",
    "singapore-queenstown",
)

FRAME_KEYS = (
    "token",
    "timestamp",
    "location",
    "cams",
    "gt_boxes",
    "gt_names",
    "gt_box_ids",
    "valid_flag",
    "lidar2ego_rotation",
    "lidar2ego_translation",
)

CAMERA_KEYS = (
    "data_path",
    "sensor2lidar_rotation",
    "sensor2lidar_translation",
    "sensor2ego_rotation",
    "sensor2ego_translation",
    "ego2global_rotation",
    "ego2global_translation",
    "camera_intrinsics",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("HD_REPO", "HorizonDrive")),
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=Path(os.environ.get("HD_MODELS", "HorizonDrive/models")),
    )
    parser.add_argument(
        "--nuscenes-root",
        type=Path,
        default=Path(os.environ.get("HD_NUSCENES_ROOT", "nuScenes/origin")),
        help="nuScenes dataroot containing samples/ and maps/",
    )
    parser.add_argument(
        "--ann",
        type=Path,
        default=Path(os.environ.get("HD_ANN_PATH", "nuscenes_val.pkl")),
    )
    parser.add_argument(
        "--sha256-manifest",
        type=Path,
        default=None,
        help="optional sha256sum-compatible manifest, relative to --models",
    )
    parser.add_argument(
        "--image-check",
        choices=("none", "first", "all"),
        default="all",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    entries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(
                    f"invalid manifest line {line_number}: {raw_line.rstrip()}"
                )
            entries[fields[1].lstrip("*")] = fields[0].lower()
    return entries


def resolve_image_path(stored_path: str, nuscenes_root: Path) -> Path:
    return Path(stored_path.replace(OLD_NUSCENES_PREFIX, str(nuscenes_root)))


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "repo": str(args.repo),
        "models": str(args.models),
        "nuscenes_root": str(args.nuscenes_root),
        "ann": str(args.ann),
    }

    def error(message: str) -> None:
        report["ok"] = False
        report["errors"].append(message)

    commit = git_commit(args.repo)
    report["commit"] = commit
    if commit != EXPECTED_COMMIT:
        error(f"repo commit is {commit!r}, expected {EXPECTED_COMMIT}")

    manifest = load_manifest(args.sha256_manifest)
    checked_models: dict[str, Any] = {}
    for relative_path, expected_size in MODEL_FILES.items():
        path = args.models / relative_path
        item = {"exists": path.is_file(), "expected_size": expected_size}
        checked_models[relative_path] = item
        if not path.is_file():
            error(f"missing model file: {path}")
            continue
        item["size"] = path.stat().st_size
        if item["size"] != expected_size:
            error(
                f"wrong size for {path}: {item['size']} != {expected_size}"
            )
        if relative_path in manifest:
            item["sha256"] = sha256(path)
            item["expected_sha256"] = manifest[relative_path]
            if item["sha256"] != item["expected_sha256"]:
                error(f"sha256 mismatch: {path}")

    for relative_path in METRIC_FILES:
        path = args.models / relative_path
        item = {"exists": path.is_file()}
        checked_models[relative_path] = item
        if not path.is_file():
            error(f"missing metric weight: {path}")
            continue
        item["size"] = path.stat().st_size
        if relative_path in manifest:
            item["sha256"] = sha256(path)
            item["expected_sha256"] = manifest[relative_path]
            if item["sha256"] != item["expected_sha256"]:
                error(f"sha256 mismatch: {path}")
    report["model_files"] = checked_models

    for map_name in MAP_NAMES:
        path = args.nuscenes_root / "maps" / "expansion" / f"{map_name}.json"
        if not path.is_file():
            error(f"missing nuScenes expansion map: {path}")

    if not args.ann.is_file():
        error(f"missing annotation file: {args.ann}")
    else:
        try:
            with args.ann.open("rb") as handle:
                annotations = pickle.load(handle)
            infos = annotations["infos"]
            scenes = annotations["scene_tokens"]
            metadata = annotations["metadata"]
            report["annotation"] = {
                "infos": len(infos),
                "scenes": len(scenes),
                "version": metadata.get("version"),
                "min_scene_frames": min(map(len, scenes)),
                "max_scene_frames": max(map(len, scenes)),
            }
            if len(infos) != EXPECTED_INFOS:
                error(f"annotation has {len(infos)} infos, expected {EXPECTED_INFOS}")
            if len(scenes) != EXPECTED_SCENES:
                error(f"annotation has {len(scenes)} scenes, expected {EXPECTED_SCENES}")
            if report["annotation"]["min_scene_frames"] < 211:
                error(
                    "annotation contains a scene shorter than the 211-frame "
                    f"paper protocol: {report['annotation']['min_scene_frames']}"
                )

            info_tokens = {info.get("token") for info in infos}
            missing_scene_tokens = [
                token
                for scene in scenes
                for token in scene
                if token not in info_tokens
            ]
            if missing_scene_tokens:
                error(
                    "scene_tokens contains tokens absent from infos; examples: "
                    + ", ".join(missing_scene_tokens[:3])
                )

            first = infos[0]
            missing_frame_keys = [key for key in FRAME_KEYS if key not in first]
            if missing_frame_keys:
                error(f"annotation frame misses keys: {missing_frame_keys}")
            front = first.get("cams", {}).get("CAM_FRONT", {})
            missing_camera_keys = [key for key in CAMERA_KEYS if key not in front]
            if missing_camera_keys:
                error(f"CAM_FRONT metadata misses keys: {missing_camera_keys}")

            image_infos = infos
            if args.image_check == "none":
                image_infos = []
            elif args.image_check == "first":
                image_infos = infos[:1]
            missing_images = []
            unique_paths = set()
            for info in image_infos:
                stored_path = info.get("cams", {}).get("CAM_FRONT", {}).get(
                    "data_path"
                )
                if not stored_path:
                    missing_images.append("<missing CAM_FRONT data_path>")
                    continue
                path = resolve_image_path(stored_path, args.nuscenes_root)
                unique_paths.add(str(path))
                if not path.is_file() and len(missing_images) < 20:
                    missing_images.append(str(path))
            report["annotation"]["checked_unique_front_images"] = len(unique_paths)
            report["annotation"]["missing_front_image_examples"] = missing_images
            if missing_images:
                error(
                    "CAM_FRONT images are missing; first examples: "
                    + ", ".join(missing_images[:3])
                )
        except Exception as exc:
            error(f"could not validate annotation: {type(exc).__name__}: {exc}")

    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized + "\n", encoding="utf-8")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
