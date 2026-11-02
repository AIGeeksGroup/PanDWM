#!/usr/bin/env python3
"""Inject OpenDWM environment captions into a nuScenes info pickle."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Any


CAMERA_PREFERENCE = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
FALLBACK_DESCRIPTION = "urban road"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add OpenDWM environment captions to a nuScenes info pkl."
    )
    parser.add_argument("--pkl-in", required=True, type=Path)
    parser.add_argument("--caption-zip", required=True, type=Path)
    parser.add_argument("--nuscenes-meta-dir", required=True, type=Path)
    parser.add_argument("--pkl-out", type=Path)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print pkl and caption schemas, then exit without writing.",
    )
    args = parser.parse_args()
    if not args.inspect and args.pkl_out is None:
        parser.error("--pkl-out is required unless --inspect is used")
    return args


def load_pickle(path: Path) -> tuple[Any, list[dict[str, Any]], str]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        infos = payload.get("infos")
        form = "dict"
    elif isinstance(payload, list):
        infos = payload
        form = "list"
    else:
        raise TypeError(
            "Unsupported pkl top-level type: " + type(payload).__name__
        )

    if not isinstance(infos, list):
        raise TypeError("The pkl infos value must be a list")
    for index, info in enumerate(infos):
        if not isinstance(info, dict):
            raise TypeError(f"infos[{index}] is not a dict")
    return payload, infos, form


def load_caption_zip(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[int]], list[str]]:
    captions: dict[str, dict[str, Any]] = {}
    times: dict[str, list[int]] = {}
    loaded_names: list[str] = []

    with zipfile.ZipFile(path) as archive:
        json_names = sorted(
            name for name in archive.namelist() if name.lower().endswith(".json")
        )
        caption_names = [
            name for name in json_names if "_caption_v2_" in name and "_times_" not in name
        ]
        time_names = [
            name for name in json_names if "_caption_v2_times_" in name
        ]
        if not caption_names or not time_names:
            raise ValueError(
                "Caption zip must contain caption_v2 and caption_v2_times JSON files"
            )

        for name in caption_names:
            with archive.open(name) as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise TypeError(f"Caption JSON is not a dict: {name}")
            overlap = captions.keys() & data.keys()
            if overlap:
                raise ValueError(f"Duplicate caption key across JSON files: {next(iter(overlap))}")
            captions.update(data)
            loaded_names.append(name)

        for name in time_names:
            with archive.open(name) as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise TypeError(f"Caption times JSON is not a dict: {name}")
            overlap = times.keys() & data.keys()
            if overlap:
                raise ValueError(f"Duplicate times key across JSON files: {next(iter(overlap))}")
            for key, value in data.items():
                if not isinstance(value, list) or not value:
                    raise ValueError(f"Caption times entry must be a non-empty list: {key}")
                times[key] = sorted(int(item) for item in value)
            loaded_names.append(name)

    return captions, times, sorted(loaded_names)


def print_inspection(
    payload: Any,
    infos: list[dict[str, Any]],
    form: str,
    captions: dict[str, dict[str, Any]],
    times: dict[str, list[int]],
    caption_names: list[str],
) -> None:
    print("PKL top-level form:", form)
    if isinstance(payload, dict):
        print("PKL top-level keys:", sorted(str(key) for key in payload.keys()))
    print("Info count:", len(infos))
    print("First info keys:", sorted(infos[0].keys()) if infos else [])
    print("Caption JSON members:", caption_names)
    print("Caption entry count:", len(captions))
    print("Caption times entry count:", len(times))
    if captions:
        key = next(iter(captions))
        value = captions[key]
        print("Caption key example:", key)
        print("Caption key fields:", key.split("|"))
        print(
            "Caption value type and keys:",
            type(value).__name__,
            sorted(value.keys()) if isinstance(value, dict) else [],
        )
        print("Caption value example:", json.dumps(value, ensure_ascii=True))
    if times:
        key = next(iter(times))
        print("Caption times key example:", key)
        print("Caption times sample:", times[key][:3])


def load_sample_to_scene(meta_dir: Path) -> dict[str, str]:
    sample_path = meta_dir / "sample.json"
    with sample_path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list):
        raise TypeError("sample.json must contain a list")

    result: dict[str, str] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise TypeError(f"sample.json entry {index} is not a dict")
        token = sample.get("token")
        scene_token = sample.get("scene_token")
        if not isinstance(token, str) or not isinstance(scene_token, str):
            raise ValueError(f"sample.json entry {index} lacks token or scene_token")
        result[token] = scene_token
    return result


def find_nearest(sorted_values: list[int], target: int) -> int:
    index = bisect.bisect_left(sorted_values, target)
    if index == 0:
        return sorted_values[0]
    if index == len(sorted_values):
        return sorted_values[-1]
    left = sorted_values[index - 1]
    right = sorted_values[index]
    return left if target - left <= right - target else right


def find_environment(
    scene_token: str,
    timestamp: int,
    captions: dict[str, dict[str, Any]],
    times: dict[str, list[int]],
) -> str | None:
    for camera in CAMERA_PREFERENCE:
        scene_camera = f"{scene_token}|{camera}"
        timeline = times.get(scene_camera)
        if not timeline:
            continue
        nearest = find_nearest(timeline, timestamp)
        caption = captions.get(f"{scene_camera}|{nearest}")
        if not isinstance(caption, dict):
            continue
        environment = caption.get("environment")
        if isinstance(environment, str) and environment.strip():
            return environment.strip()
    return None


def inject_descriptions(
    infos: list[dict[str, Any]],
    sample_to_scene: dict[str, str],
    captions: dict[str, dict[str, Any]],
    times: dict[str, list[int]],
) -> tuple[int, int]:
    real_count = 0
    fallback_count = 0

    for index, info in enumerate(infos):
        token = info.get("token")
        timestamp = info.get("timestamp")
        if not isinstance(token, str) or len(token) < 32:
            raise ValueError(f"infos[{index}] has an invalid token")
        try:
            timestamp_int = int(timestamp)
        except (TypeError, ValueError) as error:
            raise ValueError(f"infos[{index}] has an invalid timestamp") from error

        # A 33-character interpolation token starts with its 32-character
        # keyframe sample token. sample.json maps that token to a scene token.
        sample_token = token[:32]
        scene_token = sample_to_scene.get(sample_token)
        environment = None
        if scene_token is not None:
            environment = find_environment(
                scene_token, timestamp_int, captions, times
            )

        if environment is not None:
            info["description"] = environment
            real_count += 1
        else:
            existing = info.get("description")
            info["description"] = (
                existing.strip()
                if isinstance(existing, str) and existing.strip()
                else FALLBACK_DESCRIPTION
            )
            fallback_count += 1

    return real_count, fallback_count


def atomic_pickle_dump(payload: Any, output_path: Path) -> None:
    output_parent = output_path.parent
    if not output_parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory does not exist: {output_parent}"
        )

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=output_path.name + ".tmp.",
            dir=output_parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    payload, infos, form = load_pickle(args.pkl_in)
    captions, times, caption_names = load_caption_zip(args.caption_zip)

    if args.inspect:
        print_inspection(
            payload, infos, form, captions, times, caption_names
        )
        return

    assert args.pkl_out is not None
    if args.pkl_in.resolve() == args.pkl_out.resolve():
        raise ValueError("--pkl-out must differ from --pkl-in")

    sample_to_scene = load_sample_to_scene(args.nuscenes_meta_dir)
    real_count, fallback_count = inject_descriptions(
        infos, sample_to_scene, captions, times
    )
    if real_count + fallback_count != len(infos):
        raise RuntimeError("Internal caption accounting error")

    output_payload = payload if form == "dict" else {"infos": infos}
    atomic_pickle_dump(output_payload, args.pkl_out)
    print("Caption injection complete")
    print("Total infos:", len(infos))
    print("Infos with real caption:", real_count)
    print("Infos falling back:", fallback_count)
    print("Output:", args.pkl_out)


if __name__ == "__main__":
    main()
