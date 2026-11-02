#!/usr/bin/env python3
"""Audit whether Vista's official nuScenes windows cover common clip anchors."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from dwm_pano.datasets.nuscenes_pano import (
    _enumerate_pano_clips,
    _group_infos_by_scene,
)


def load_common_clips(annotation_path: str):
    with open(annotation_path, "rb") as handle:
        payload = pickle.load(handle)
    infos = sorted(payload["infos"], key=lambda item: item["timestamp"])
    token_to_scene = {
        token: "scene_group_{:04d}".format(scene_index)
        for scene_index, tokens in enumerate(payload["scene_tokens"])
        for token in tokens
    }
    grouped = _group_infos_by_scene(
        infos, token2scene=token_to_scene, strict=True)
    clips = [
        clip for clip, _audit in _enumerate_pano_clips(
            grouped, 35, [(10.0, 20.0)])
    ]
    if len(clips) != 149:
        raise ValueError("expected 149 common clips, found {}".format(
            len(clips)))
    return clips


def frame_timestamp(path: str) -> int:
    stem = Path(path).stem
    return int(stem.rsplit("__", 1)[1])


def common_camera_timestamp(info) -> int:
    camera = info["cams"]["CAM_FRONT"]
    for key in ("data_path", "filename", "image_path"):
        if camera.get(key):
            return frame_timestamp(camera[key])
    if camera.get("timestamp") is not None:
        return int(camera["timestamp"])
    raise KeyError("CAM_FRONT record has no path or timestamp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--common-pkl", required=True)
    args = parser.parse_args()

    with args.annotation.open(encoding="utf-8") as handle:
        annotation = json.load(handle)
    starts = {frame_timestamp(item["frames"][0]): item for item in annotation}
    clips = load_common_clips(args.common_pkl)
    info_timestamp_matches = {
        str(offset): sum(
            int(clip[offset]["timestamp"]) in starts for clip in clips)
        for offset in range(len(clips[0]))
    }
    camera_timestamp_matches = {
        str(offset): sum(
            common_camera_timestamp(clip[offset]) in starts for clip in clips)
        for offset in range(len(clips[0]))
    }
    matched_frame9 = [
        {
            "clip_index": index,
            "info_timestamp_us": int(clip[9]["timestamp"]),
            "camera_timestamp_us": common_camera_timestamp(clip[9]),
        }
        for index, clip in enumerate(clips)
        if common_camera_timestamp(clip[9]) in starts
    ]
    print(json.dumps({
        "official_annotation_windows": len(annotation),
        "common_clips": len(clips),
        "official_start_matches_by_info_timestamp_offset": (
            info_timestamp_matches),
        "official_start_matches_by_camera_timestamp_offset": (
            camera_timestamp_matches),
        "frame9_matches": matched_frame9,
    }, indent=2))


if __name__ == "__main__":
    main()
