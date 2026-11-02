#!/usr/bin/env python3
"""Keep complete scenes whose timestamp-keyed local files exist."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any


ENV_PYTHON = "/path/to/pandwm_root/envs/pandwm/bin/python3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a PanDWM info pkl to scenes with complete local files."
        ),
        epilog=(
            "Usage on h20: " + ENV_PYTHON
            + " scripts/h20/filter_pkl_to_available.py --pkl-in IN "
            "--rgb-dir RGB [--seg-dir SEG] [--require-depth DEPTH] "
            "--pkl-out OUT [--holdout-scenes N --holdout-out HOLDOUT] "
            "[--min-scene-frames N]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pkl-in", required=True, type=Path)
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--seg-dir", type=Path)
    parser.add_argument(
        "--require-depth",
        type=Path,
        metavar="DIR",
        help="Also require a timestamp-keyed .h5 file in DIR.",
    )
    parser.add_argument("--pkl-out", required=True, type=Path)
    parser.add_argument(
        "--holdout-scenes",
        type=int,
        default=0,
        metavar="N",
        help="Move the lexicographically last N surviving scenes to holdout.",
    )
    parser.add_argument(
        "--holdout-out",
        type=Path,
        metavar="PATH",
        help="Output pkl for scenes removed from training.",
    )
    parser.add_argument(
        "--min-scene-frames",
        type=int,
        default=1,
        metavar="N",
        help="Require at least N infos in a complete scene (default: 1).",
    )
    args = parser.parse_args()
    if args.min_scene_frames < 1:
        parser.error("--min-scene-frames must be at least 1")
    if args.holdout_scenes < 0:
        parser.error("--holdout-scenes must be nonnegative")
    if args.holdout_scenes > 0 and args.holdout_out is None:
        parser.error("--holdout-out is required when --holdout-scenes is set")
    if args.holdout_scenes == 0 and args.holdout_out is not None:
        parser.error("--holdout-out requires --holdout-scenes N with N > 0")
    return args


def require_numpy_126() -> None:
    try:
        import numpy
    except ImportError as error:
        raise RuntimeError(
            "NumPy is unavailable. Run this script with " + ENV_PYTHON
        ) from error

    version_parts = numpy.__version__.split(".")
    if version_parts[:2] != ["1", "26"]:
        raise RuntimeError(
            "Refusing to write a pkl with NumPy " + numpy.__version__
            + "; run with " + ENV_PYTHON + " (NumPy 1.26)."
        )


def load_payload(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("The pkl top level must be a dict")
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise TypeError("The pkl 'infos' value must be a list")
    for index, info in enumerate(infos):
        if not isinstance(info, dict):
            raise TypeError(f"infos[{index}] is not a dict")
    return payload, infos


def group_from_scene_tokens(
    infos: list[dict[str, Any]], scene_tokens: Any
) -> list[list[dict[str, Any]]]:
    if not isinstance(scene_tokens, list):
        raise TypeError("The pkl 'scene_tokens' value must be a list")
    if not all(isinstance(members, list) for members in scene_tokens):
        raise TypeError("Each scene_tokens entry must be a token list")

    exact_to_scene: dict[str, int] = {}
    parent_to_scene: dict[str, int] = {}
    listed_tokens: set[str] = set()
    for scene_index, members in enumerate(scene_tokens):
        if not members:
            raise ValueError(f"scene_tokens[{scene_index}] is empty")
        for member in members:
            if not isinstance(member, str) or len(member) < 32:
                raise ValueError(
                    f"scene_tokens[{scene_index}] has an invalid token"
                )
            if member in listed_tokens:
                raise ValueError("scene_tokens contains a duplicate token: " + member)
            listed_tokens.add(member)
            exact_to_scene[member] = scene_index
            parent = member[:32]
            previous = parent_to_scene.setdefault(parent, scene_index)
            if previous != scene_index:
                raise ValueError(
                    "A token[:32] parent appears in more than one scene: "
                    + parent
                )

    groups: list[list[dict[str, Any]]] = [
        [] for _ in range(len(scene_tokens))
    ]
    seen_info_tokens: set[str] = set()
    for info_index, info in enumerate(infos):
        token = info.get("token")
        if not isinstance(token, str) or len(token) < 32:
            raise ValueError(f"infos[{info_index}] has an invalid token")
        if token in seen_info_tokens:
            raise ValueError("infos contains a duplicate token: " + token)
        seen_info_tokens.add(token)
        scene_index = exact_to_scene.get(token)
        if scene_index is None:
            scene_index = parent_to_scene.get(token[:32])
        if scene_index is None:
            raise ValueError(
                f"infos[{info_index}] is absent from scene_tokens: {token}"
            )
        groups[scene_index].append(info)

    if seen_info_tokens != listed_tokens:
        missing_infos = sorted(listed_tokens - seen_info_tokens)
        unlisted_infos = sorted(seen_info_tokens - listed_tokens)
        raise ValueError(
            "scene_tokens is inconsistent with infos; missing infos: "
            + repr(missing_infos[:3])
            + ", unlisted infos: "
            + repr(unlisted_infos[:3])
        )
    return groups


def group_from_info_scene_token(
    infos: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for info_index, info in enumerate(infos):
        scene_token = info.get("scene_token")
        if not isinstance(scene_token, str) or not scene_token:
            raise ValueError(
                "The pkl has no scene_tokens list and "
                f"infos[{info_index}] has no scene_token"
            )
        groups.setdefault(scene_token, []).append(info)
    return list(groups.values())


def group_infos(
    payload: dict[str, Any], infos: list[dict[str, Any]]
) -> tuple[list[list[dict[str, Any]]], list[str], str]:
    if "scene_tokens" in payload:
        groups = group_from_scene_tokens(infos, payload["scene_tokens"])
        scene_keys = []
        for scene_index, (members, scene) in enumerate(
            zip(payload["scene_tokens"], groups)
        ):
            explicit_keys = {
                info["scene_token"]
                for info in scene
                if isinstance(info.get("scene_token"), str)
                and info["scene_token"]
            }
            if len(explicit_keys) > 1:
                raise ValueError(
                    f"scene_tokens[{scene_index}] has multiple scene_token values"
                )
            if explicit_keys:
                scene_keys.append(next(iter(explicit_keys)))
            else:
                # Preserve P1's token[:32] parent semantics exactly. The
                # public pkl has per-scene sample-token lists rather than an
                # explicit official scene token, so the smallest parent token
                # is the stable scene key in that schema.
                scene_keys.append(min(member[:32] for member in members))
        return (
            groups,
            scene_keys,
            "top-level scene_tokens",
        )
    groups = group_from_info_scene_token(infos)
    return groups, [scene[0]["scene_token"] for scene in groups], \
        "info scene_token"


def available_names(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory is missing: {directory}")
    with os.scandir(directory) as entries:
        return {
            entry.name
            for entry in entries
            if entry.is_file() and entry.name.endswith(suffix)
        }


def scene_is_available(
    scene: list[dict[str, Any]],
    rgb_names: set[str],
    seg_names: set[str] | None,
    depth_names: set[str] | None,
    min_scene_frames: int,
) -> bool:
    if len(scene) < min_scene_frames:
        return False
    for info in scene:
        try:
            timestamp = int(info["timestamp"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("An info has an invalid timestamp") from error
        if f"{timestamp}.jpg" not in rgb_names:
            return False
        if seg_names is not None and f"{timestamp}.png" not in seg_names:
            return False
        if depth_names is not None and f"{timestamp}.h5" not in depth_names:
            return False
    return True


def make_split_payload(
    payload: dict[str, Any],
    infos: list[dict[str, Any]],
    scenes: list[list[dict[str, Any]]],
    selected_indices: list[int],
) -> dict[str, Any]:
    selected_index_set = set(selected_indices)
    selected_info_ids = {
        id(info)
        for index, scene in enumerate(scenes)
        if index in selected_index_set
        for info in scene
    }
    output_payload = dict(payload)
    output_payload["infos"] = [
        info for info in infos if id(info) in selected_info_ids
    ]
    if "scene_tokens" in payload:
        output_payload["scene_tokens"] = [
            payload["scene_tokens"][index] for index in selected_indices
        ]
    return output_payload


def atomic_pickle_dump(payload: dict[str, Any], output_path: Path) -> None:
    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory is missing: {output_path.parent}"
        )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=output_path.name + ".tmp.",
            dir=output_path.parent,
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
    require_numpy_126()
    if args.pkl_in.resolve() == args.pkl_out.resolve():
        raise ValueError("--pkl-out must differ from --pkl-in")
    if args.holdout_out is not None:
        resolved_paths = {
            args.pkl_in.resolve(),
            args.pkl_out.resolve(),
            args.holdout_out.resolve(),
        }
        if len(resolved_paths) != 3:
            raise ValueError(
                "--pkl-in, --pkl-out, and --holdout-out must differ"
            )
        for output_path in (args.pkl_out, args.holdout_out):
            if not output_path.parent.is_dir():
                raise FileNotFoundError(
                    f"Output parent directory is missing: {output_path.parent}"
                )
    elif not args.pkl_out.parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory is missing: {args.pkl_out.parent}"
        )

    payload, infos = load_payload(args.pkl_in)
    scenes, scene_keys, grouping_mode = group_infos(payload, infos)
    if len(set(scene_keys)) != len(scene_keys):
        raise ValueError("Scene split keys are not unique")
    rgb_names = available_names(args.rgb_dir, ".jpg")
    seg_names = (
        available_names(args.seg_dir, ".png")
        if args.seg_dir is not None
        else None
    )
    depth_names = (
        available_names(args.require_depth, ".h5")
        if args.require_depth is not None
        else None
    )

    kept_indices = [
        index
        for index, scene in enumerate(scenes)
        if scene_is_available(
            scene,
            rgb_names,
            seg_names,
            depth_names,
            args.min_scene_frames,
        )
    ]
    kept_info_count = sum(len(scenes[index]) for index in kept_indices)

    print("Scene grouping:", grouping_mode)
    print(f"Scenes kept/total: {len(kept_indices)}/{len(scenes)}")
    print(f"Infos kept/total: {kept_info_count}/{len(infos)}")
    if not kept_indices:
        raise RuntimeError("Refusing to write a pkl with 0 surviving scenes")

    holdout_indices: list[int] = []
    train_indices = kept_indices
    if args.holdout_scenes > 0:
        if args.holdout_scenes >= len(kept_indices):
            raise RuntimeError(
                "Refusing to write empty train or holdout split: "
                f"requested {args.holdout_scenes} holdout scenes from "
                f"{len(kept_indices)} surviving scenes"
            )
        holdout_indices = sorted(
            kept_indices,
            key=lambda index: scene_keys[index],
        )[-args.holdout_scenes:]
        holdout_index_set = set(holdout_indices)
        train_indices = [
            index for index in kept_indices if index not in holdout_index_set
        ]
        if not train_indices or not holdout_indices:
            raise RuntimeError("Refusing to write an empty train or holdout split")
        print("Holdout scene keys:")
        for index in holdout_indices:
            print(" ", scene_keys[index])

    train_payload = make_split_payload(
        payload, infos, scenes, train_indices
    )
    if not train_payload["infos"]:
        raise RuntimeError("Refusing to write a train pkl with 0 infos")

    if args.holdout_out is not None:
        holdout_payload = make_split_payload(
            payload, infos, scenes, holdout_indices
        )
        if not holdout_payload["infos"]:
            raise RuntimeError("Refusing to write a holdout pkl with 0 infos")
        atomic_pickle_dump(holdout_payload, args.holdout_out)
        print(
            f"Holdout scenes/infos: {len(holdout_indices)}/"
            f"{len(holdout_payload['infos'])}"
        )
        print("Wrote:", args.holdout_out)

    atomic_pickle_dump(train_payload, args.pkl_out)
    print(
        f"Train scenes/infos: {len(train_indices)}/"
        f"{len(train_payload['infos'])}"
    )
    print("Wrote:", args.pkl_out)


if __name__ == "__main__":
    main()
