#!/usr/bin/env python3
"""Build PKL shards for official 8-way OmniNWM sequence-parallel inference.

The official ``hybrid`` plugin sets ``sp_size=8`` and consequently creates
eight singleton data-parallel groups when inference is launched on eight
GPUs.  ``prepare_dataloader`` constructs its ``DistributedSampler`` over that
data-parallel group (size 1), not over WORLD.  Each sequence-parallel rank must
therefore iterate the same unreplicated scene list.  This adapter leaves the
official source untouched and only makes small NumPy-1-compatible PKL shards.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


class NumpyCompatUnpickler(pickle.Unpickler):
    """Read NumPy-2 pickles in the official NumPy-1 OmniNWM environment."""

    def find_class(self, module, name):
        if module == "numpy._core":
            module = "numpy.core"
        elif module.startswith("numpy._core."):
            module = "numpy.core." + module[len("numpy._core."):]
        return super().find_class(module, name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--clips-per-shard", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)

    with args.data.open("rb") as handle:
        payload = NumpyCompatUnpickler(handle).load()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scenes = payload["scene_tokens"]
    if len(scenes) != manifest["clips"]:
        raise ValueError((len(scenes), manifest["clips"]))
    if any(len(scene) != 33 for scene in scenes):
        raise ValueError("every OmniNWM scene must contain exactly 33 frames")

    info_by_token = {item["token"]: item for item in payload["infos"]}
    if len(info_by_token) != len(payload["infos"]):
        raise ValueError("duplicate info tokens in source PKL")

    summary = {
        "protocol": "official-hybrid-sp8-dp1-unreplicated",
        "source_data": str(args.data),
        "source_manifest": str(args.manifest),
        "sequence_parallel_size": 8,
        "data_parallel_size": 1,
        "clips_per_shard": args.clips_per_shard,
        "logical_clips": len(scenes),
        "shards": [],
    }
    for shard_id, start in enumerate(range(0, len(scenes), args.clips_per_shard)):
        logical_scenes = scenes[start:start + args.clips_per_shard]
        logical_ids = list(range(start, start + len(logical_scenes)))
        selected_tokens = {token for scene in logical_scenes for token in scene}
        selected_infos = [
            item for item in payload["infos"] if item["token"] in selected_tokens
        ]
        if len(selected_infos) != 33 * len(logical_scenes):
            raise ValueError(
                f"shard {shard_id}: expected {33 * len(logical_scenes)} infos, "
                f"found {len(selected_infos)}")

        target = args.out / f"shard_{shard_id:02d}.pkl"
        shard_payload = {
            "infos": selected_infos,
            "metadata": payload["metadata"],
            "scene_tokens": logical_scenes,
        }
        with target.open("wb") as handle:
            pickle.dump(shard_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        summary["shards"].append({
            "shard": shard_id,
            "path": str(target),
            "logical_clip_ids": logical_ids,
            "logical_count": len(logical_ids),
            "dataset_entries": len(logical_scenes),
            "per_sp_rank_order": logical_ids,
            "infos": len(selected_infos),
        })
        print(
            f"wrote {target} logical={len(logical_ids)} "
            f"dataset_entries={len(logical_scenes)}", flush=True)

    summary_path = args.out / "sampler_audit.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
