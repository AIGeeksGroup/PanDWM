"""Create frozen, scene-token-based Slurm evaluation task lists."""

import argparse
import json
from pathlib import Path


def build_tasks(split, kind):
    dev = split["dev"]
    if len(dev) != 8 or len({s["scene_token"] for s in dev}) != 8:
        raise ValueError("expected eight unique development scene tokens")
    if kind == "baseline":
        arms = [("baseline", seed) for seed in (0, 42, 1000)]
        arms += [(arm, 42) for arm in ("control", "self")]
        return [dict(scene_token=s["scene_token"], scene_index=s["index"],
                     arm=arm, seed=seed, windows=12, history_mode="clean",
                     noise_sigma=0.0, align_alpha=0.5)
                for s in dev for arm, seed in arms]
    if kind == "diagnostic":
        already_seen = [s for s in dev if s["index"] in (14, 32, 49)]
        variants = (("flow_noise", 0.02, 0.5),
                    ("flow_noise", 0.05, 0.5),
                    ("vae_roundtrip", 0.0, 0.5),
                    ("moment_align", 0.0, 0.5))
        return [dict(scene_token=s["scene_token"], scene_index=s["index"],
                     arm="baseline", seed=42, windows=12,
                     history_mode=mode, noise_sigma=sigma,
                     align_alpha=alpha)
                for s in already_seen for mode, sigma, alpha in variants]
    if kind == "diagnostic_validate":
        return [dict(scene_token=s["scene_token"], scene_index=s["index"],
                     arm="baseline", seed=seed, windows=12,
                     history_mode="flow_noise", noise_sigma=0.05,
                     align_alpha=0.5)
                for s in dev for seed in (0, 42, 1000)]
    if kind == "noise_pilot":
        return [dict(scene_token=s["scene_token"], scene_index=s["index"],
                     arm=arm, seed=seed, windows=12, history_mode=mode,
                     noise_sigma=sigma, align_alpha=0.5)
                for s in dev for arm in ("noise_control", "noise")
                for seed in (42, 1000)
                for mode, sigma in (("clean", 0.0), ("flow_noise", 0.02))]
    raise ValueError("unknown task kind")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--kind", choices=("baseline", "diagnostic",
                                           "diagnostic_validate", "noise_pilot"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = build_tasks(json.loads(args.split.read_text()), args.kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"kind": args.kind, "tasks": tasks}, indent=2) + "\n")
    print(f"{args.output}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
