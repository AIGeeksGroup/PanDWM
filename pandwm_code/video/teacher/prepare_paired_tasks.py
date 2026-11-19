"""Freeze development tasks for history ladder and condition ablation."""

import argparse
import json
from pathlib import Path


def make_tasks(split, kind):
    dev = split["dev"]
    heldout = {x["scene_token"] for x in split["heldout"]}
    if len(dev) != 8 or len(heldout) != 75:
        raise ValueError("unexpected development/heldout split")
    if {x["scene_token"] for x in dev} & heldout:
        raise ValueError("development and heldout scenes overlap")
    if kind == "ladder":
        return [dict(scene_token=s["scene_token"], scene_index=s["index"],
                     seed=seed, mode="ladder", history_mode="clean",
                     noise_sigma=0.0, condition_mask="full", windows=12)
                for s in dev for seed in (0, 42)]
    if kind == "condition_pilot":
        selected = [s for s in dev if s["index"] in (14, 32)]
        if len(selected) != 2:
            raise ValueError("missing predeclared condition pilot scenes")
        return [dict(scene_token=s["scene_token"], scene_index=s["index"],
                     seed=seed, mode="condition", history_mode=history,
                     noise_sigma=sigma, condition_mask=mask, windows=12)
                for s in selected for seed in (0, 42)
                for history, sigma in (("clean", 0.0), ("flow_noise", 0.05))
                for mask in ("full", "no_map", "no_box", "no_map_box")]
    raise ValueError(f"unknown task kind: {kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--kind", choices=("ladder", "condition_pilot"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = make_tasks(json.loads(args.split.read_text()), args.kind)
    args.output.write_text(json.dumps({"kind": args.kind, "tasks": tasks},
                                     indent=2) + "\n")
    print(args.output, len(tasks))


if __name__ == "__main__":
    main()
