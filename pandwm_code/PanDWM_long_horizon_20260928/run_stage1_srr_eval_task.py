"""Evaluate a paired Stage-1 checkpoint through the uncompressed RGB runner."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def task_name(task):
    return f"teacher_{task['arm']}_scene{task['scene_index']}_seed{task['seed']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--train-tag", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(args.tasks.read_text())["tasks"][args.index]
    if task["arm"] not in ("init", "control", "recovery"):
        raise ValueError("invalid arm")
    mode = task.get("mode", "condition")
    if mode not in ("ladder", "condition"):
        raise ValueError("invalid mode")
    run_root = args.data_root / "pandwm_runs/long_horizon_20260928"
    if task["arm"] == "init":
        # The shared initialization of both paired arms (srr_config/*.json).
        checkpoint = args.data_root / "pandwm_data/stage1_causal_20000.pth"
    else:
        checkpoint = (run_root / args.train_tag / task["arm"] /
                      "checkpoints" / f"{args.steps}.pth")
    output = run_root / args.run_tag / task_name(task)
    base = (args.data_root /
            "pandwm_code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_hpc3.json")
    if output.exists():
        raise FileExistsError(output)
    for path in (checkpoint, base):
        if not path.is_file():
            raise FileNotFoundError(path)
    cmd = [sys.executable, "run_paired_diagnostics.py",
           "--base-config", str(base), "--checkpoint", str(checkpoint),
           "--output", str(output), "--scene-token", task["scene_token"],
           "--seed", str(task["seed"]), "--windows", str(task["windows"]),
           "--mode", mode, "--engine", "teacher",
           "--history-mode", "clean", "--noise-sigma", "0",
           "--condition-mask", "full", "--save-review"]
    env = os.environ.copy()
    env["PANDWM_PANO_BLEND_W"] = "128"
    print(json.dumps({"task": task, "command": cmd}), flush=True)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
