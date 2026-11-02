"""Launch a frozen paired-diagnostic task without shell interpolation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def task_name(task):
    if task["mode"] == "ladder":
        return f"ladder_scene{task['scene_index']}_seed{task['seed']}"
    return (f"condition_{task['history_mode']}_{task['condition_mask']}"
            f"_scene{task['scene_index']}_seed{task['seed']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(args.tasks.read_text())["tasks"][args.index]
    output = args.data_root / "pandwm_runs/long_horizon_20260928" / \
        args.run_tag / task_name(task)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    base = (args.data_root /
            "pandwm_code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_hpc3.json")
    checkpoint = args.data_root / "pandwm_data/cd_10000_ema.pth"
    for path in (base, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    cmd = [sys.executable, "run_paired_diagnostics.py",
           "--base-config", str(base), "--checkpoint", str(checkpoint),
           "--output", str(output), "--scene-token", task["scene_token"],
           "--seed", str(task["seed"]), "--mode", task["mode"],
           "--windows", str(task["windows"]),
           "--history-mode", task["history_mode"],
           "--noise-sigma", str(task["noise_sigma"]),
           "--condition-mask", task["condition_mask"], "--save-review"]
    env = os.environ.copy()
    env["PANDWM_4STEP_SAMPLER"] = "stochastic"
    env["PANDWM_PANO_BLEND_W"] = "128"
    print(json.dumps({"task": task, "command": cmd}), flush=True)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
