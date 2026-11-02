"""Execute one frozen Slurm array task without shell interpolation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def task_directory_name(task):
    mode = task["history_mode"]
    variant = mode
    if mode == "flow_noise":
        variant += f"_{task['noise_sigma']:.3f}"
    if mode == "moment_align":
        variant += f"_{task['align_alpha']:.3f}"
    return f"{task['arm']}_{variant}_scene{task['scene_index']}_seed{task['seed']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    payload = json.loads(args.tasks.read_text())
    task = payload["tasks"][args.index]
    arm = task["arm"]
    h = args.data_root
    if arm == "baseline":
        checkpoint = h / "pandwm_data/cd_10000_ema.pth"
    elif arm in ("control", "self"):
        checkpoint = (h / "pandwm_runs/self_forcing_stage2_20260915/production" /
                      arm / "checkpoints/8000_ema.pth")
    elif arm in ("noise_control", "noise"):
        checkpoint = (h / "pandwm_runs/long_horizon_20260928/noise_pilot" /
                      ("control" if arm == "noise_control" else "noise") /
                      "checkpoints/250_ema.pth")
    else:
        raise ValueError(f"unknown arm {arm}")
    base = h / "pandwm_code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_hpc3.json"
    for required in (base, checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)
    output = (h / "pandwm_runs/long_horizon_20260928" / args.run_tag /
              task_directory_name(task))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cmd = [sys.executable, str(Path(__file__).with_name("run_long_rollout.py")),
           "--base-config", str(base), "--checkpoint", str(checkpoint),
           "--output", str(output), "--scene-token", task["scene_token"],
           "--seed", str(task["seed"]), "--windows", str(task["windows"]),
           "--history-mode", task["history_mode"],
           "--noise-sigma", str(task["noise_sigma"]),
           "--align-alpha", str(task["align_alpha"])]
    print(json.dumps({"task": task, "checkpoint": str(checkpoint),
                      "output": str(output), "command": cmd}), flush=True)
    env = os.environ.copy()
    env["PANDWM_4STEP_SAMPLER"] = "stochastic"
    env["PANDWM_PANO_BLEND_W"] = "128"
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
