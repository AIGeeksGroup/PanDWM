"""Render one 195-frame closed-loop video (3 GT frames + 12 generated windows).

Thin wrapper over run_long_rollout.py so that trained SRR checkpoints can be
rendered with the same protocol as the paired eval (teacher: 50 steps, no CFG).
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

RUNS = "pandwm_runs/long_horizon_20260928"
V2B = f"{RUNS}/srr_teacher_w123_r05_lr1e5_2000_v2b"
ARMS = {
    "b0_student": ("pandwm_data/cd_10000_ema.pth", "student"),
    "teacher_init": ("pandwm_data/stage1_causal_20000.pth", "teacher"),
    "v2b_control": (f"{V2B}/control/checkpoints/2000.pth", "teacher"),
    "v2b_recovery": (f"{V2B}/recovery/checkpoints/2000.pth", "teacher"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    task = json.loads(args.tasks.read_text())["tasks"][args.index]
    relative, engine = ARMS[task["arm"]]
    checkpoint = args.data_root / relative
    base = args.data_root / "pandwm_code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_hpc3.json"
    for path in (checkpoint, base):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = (args.data_root / RUNS / args.run_tag /
              f"{task['arm']}_scene{task['scene_index']}_seed{task['seed']}")
    if output.exists():
        raise FileExistsError(output)
    cmd = [sys.executable, str(Path(__file__).with_name("run_long_rollout.py")),
           "--base-config", str(base), "--checkpoint", str(checkpoint),
           "--output", str(output), "--scene-token", task["scene_token"],
           "--seed", str(task["seed"]), "--windows", "12",
           "--history-mode", "clean", "--engine", engine]
    if engine == "teacher":
        cmd += ["--teacher-guidance-scale", "0"]
    print(json.dumps({"task": task, "command": cmd}), flush=True)
    env = os.environ.copy()
    env["PANDWM_4STEP_SAMPLER"] = "stochastic"
    env["PANDWM_PANO_BLEND_W"] = "128"
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
