#!/usr/bin/env bash
#SBATCH --job-name=pandwm_teacher
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%A_%a.log
set -euo pipefail

DATA_ROOT=/path/to/pandwm_root
CODE_ROOT=$DATA_ROOT/pandwm_code/PanDWM_long_horizon_20260928
TASKS=${TASKS:?set teacher task JSON filename}
RUN_TAG=${RUN_TAG:?set unique run tag}
TASK_INDEX=${SLURM_ARRAY_TASK_ID:-0}
source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$CODE_ROOT/src:$DATA_ROOT/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((32000 + SLURM_JOB_ID % 10000))
unset HF_AUTO_UPLOAD_REPO || true
cd "$CODE_ROOT"
"$DATA_ROOT/pandwm_venv/bin/python" - "$TASKS" "$TASK_INDEX" "$RUN_TAG" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

from run_eval_task import task_directory_name

tasks_file, index, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
task = json.loads(Path(tasks_file).read_text())["tasks"][index]
assert task["arm"] in ("teacher_cfg3", "teacher_nocfg")
data = Path("/path/to/pandwm_root")
checkpoint = data / "pandwm_data/stage1_causal_20000.pth"
base = data / "pandwm_code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_hpc3.json"
output = data / "pandwm_runs/long_horizon_20260928" / tag / task_directory_name(task)
for path in (checkpoint, base):
    if not path.is_file():
        raise FileNotFoundError(path)
if output.exists():
    raise FileExistsError(output)
cmd = [sys.executable, "run_teacher_rollout.py", "--base-config", str(base),
       "--checkpoint", str(checkpoint), "--output", str(output),
       "--scene-token", task["scene_token"], "--seed", str(task["seed"]),
       "--windows", str(task["windows"]), "--history-mode", "clean",
       "--engine", "teacher", "--teacher-guidance-scale",
       "3" if task["arm"] == "teacher_cfg3" else "0"]
if task.get("oracle_window"):
    cmd.extend(["--oracle-window", str(task["oracle_window"])])
print(json.dumps({"task": task, "command": cmd}), flush=True)
subprocess.run(cmd, check=True)
PY
