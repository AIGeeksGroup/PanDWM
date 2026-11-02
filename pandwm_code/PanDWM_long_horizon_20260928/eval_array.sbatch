#!/usr/bin/env bash
#SBATCH --job-name=pandwm_195dev
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%A_%a.log
set -euo pipefail

DATA_ROOT=/path/to/pandwm_root
CODE_ROOT=$DATA_ROOT/pandwm_code/PanDWM_long_horizon_20260928
TASKS=${TASKS:?set TASKS to a JSON task list in CODE_ROOT}
RUN_TAG=${RUN_TAG:?set a unique RUN_TAG}
TASKS_PER_SLOT=${TASKS_PER_SLOT:-1}
TASK_INDEX_STRIDE=${TASK_INDEX_STRIDE:-0}
test -f "$CODE_ROOT/$TASKS"
source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$CODE_ROOT/src:$DATA_ROOT/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((31000 + SLURM_JOB_ID % 10000))
export TORCH_HOME=$DATA_ROOT/.cache/torch
unset HF_AUTO_UPLOAD_REPO || true
cd "$CODE_ROOT"
for ((slot=0; slot<TASKS_PER_SLOT; slot++)); do
  task_index=$((SLURM_ARRAY_TASK_ID + slot * TASK_INDEX_STRIDE))
  "$DATA_ROOT/pandwm_venv/bin/python" run_eval_task.py \
    --tasks "$CODE_ROOT/$TASKS" --index "$task_index" \
    --data-root "$DATA_ROOT" --run-tag "$RUN_TAG"
done
