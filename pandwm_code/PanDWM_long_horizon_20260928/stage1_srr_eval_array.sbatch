#!/usr/bin/env bash
#SBATCH --job-name=pandwm_srr_eval
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%A_%a.log
set -euo pipefail
DATA_ROOT=/path/to/pandwm_root
CODE_ROOT=$DATA_ROOT/pandwm_code/PanDWM_long_horizon_20260928
TASKS=${TASKS:-tasks_srr_teacher_dev195.json}
RUN_TAG=${RUN_TAG:?set RUN_TAG}
TRAIN_TAG=${TRAIN_TAG:?set TRAIN_TAG}
STEPS=${STEPS:?set STEPS}
# The account allows 50 submitted jobs (array elements count), so one
# element may run several tasks: index = OFFSET + array_id + slot * STRIDE.
TASKS_PER_SLOT=${TASKS_PER_SLOT:-1}
TASK_INDEX_STRIDE=${TASK_INDEX_STRIDE:-0}
TASK_OFFSET=${TASK_OFFSET:-0}
source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$CODE_ROOT/vendor:$CODE_ROOT/src:$DATA_ROOT/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
# A fixed JOBID-derived port collided (EADDRINUSE) on a shared node;
# ask the kernel for a free one instead.
export MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
export TORCH_HOME=$DATA_ROOT/.cache/torch
unset HF_AUTO_UPLOAD_REPO || true
cd "$CODE_ROOT"
for ((slot=0; slot<TASKS_PER_SLOT; slot++)); do
  TASK_INDEX=$((TASK_OFFSET + ${SLURM_ARRAY_TASK_ID:-0} + slot * TASK_INDEX_STRIDE))
  "$DATA_ROOT/pandwm_venv/bin/python" run_stage1_srr_eval_task.py \
    --tasks "$TASKS" --index "$TASK_INDEX" --run-tag "$RUN_TAG" \
    --train-tag "$TRAIN_TAG" --steps "$STEPS" --data-root "$DATA_ROOT"
done
