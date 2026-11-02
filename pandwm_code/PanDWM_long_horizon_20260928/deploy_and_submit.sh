#!/usr/bin/env bash
# Deploy this isolated experiment and submit one explicitly selected HPC phase.
set -euo pipefail

PHASE=${1:?usage: deploy_and_submit.sh baseline|diagnostic|diagnostic_validate|noise_smoke|noise_train|noise_eval}
case "$PHASE" in baseline|diagnostic|diagnostic_validate|noise_smoke|noise_train|noise_eval) ;; *) exit 2 ;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HPC_HOST=${HPC_HOST:-hpc}
LOCAL_BIND_ADDR=${LOCAL_BIND_ADDR:-10.13.138.69}
REMOTE=/path/to/pandwm_root/pandwm_code/PanDWM_long_horizon_20260928
SSH_OPTS=(-b "$LOCAL_BIND_ADDR" -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes)

ssh "${SSH_OPTS[@]}" "$HPC_HOST" "mkdir -p '$REMOTE'"
rsync -az --exclude='results/' --exclude='__pycache__/' --exclude='*.pyc' \
  -e "ssh -b $LOCAL_BIND_ADDR -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes" \
  "$HERE/" "$HPC_HOST:$REMOTE/"

case "$PHASE" in
  baseline)
    ssh "${SSH_OPTS[@]}" "$HPC_HOST" \
      "cd '$REMOTE' && sbatch --array=0-39%4 --export=ALL,TASKS=tasks_baseline_dev195.json,RUN_TAG=baseline_dev195_v1 eval_array.sbatch"
    ;;
  diagnostic)
    ssh "${SSH_OPTS[@]}" "$HPC_HOST" \
      "cd '$REMOTE' && sbatch --array=0-5%4 --time=04:30:00 --export=ALL,TASKS=tasks_diagnostic_dev195.json,RUN_TAG=diagnostic_dev195_v1,TASKS_PER_SLOT=2,TASK_INDEX_STRIDE=6 eval_array.sbatch"
    ;;
  diagnostic_validate)
    ssh "${SSH_OPTS[@]}" "$HPC_HOST" \
      "cd '$REMOTE' && sbatch --array=0-23%4 --export=ALL,TASKS=tasks_diagnostic_validate_dev195.json,RUN_TAG=diagnostic_validate_dev195_v1 eval_array.sbatch"
    ;;
  noise_smoke)
    ssh "${SSH_OPTS[@]}" "$HPC_HOST" \
      "cd '$REMOTE' && sbatch --export=ALL,ARM=control,STEPS=2,RUN_TAG=noise_smoke train_noise_pilot.sbatch && sbatch --export=ALL,ARM=noise,STEPS=2,RUN_TAG=noise_smoke train_noise_pilot.sbatch"
    ;;
  noise_train)
    ssh "${SSH_OPTS[@]}" "$HPC_HOST" \
      "cd '$REMOTE' && sbatch --export=ALL,ARM=control,RUN_TAG=noise_pilot train_noise_pilot.sbatch && sbatch --export=ALL,ARM=noise,RUN_TAG=noise_pilot train_noise_pilot.sbatch"
    ;;
  noise_eval)
    # Submit one half at a time: this cluster limits the number of queued
    # array elements per user, including tasks waiting for a GPU.
    TASK_RANGE=${2:-0-31}
    case "$TASK_RANGE" in 0-31|32-63) ;; *) echo "noise_eval range must be 0-31 or 32-63" >&2; exit 2 ;; esac
    ssh "${SSH_OPTS[@]}" "$HPC_HOST" \
      "cd '$REMOTE' && sbatch --array=$TASK_RANGE%4 --export=ALL,TASKS=tasks_noise_pilot_dev195.json,RUN_TAG=noise_pilot_dev195_v1 eval_array.sbatch"
    ;;
esac
