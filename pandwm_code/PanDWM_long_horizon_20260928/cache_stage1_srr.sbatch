#!/usr/bin/env bash
#SBATCH --job-name=pandwm_srr_cache
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%A_%a.log
set -euo pipefail
DATA_ROOT=/path/to/pandwm_root
CODE_ROOT=$DATA_ROOT/pandwm_code/PanDWM_long_horizon_20260928
RUN_ROOT=$DATA_ROOT/pandwm_runs/long_horizon_20260928
SHARD=${SLURM_ARRAY_TASK_ID:-0}
SHARDS=${SHARDS:-4}
MAX_ITEMS=${MAX_ITEMS:-0}
MAX_TARGET_WINDOW=${MAX_TARGET_WINDOW:-1}
CACHE_TAG=${CACHE_TAG:-srr_cache_train67_v1}
CACHE_CONFIG=${CACHE_CONFIG:-$RUN_ROOT/srr_config/recovery.json}
source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$CODE_ROOT/src:$DATA_ROOT/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((31000 + SLURM_JOB_ID % 10000))
unset HF_AUTO_UPLOAD_REPO || true
cd "$CODE_ROOT"
"$DATA_ROOT/pandwm_venv/bin/python" cache_stage1_srr_history.py \
  --config "$CACHE_CONFIG" \
  --scene-manifest "$CODE_ROOT/inputs/split_195_scene_manifest.json" \
  --checkpoint "$DATA_ROOT/pandwm_data/stage1_causal_20000.pth" \
  --cache-dir "$RUN_ROOT/$CACHE_TAG" \
  --shard "$SHARD" --shards "$SHARDS" \
  --max-items "$MAX_ITEMS" --max-target-window "$MAX_TARGET_WINDOW"
