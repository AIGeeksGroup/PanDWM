#!/usr/bin/env bash
#SBATCH --job-name=pandwm_gt_box_cal
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail
DATA_ROOT=/path/to/pandwm_root
CODE_ROOT=$DATA_ROOT/pandwm_code/PanDWM_long_horizon_20260928
RUN_ROOT=$DATA_ROOT/pandwm_runs/long_horizon_20260928
SCENE_LIMIT=${SCENE_LIMIT:-8}
RUN_TAG=${RUN_TAG:-gt_box_calibration_v1}
source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$CODE_ROOT/src:$DATA_ROOT/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_HOME=$DATA_ROOT/.cache/torch
unset HF_AUTO_UPLOAD_REPO || true
cd "$CODE_ROOT"
"$DATA_ROOT/pandwm_venv/bin/python" calibrate_gt_box_detector.py \
  --base-config "$DATA_ROOT/pandwm_code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_hpc3.json" \
  --scene-manifest "$CODE_ROOT/inputs/split_195_scene_manifest.json" \
  --output "$RUN_ROOT/$RUN_TAG" --scene-limit "$SCENE_LIMIT"
