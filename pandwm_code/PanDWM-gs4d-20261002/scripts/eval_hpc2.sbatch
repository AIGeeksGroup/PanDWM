#!/usr/bin/env bash
#SBATCH --job-name=gs4d_eval
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
# CHECKPOINT=<raw decoder state dict>; NAME=<report name>; CLIPS (default 50).
set -euo pipefail
source $HOME/pandwm_code/PanDWM-gs4d-20261002/scripts/common_env.sh
python -m dwm_pano_4d.evaluate_4d -c $GS4D/configs/gs4d_hpc2.json \
  --checkpoint ${CHECKPOINT:?} \
  --data-pkl $H/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl \
  --clips ${CLIPS:-50} --output $RUNS/eval/${NAME:?}.json
