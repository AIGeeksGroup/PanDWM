#!/usr/bin/env bash
#SBATCH --job-name=gs4d_train
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.log
# Long run. RUN_TAG names the run dir; RESUME_FROM=<step> continues it.
set -euo pipefail
source $HOME/pandwm_code/PanDWM-gs4d-20261002/scripts/common_env.sh
OUT=$RUNS/${RUN_TAG:-train_v1}
EXTRA=()
[[ -n "${RESUME_FROM:-}" ]] && EXTRA=(--resume-from $RESUME_FROM)
python $GSAA/src/dwm_pano/train.py -c $GS4D/configs/${CONFIG:-gs4d_hpc2.json} -o $OUT \
  --max-steps ${STEPS:-15000} --log-steps 20 --preview-steps 100000000 \
  --evaluation-steps 100000000 --checkpointing-steps ${CKPT_STEPS:-1000} \
  --keep-last-ckpts 5 "${EXTRA[@]}"
