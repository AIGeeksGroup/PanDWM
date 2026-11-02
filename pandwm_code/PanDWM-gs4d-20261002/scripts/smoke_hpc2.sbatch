#!/usr/bin/env bash
#SBATCH --job-name=gs4d_smoke
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
# 20 training steps (all 4D losses, backward, checkpoint) + a 5-clip eval of
# the G2-AA baseline, to validate code and memory before a long run.
set -euo pipefail
source $(dirname "$0")/common_env.sh 2>/dev/null || source $HOME/pandwm_code/PanDWM-gs4d-20261002/scripts/common_env.sh
OUT=$RUNS/smoke_${SLURM_JOB_ID}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python $GSAA/src/dwm_pano/train.py -c $GS4D/configs/gs4d_hpc2_smoke.json -o $OUT \
  --max-steps ${STEPS:-20} --log-steps 1 --preview-steps 100000000 \
  --evaluation-steps 100000000 --checkpointing-steps ${STEPS:-20}
ls -la $OUT/checkpoints
python -m dwm_pano_4d.evaluate_4d -c $GS4D/configs/gs4d_hpc2_smoke.json \
  --checkpoint $H/pandwm_gs_aa_experiment/runs/g2_mix75_250_aa_eps0to0p1/checkpoints/250.pth \
  --data-pkl $H/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl \
  --clips 5 --output $OUT/eval_g2_baseline_5clips.json
echo GS4D_SMOKE_DONE
