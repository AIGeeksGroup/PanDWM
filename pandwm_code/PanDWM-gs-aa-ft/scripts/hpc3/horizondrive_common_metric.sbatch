#!/bin/bash
#SBATCH --job-name=hdrive_metric
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

HORIZON_JOB=${HORIZON_JOB:?export the completed HorizonDrive common job id}
HORIZON_SEED=${HORIZON_SEED:?export the corresponding generation seed}
H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
DUMP=$H/horizondrive_runs/common/job_${HORIZON_JOB}/lossless
I3D=$H/pandwm_code/TATS_fvd/i3d_pretrained_400.pt
OUT=$H/horizondrive_runs/common/job_${HORIZON_JOB}/common_front_fvd.json

export PYTHONPATH=$P/src:$H/pandwm_code/TATS_fvd
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
source /share/anaconda3/bin/activate pytorch271

"$H/pandwm_venv/bin/python" \
  "$P/scripts/horizondrive/evaluate_common_front.py" \
  --dump "$DUMP" \
  --i3d "$I3D" \
  --condition-frames 3 \
  --frames 32 \
  --expected-clips 149 \
  --generation-seed "$HORIZON_SEED" \
  --device cuda \
  --out "$OUT"
