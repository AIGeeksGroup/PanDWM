#!/bin/bash
#SBATCH --job-name=hdrive_r2r
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail

SOURCE_JOB=${SOURCE_JOB:?set SOURCE_JOB to the generation job directory}
SPLIT_REPEATS=${SPLIT_REPEATS:-20}
SPLIT_SEED=${SPLIT_SEED:-20260909}
H=/path/to/pandwm_root
P=$H/pandwm_code/gaussian/gs3d
REAL=$H/horizondrive_runs/common/job_${SOURCE_JOB}/lossless/real
I3D=$H/pandwm_code/TATS_fvd/i3d_pretrained_400.pt
OUT=$H/horizondrive_runs/common/job_${SOURCE_JOB}/real_vs_real.json

test -d "$REAL"
test -f "$I3D"
export PYTHONPATH=$H/pandwm_code/src:$H/pandwm_code/TATS_fvd
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
source /path/to/conda/bin/activate pandwm

"$H/pandwm_venv/bin/python" \
  "$P/scripts/horizondrive/evaluate_real_vs_real.py" \
  --real-dir "$REAL" \
  --i3d "$I3D" \
  --condition-frames 3 \
  --frames 32 \
  --expected-clips 149 \
  --split-size 74 \
  --repeats "$SPLIT_REPEATS" \
  --split-seed "$SPLIT_SEED" \
  --device cuda \
  --out "$OUT"
