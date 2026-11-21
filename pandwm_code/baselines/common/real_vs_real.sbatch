#!/bin/bash
#SBATCH --job-name=real_vs_real
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
# Real-vs-real FID/FVD reference on random half splits of the 149 common clips.
# Usage: REAL=.. OUT=.. OFFSET=.. FRAMES=.. [SPLIT_REPEATS=20] [SPLIT_SEED=20260909] sbatch real_vs_real.sbatch
#   Epona / 16-frame protocol: REAL=$R/horizondrive_runs/common/job_<id>/lossless/real OFFSET=10 FRAMES=16 SPLIT_SEED=20260910
#                 OUT=$R/external_runs/epona/common_target_10ctx_16pred/real_vs_real.json
#   HorizonDrive: REAL=$R/horizondrive_runs/common/job_<id>/lossless/real OFFSET=3 FRAMES=32
#                 OUT=$R/horizondrive_runs/common/job_<id>/real_vs_real.json
#   (R=/path/to/pandwm_root)
set -euo pipefail

REAL=${REAL:?export REAL}
OUT=${OUT:?export OUT}
OFFSET=${OFFSET:?export OFFSET}
FRAMES=${FRAMES:?export FRAMES}
SPLIT_REPEATS=${SPLIT_REPEATS:-20}
SPLIT_SEED=${SPLIT_SEED:-20260909}
H=/path/to/pandwm_root
P=$H/pandwm_code/baselines
I3D=$H/pandwm_code/TATS_fvd/i3d_pretrained_400.pt

test -d "$REAL"
test -f "$I3D"
mkdir -p "$(dirname "$OUT")"
export PYTHONPATH=$H/pandwm_code/src:$H/pandwm_code/TATS_fvd
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
source /path/to/conda/bin/activate pandwm

"$H/pandwm_venv/bin/python" \
  "$P/common/evaluate_real_vs_real.py" \
  --real-dir "$REAL" \
  --i3d "$I3D" \
  --condition-frames "$OFFSET" \
  --frames "$FRAMES" \
  --expected-clips 149 \
  --split-size 74 \
  --repeats "$SPLIT_REPEATS" \
  --split-seed "$SPLIT_SEED" \
  --device cuda \
  --out "$OUT"
