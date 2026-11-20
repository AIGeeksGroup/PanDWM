#!/bin/bash
#SBATCH --job-name=common_tgt_metric
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/pandwm_root/external_runs/epona/logs/%x_%j.log
# FID/FVD for one generated common-clip dump (real/ and fake/ tensors) using the I3D detector.
# Usage: METHOD=.. SEED=.. DUMP=.. OFFSET=.. [FRAMES=16] OUT=.. sbatch common_target_metric.sbatch
#   Epona:        METHOD=epona SEED=0 OFFSET=10 DUMP=$R/external_runs/epona/common_target_10ctx_16pred/seed_0/lossless
#                 OUT=$R/external_runs/epona/common_target_10ctx_16pred/seed_0/fid_fvd.json
#   HorizonDrive: METHOD=horizondrive SEED=<seed> OFFSET=3 FRAMES=32 DUMP=$R/horizondrive_runs/common/job_<id>/lossless
#                 OUT=$R/horizondrive_runs/common/job_<id>/common_front_fvd.json
#   (R=/path/to/pandwm_root)
set -euo pipefail

METHOD=${METHOD:?export METHOD}
SEED=${SEED:?export SEED}
DUMP=${DUMP:?export DUMP}
OFFSET=${OFFSET:?export OFFSET}
OUT=${OUT:?export OUT}
H=/path/to/pandwm_root
P=$H/pandwm_code/baselines
I3D=$H/pandwm_code/TATS_fvd/i3d_pretrained_400.pt

test -d "$DUMP/real"
test -d "$DUMP/fake"
test -f "$I3D"
test "$(find "$DUMP/real" -maxdepth 1 -name '*.pt' | wc -l)" -eq 149
test "$(find "$DUMP/fake" -maxdepth 1 -name '*.pt' | wc -l)" -eq 149
mkdir -p "$(dirname "$OUT")"

source /path/to/conda/bin/activate pandwm
export PYTHONPATH=$H/pandwm_code/src:$H/pandwm_code/TATS_fvd
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
echo "method=$METHOD seed=$SEED dump=$DUMP offset=$OFFSET"
"$H/pandwm_venv/bin/python" \
  "$P/horizondrive/evaluate_common_front.py" \
  --dump "$DUMP" \
  --i3d "$I3D" \
  --condition-frames "$OFFSET" \
  --frames "${FRAMES:-16}" \
  --expected-clips 149 \
  --generation-seed "$SEED" \
  --device cuda \
  --out "$OUT"
