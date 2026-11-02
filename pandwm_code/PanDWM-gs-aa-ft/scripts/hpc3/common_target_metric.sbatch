#!/bin/bash
#SBATCH --job-name=common_tgt_metric
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=/path/to/pandwm_root/external_runs/epona/logs/%x_%j.log
set -euo pipefail

METHOD=${METHOD:?export METHOD}
SEED=${SEED:?export SEED}
DUMP=${DUMP:?export DUMP}
OFFSET=${OFFSET:?export OFFSET}
OUT=${OUT:?export OUT}
H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
I3D=$H/pandwm_code/TATS_fvd/i3d_pretrained_400.pt

test -d "$DUMP/real"
test -d "$DUMP/fake"
test -f "$I3D"
test "$(find "$DUMP/real" -maxdepth 1 -name '*.pt' | wc -l)" -eq 149
test "$(find "$DUMP/fake" -maxdepth 1 -name '*.pt' | wc -l)" -eq 149
mkdir -p "$(dirname "$OUT")"

source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$P/src:$H/pandwm_code/TATS_fvd
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
echo "method=$METHOD seed=$SEED dump=$DUMP offset=$OFFSET"
"$H/pandwm_venv/bin/python" \
  "$P/scripts/horizondrive/evaluate_common_front.py" \
  --dump "$DUMP" \
  --i3d "$I3D" \
  --condition-frames "$OFFSET" \
  --frames 16 \
  --expected-clips 149 \
  --generation-seed "$SEED" \
  --device cuda \
  --out "$OUT"
