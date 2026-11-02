#!/bin/bash
#SBATCH --job-name=epona_metric
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=/path/to/pandwm_root/external_runs/epona/logs/%x_%j.log
set -euo pipefail

SEED=${SEED:?export SEED}
H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
DUMP=$H/external_runs/epona/common_target_10ctx_16pred/seed_${SEED}/lossless
OUT=$H/external_runs/epona/common_target_10ctx_16pred/seed_${SEED}/fid_fvd.json
I3D=$H/pandwm_code/TATS_fvd/i3d_pretrained_400.pt

test -d "$DUMP/real"
test -d "$DUMP/fake"
test -f "$I3D"
test "$(find "$DUMP/fake" -maxdepth 1 -name '*.pt' | wc -l)" -eq 149

source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$P/src:$H/pandwm_code/TATS_fvd
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
"$H/pandwm_venv/bin/python" \
  "$P/scripts/horizondrive/evaluate_common_front.py" \
  --dump "$DUMP" \
  --i3d "$I3D" \
  --condition-frames 10 \
  --frames 16 \
  --expected-clips 149 \
  --generation-seed "$SEED" \
  --device cuda \
  --out "$OUT"
