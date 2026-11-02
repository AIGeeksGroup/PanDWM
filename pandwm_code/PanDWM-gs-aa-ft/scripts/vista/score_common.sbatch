#!/bin/bash
#SBATCH --job-name=vista_score
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

SEED=${SEED:?SEED is required}
H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
DUMP=$H/vista_runs/common_seed${SEED}
I3D=$H/pandwm_code/TATS_fvd/i3d_pretrained_400.pt
OUT=$DUMP/common_front_seed${SEED}.json

test -f "$I3D"
test "$(find "$DUMP/real" -maxdepth 1 -type f -name 'clip_*.pt' | wc -l)" -eq 149
test "$(find "$DUMP/fake" -maxdepth 1 -type f -name 'clip_*.pt' | wc -l)" -eq 149
test ! -e "$OUT" || { echo "refusing to overwrite $OUT" >&2; exit 2; }

export PYTHONUNBUFFERED=1
export PYTHONPATH=$P/src:$H/pandwm_code/TATS_fvd
cd "$P"
$H/pandwm_venv/bin/python scripts/horizondrive/evaluate_common_front.py \
  --dump "$DUMP" \
  --i3d "$I3D" \
  --condition-frames 1 \
  --frames 16 \
  --expected-clips 149 \
  --generation-seed "$SEED" \
  --out "$OUT"
cat "$OUT"
