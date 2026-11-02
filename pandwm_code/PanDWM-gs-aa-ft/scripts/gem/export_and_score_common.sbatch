#!/bin/bash
#SBATCH --job-name=gem_score
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.log
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
SEEDS=(0 42 1000)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
RUN=$H/gem_runs/common149/seed_${SEED}
DUMP=$H/gem_runs/common149_dump/seed_${SEED}
COMMON=$H/vista_common_front149/manifest.json
I3D=$H/pandwm_code/TATS_fvd/i3d_pretrained_400.pt
OUT=$DUMP/common_front_seed${SEED}.json

test -f "$I3D"
test ! -e "$DUMP"
for SHARD in 0 1 2 3 4; do
  test -f "$RUN/shard_${SHARD}/input_order.json"
done

export PYTHONUNBUFFERED=1
export PYTHONPATH=$P/src:$H/pandwm_code/TATS_fvd
cd "$P"
"$H/pandwm_venv/bin/python" scripts/gem/export_common_dump.py \
  --run-root "$RUN" \
  --common-manifest "$COMMON" \
  --out "$DUMP" \
  --seed "$SEED"

test "$(find "$DUMP/real" -maxdepth 1 -type f -name 'clip_*.pt' | wc -l)" -eq 149
test "$(find "$DUMP/fake" -maxdepth 1 -type f -name 'clip_*.pt' | wc -l)" -eq 149
"$H/pandwm_venv/bin/python" scripts/horizondrive/evaluate_common_front.py \
  --dump "$DUMP" \
  --i3d "$I3D" \
  --condition-frames 1 \
  --frames 16 \
  --expected-clips 149 \
  --generation-seed "$SEED" \
  --out "$OUT"
cat "$OUT"
