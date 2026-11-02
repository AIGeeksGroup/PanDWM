#!/bin/bash
#SBATCH --job-name=omni_score3
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
SEEDS=(0 42 1000)
OMNI_MANIFEST=$H/omninwm_runs/common149_adapter/manifest.json
COMMON_MANIFEST=$H/vista_common_front149/manifest.json
I3D=$H/pandwm_code/TATS_fvd/i3d_pretrained_400.pt

test -f "$I3D"
test -f "$OMNI_MANIFEST"
test -f "$COMMON_MANIFEST"

export PYTHONUNBUFFERED=1
export PYTHONPATH=$P/src:$H/pandwm_code/TATS_fvd
cd "$P"
for SEED in "${SEEDS[@]}"; do
  RUN=$H/omninwm_runs/common149/seed_${SEED}
  DUMP=$H/omninwm_runs/common149_dump/seed_${SEED}
  OUT=$DUMP/common_front_seed${SEED}.json

  for SHARD in $(seq 0 14); do
    test -d "$RUN/shard_${SHARD}/lossless"
  done

  if [ ! -d "$DUMP" ]; then
    "$H/pandwm_venv/bin/python" scripts/omninwm/export_common_dump.py \
      --run-root "$RUN" \
      --omni-manifest "$OMNI_MANIFEST" \
      --common-manifest "$COMMON_MANIFEST" \
      --seed "$SEED" \
      --out "$DUMP"
  fi

  test "$(find "$DUMP/real" -maxdepth 1 -type f -name 'clip_*.pt' | wc -l)" -eq 149
  test "$(find "$DUMP/fake" -maxdepth 1 -type f -name 'clip_*.pt' | wc -l)" -eq 149
  if [ ! -f "$OUT" ]; then
    "$H/pandwm_venv/bin/python" scripts/horizondrive/evaluate_common_front.py \
      --dump "$DUMP" \
      --i3d "$I3D" \
      --condition-frames 0 \
      --frames 16 \
      --expected-clips 149 \
      --generation-seed "$SEED" \
      --out "$OUT"
  fi
  cat "$OUT"
done
