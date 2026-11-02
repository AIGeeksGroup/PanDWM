#!/bin/bash
#SBATCH --job-name=gem_common
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --array=0-14
#SBATCH --output=%x_%A_%a.log
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
SOURCE_GEM=$H/external_baselines/GEM
COMMON=$H/vista_common_front149
WEIGHT=$H/external_models/gem/gem.safetensors
EXPECTED_COMMIT=f8dae58270dca14d2646d4833391856fefd5e488
EXPECTED_WEIGHT_SHA=0c243cddd1bad88cfd20ce367724b15266692ecbf37e5effbfb859493ff04c0d

SEEDS=(0 42 1000)
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
SHARD=$((SLURM_ARRAY_TASK_ID % 5))
SEED=${SEEDS[$SEED_INDEX]}
START=$((SHARD * 30))
END=$((START + 30))
if [ "$END" -gt 149 ]; then END=149; fi

test "$(sed -n '1p' "$SOURCE_GEM/.git/refs/heads/main")" = "$EXPECTED_COMMIT"
test "$(stat -c %s "$WEIGHT")" -eq 15589259920
test -f "$COMMON/manifest.json"

WORK=/tmp/<user>_gem_${SLURM_JOB_ID}_${SEED}_${SHARD}
test ! -e "$WORK"
cp -a "$SOURCE_GEM" "$WORK"
test "$WORK" = "/tmp/<user>_gem_${SLURM_JOB_ID}_${SEED}_${SHARD}"
cleanup_workdir() {
  if [ -d "$WORK" ]; then
    rm -r "$WORK"
  fi
}
trap cleanup_workdir EXIT
if [ -e "$WORK/datasets/zero-samples" ]; then
  rm -r "$WORK/datasets/zero-samples"
fi
mkdir -p "$WORK/checkpoints"
ln -sfn "$WEIGHT" "$WORK/checkpoints/gem_new_stage2.safetensors"

OUT=$H/gem_runs/common149/seed_${SEED}/shard_${SHARD}
test ! -e "$OUT"
mkdir -p "$OUT"
"$H/vista_venv/bin/python" "$P/scripts/gem/prepare_common_shard.py" \
  --common-root "$COMMON" \
  --input-dir "$WORK/datasets/zero-samples" \
  --manifest "$OUT/input_order.json" \
  --start "$START" \
  --end "$END"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=$H/external_models/hf_cache
export TORCH_HOME=$H/.cache/torch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$P/scripts/gem/optional_stubs:$WORK

echo "official_commit=$EXPECTED_COMMIT official_weight_sha256=$EXPECTED_WEIGHT_SHA"
echo "protocol=common149 seed=$SEED shard=$SHARD range=[$START,$END) frames=25 steps=50"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
cd "$WORK"
"$H/vista_venv/bin/python" sample.py \
  --dataset ZeroShot \
  --save "$OUT" \
  --condition_type unconditional \
  --n_frames 25 \
  --n_steps 50 \
  --seed "$SEED" \
  --rand_gen

EXPECTED=$((END - START))
ACTUAL=$(find "$OUT" -type f -path '*/virtual/npy/*.npy' | wc -l)
test "$ACTUAL" -eq "$EXPECTED"
echo "gem_common_shard_completed seed=$SEED shard=$SHARD clips=$ACTUAL"
