#!/bin/bash
#SBATCH --job-name=gem_official_smoke
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
GEM=$H/external_baselines/GEM
WEIGHT=$H/external_models/gem/gem.safetensors
EXPECTED_COMMIT=f8dae58270dca14d2646d4833391856fefd5e488
EXPECTED_WEIGHT_SHA=0c243cddd1bad88cfd20ce367724b15266692ecbf37e5effbfb859493ff04c0d
DINO=$H/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth
RAFT=$H/.cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth
EXPECTED_DINO_SHA=0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
EXPECTED_RAFT_SHA=ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322
INPUT=$H/vista_common_front149/clip_0000/00.jpg

# The shared checkout is audited clean before submission. Some compute nodes
# omit git from PATH, so verify the pinned ref without invoking git in the job.
test "$(sed -n '1p' "$GEM/.git/HEAD")" = "ref: refs/heads/main"
test "$(sed -n '1p' "$GEM/.git/refs/heads/main")" = "$EXPECTED_COMMIT"
test -d "$H/.cache/torch/hub/facebookresearch_dinov2_main"
test -f "$INPUT"
echo "$EXPECTED_WEIGHT_SHA  $WEIGHT" | sha256sum --check --strict
echo "$EXPECTED_DINO_SHA  $DINO" | sha256sum --check --strict
echo "$EXPECTED_RAFT_SHA  $RAFT" | sha256sum --check --strict

# Upstream sample.py uses fixed relative asset paths. These are untracked links;
# no line in the official repository is modified.
mkdir -p "$GEM/checkpoints" "$GEM/datasets/zero-samples"
ln -sfn "$WEIGHT" "$GEM/checkpoints/gem_new_stage2.safetensors"
ln -sfn "$INPUT" "$GEM/datasets/zero-samples/input.jpg"
OUT=$H/gem_runs/official_smoke
mkdir -p "$OUT"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=$H/external_models/hf_cache
export TORCH_HOME=$H/.cache/torch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$P/scripts/gem/optional_stubs:$GEM

echo "official_commit=$EXPECTED_COMMIT official_weight_sha256=$EXPECTED_WEIGHT_SHA"
echo "protocol=official sample.py, ZeroShot one-image smoke, 25 frames, 50 steps"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
cd "$GEM"
"$H/vista_venv/bin/python" sample.py \
  --dataset ZeroShot \
  --save "$OUT" \
  --condition_type unconditional \
  --n_frames 25 \
  --n_steps 50 \
  --seed 0 \
  --rand_gen
echo "gem_official_smoke_completed out=$OUT"
