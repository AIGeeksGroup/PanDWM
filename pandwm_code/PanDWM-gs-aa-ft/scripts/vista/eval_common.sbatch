#!/bin/bash
#SBATCH --job-name=vista_common
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
# Submit only after the official sample.py smoke passes, for example:
#   sbatch --export=ALL,SEED=0,SHARDS=5,SHARD=0 eval_common.sbatch
set -euo pipefail

SEED=${SEED:?SEED is required}
SHARDS=${SHARDS:?SHARDS is required}
SHARD=${SHARD:?SHARD is required}
H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
VISTA=$H/external_baselines/Vista
WEIGHT=$H/external_models/vista/vista.safetensors
OPENCLIP=$H/external_models/open_clip/open_clip_pytorch_model.bin
INPUT=$H/vista_common_front149/manifest.json
OUT=$H/vista_runs/common_seed${SEED}
EXPECTED_COMMIT=cc9821b4253ca7987c32757613d2fc2448fa9f5d
EXPECTED_SHA256=1b267bb753aa5c5e4d363f6e4c864dd4e343996dedc18b5ab20af6d20be34b98
EXPECTED_OPENCLIP_SHA256=9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4

# Some compute nodes do not expose git in PATH.  The checkout is shared and was
# audited clean before submission, so verify its pinned branch ref without git.
test "$(sed -n '1p' "$VISTA/.git/HEAD")" = "ref: refs/heads/main"
test "$(sed -n '1p' "$VISTA/.git/refs/heads/main")" = "$EXPECTED_COMMIT"
test -f "$INPUT"
echo "$EXPECTED_SHA256  $WEIGHT" | sha256sum --check --strict
echo "$EXPECTED_OPENCLIP_SHA256  $OPENCLIP" | sha256sum --check --strict
mkdir -p "$OUT"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=$H/external_models/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$VISTA:$P/src
echo "official_commit=$EXPECTED_COMMIT seed=$SEED shard=$SHARD/$SHARDS"
echo "protocol=1 context frame, 25 output frames, 50 official steps, free action"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
cd "$P"
$H/vista_venv/bin/python scripts/vista/evaluate_common.py \
  --vista-root "$VISTA" \
  --checkpoint "$WEIGHT" \
  --manifest "$INPUT" \
  --out "$OUT" \
  --seed "$SEED" \
  --steps 50 \
  --cfg-scale 2.5 \
  --shards "$SHARDS" \
  --shard "$SHARD"
echo "completed out=$OUT seed=$SEED shard=$SHARD/$SHARDS"
