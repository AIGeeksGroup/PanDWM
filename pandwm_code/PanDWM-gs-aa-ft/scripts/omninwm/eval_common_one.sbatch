#!/bin/bash
# Submit exactly one OmniNWM common149 shard/seed pair.
# Required exports: OMNI_SHARD=0..14 and OMNI_SEED in {0,42,1000}.
#SBATCH --job-name=omni_one
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=640G
#SBATCH --time=08:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
REPO=$H/external_baselines/OmniNWM
V=$H/omninwm_venv
ASSETS=$H/external_models/omninwm
MODEL=$ASSETS/release/ckpt/dit
VAE=$ASSETS/hunyuan_vae.safetensors
DATA_ROOT=$H/omninwm_runs/common149_adapter
SHARD_ROOT=$DATA_ROOT/sp_shards_dp1

SHARD=${OMNI_SHARD:?export OMNI_SHARD=0..14}
SEED=${OMNI_SEED:?export OMNI_SEED=0,42,or1000}
case "$SHARD" in
  ''|*[!0-9]*) echo "invalid OMNI_SHARD=$SHARD" >&2; exit 2 ;;
esac
if [ "$SHARD" -lt 0 ] || [ "$SHARD" -gt 14 ]; then
  echo "invalid OMNI_SHARD=$SHARD" >&2
  exit 2
fi
case "$SEED" in
  0|42|1000) ;;
  *) echo "invalid OMNI_SEED=$SEED" >&2; exit 2 ;;
esac

START=$((SHARD * 10))
END=$((START + 10))
if [ "$END" -gt 149 ]; then END=149; fi
EXPECTED=$((END - START))
DATA=$SHARD_ROOT/shard_$(printf '%02d' "$SHARD").pkl
OUT=$H/omninwm_runs/common149/seed_${SEED}/shard_${SHARD}
TMP=$H/omninwm_runs/common149_work/job_${SLURM_JOB_ID}/seed_${SEED}/shard_${SHARD}

test "$(sed -n '1p' "$REPO/.git/refs/heads/master")" = \
  "8e2333ee0dfb7be7a477e51a97e644bf7ec53039"
test -f "$MODEL/model/model.safetensors.index.json"
test -f "$VAE"
test -f "$DATA"
test -f "$DATA_ROOT/manifest.json"
test -f "$SHARD_ROOT/sampler_audit.json"

if [ -d "$OUT/lossless" ]; then
  ACTUAL=$(find "$OUT/lossless" -maxdepth 1 -type f -name 'clip_*.pt' | wc -l)
  test "$ACTUAL" -eq "$EXPECTED"
  echo "already_complete seed=$SEED shard=$SHARD clips=$ACTUAL"
  exit 0
fi
test ! -e "$OUT"
test ! -e "$TMP"
mkdir -p "$(dirname "$OUT")" "$(dirname "$TMP")"

module load gcc/11.5
module load cuda/12.1
export OMNINWM_OFFICIAL_REPO=$REPO
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH=$REPO
export OMNINWM_TRITON_CACHE_ROOT=$H/omninwm_runs/triton_cache/job_${SLURM_JOB_ID}

cd "$REPO"
echo "protocol=common149 seed=$SEED shard=$SHARD range=[$START,$END) steps=50 frames=33 views=6"
echo "sampler_protocol=official_hybrid_sp8_dp1_unreplicated"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
"$V/bin/python" -m torch.distributed.run --nproc-per-node 8 \
  --no-python /bin/bash "$P/scripts/omninwm/run_with_rank_cache.sh" \
  "$V/bin/python" "$P/scripts/omninwm/inference_lossless_wrapper.py" \
  configs/inference/infer.py \
  --model.from_pretrained "$MODEL" \
  --ae.from_pretrained "$VAE" \
  --dataset.pkl_path "$DATA" \
  --seed "$SEED" \
  --save-dir "$TMP" \
  --outputs "$TMP"

ACTUAL=$(find "$TMP/lossless" -maxdepth 1 -type f -name 'clip_*.pt' | wc -l)
test "$ACTUAL" -eq "$EXPECTED"
mv "$TMP" "$OUT"
echo "omninwm_common_one_completed seed=$SEED shard=$SHARD clips=$ACTUAL"
