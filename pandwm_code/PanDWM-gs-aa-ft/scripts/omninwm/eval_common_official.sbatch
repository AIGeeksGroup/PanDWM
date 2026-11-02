#!/bin/bash
#SBATCH --job-name=omni_common
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=640G
#SBATCH --time=24:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --array=0-14%2
#SBATCH --output=%x_%A_%a.log
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

SEEDS=(0 42 1000)
SHARD=$SLURM_ARRAY_TASK_ID
START=$((SHARD * 10))
END=$((START + 10))
if [ "$END" -gt 149 ]; then END=149; fi
DATA=$SHARD_ROOT/shard_$(printf '%02d' "$SHARD").pkl

test "$(sed -n '1p' "$REPO/.git/refs/heads/master")" = \
  "8e2333ee0dfb7be7a477e51a97e644bf7ec53039"
test -f "$MODEL/model/model.safetensors.index.json"
test -f "$VAE"
test -f "$DATA"
test -f "$DATA_ROOT/manifest.json"
test -f "$SHARD_ROOT/sampler_audit.json"

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
echo "sampler_protocol=official_hybrid_sp8_dp1_unreplicated"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
EXPECTED=$((END - START))
for SEED in "${SEEDS[@]}"; do
  OUT=$H/omninwm_runs/common149/seed_${SEED}/shard_${SHARD}
  if [ -d "$OUT/lossless" ]; then
    ACTUAL=$(find "$OUT/lossless" -maxdepth 1 -type f -name 'clip_*.pt' | wc -l)
    test "$ACTUAL" -eq "$EXPECTED"
    echo "already_complete seed=$SEED shard=$SHARD clips=$ACTUAL"
    continue
  fi
  test ! -e "$OUT"
  TMP=$H/omninwm_runs/common149_work/job_${SLURM_JOB_ID}/seed_${SEED}/shard_${SHARD}
  test ! -e "$TMP"
  mkdir -p "$(dirname "$TMP")" "$(dirname "$OUT")"
  echo "protocol=common149 seed=$SEED shard=$SHARD range=[$START,$END) steps=50 frames=33 views=6"
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
  echo "omninwm_common_shard_completed seed=$SEED shard=$SHARD clips=$ACTUAL"
done
