#!/bin/bash
#SBATCH --job-name=omni_smoke
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=640G
#SBATCH --time=02:00:00
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
DATA=$H/omninwm_runs/common149_adapter/sp_shards_dp1/shard_00.pkl
OUT=$H/omninwm_runs/official_smoke_dp1_two

test "$(sed -n '1p' "$REPO/.git/refs/heads/master")" = \
  "8e2333ee0dfb7be7a477e51a97e644bf7ec53039"
test -f "$MODEL/model/model.safetensors.index.json"
test -f "$VAE"
test -f "$DATA"
test ! -e "$OUT"

module load gcc/11.5
module load cuda/12.1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH=$REPO
export OMNINWM_TRITON_CACHE_ROOT=$H/omninwm_runs/triton_cache/job_${SLURM_JOB_ID}

cd "$REPO"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
"$V/bin/python" -m torch.distributed.run --nproc-per-node 8 \
  --no-python /bin/bash "$P/scripts/omninwm/run_with_rank_cache.sh" \
  "$V/bin/python" tools/inference.py configs/inference/infer.py \
  --model.from_pretrained "$MODEL" \
  --ae.from_pretrained "$VAE" \
  --dataset.pkl_path "$DATA" \
  --start-index 0 \
  --end-index 2 \
  --seed 42 \
  --save-dir "$OUT" \
  --outputs "$OUT"

test "$(find "$OUT/video" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -eq 2
find "$OUT" -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
echo "omninwm_official_two_clip_smoke_completed"
