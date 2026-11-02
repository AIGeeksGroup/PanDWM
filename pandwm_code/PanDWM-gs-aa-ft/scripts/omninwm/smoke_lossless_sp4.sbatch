#!/bin/bash
# Validate a resource-only SP=4 inference layout before using it for scoring.
# The official README documents SP=8; results from this job must be compared
# against the existing SP=8 output with the same checkpoint, input, and seed.
#SBATCH --job-name=omni_sp4_smoke
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
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
OUT=$H/omninwm_runs/lossless_smoke_sp4_one

test "$(sed -n '1p' "$REPO/.git/refs/heads/master")" = \
  "8e2333ee0dfb7be7a477e51a97e644bf7ec53039"
test -f "$MODEL/model/model.safetensors.index.json"
test -f "$VAE"
test -f "$DATA"
test ! -e "$OUT"

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
echo "protocol=sp4_equivalence_smoke batch_size=1 seed=42 steps=50 frames=33 views=6"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
"$V/bin/python" -m torch.distributed.run --nproc-per-node 4 \
  --no-python /bin/bash "$P/scripts/omninwm/run_with_rank_cache.sh" \
  "$V/bin/python" "$P/scripts/omninwm/inference_lossless_wrapper.py" \
  configs/inference/infer.py \
  --plugin_config.sp_size 4 \
  --model.from_pretrained "$MODEL" \
  --ae.from_pretrained "$VAE" \
  --dataset.pkl_path "$DATA" \
  --start-index 0 \
  --end-index 1 \
  --seed 42 \
  --save-dir "$OUT" \
  --outputs "$OUT"

test "$(find "$OUT/lossless" -maxdepth 1 -type f -name '*.pt' | wc -l)" -eq 1
test "$(find "$OUT/video" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -eq 1
"$V/bin/python" - "$OUT/lossless/clip_0000.pt" <<'PY'
import sys, torch
x = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
assert x.shape == (6, 3, 33, 448, 800) and x.dtype == torch.uint8
print("sp4_lossless_smoke_ok", tuple(x.shape), x.dtype)
PY
