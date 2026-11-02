#!/bin/bash
#SBATCH --job-name=omni_lossless
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
OUT=$H/omninwm_runs/lossless_smoke_dp1_two

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
export OMNINWM_OFFICIAL_REPO=$REPO
export OMNINWM_TRITON_CACHE_ROOT=$H/omninwm_runs/triton_cache/job_${SLURM_JOB_ID}

cd "$REPO"
"$V/bin/python" -m torch.distributed.run --nproc-per-node 8 \
  --no-python /bin/bash "$P/scripts/omninwm/run_with_rank_cache.sh" \
  "$V/bin/python" "$P/scripts/omninwm/inference_lossless_wrapper.py" \
  configs/inference/infer.py \
  --model.from_pretrained "$MODEL" \
  --ae.from_pretrained "$VAE" \
  --dataset.pkl_path "$DATA" \
  --start-index 0 \
  --end-index 2 \
  --seed 42 \
  --save-dir "$OUT" \
  --outputs "$OUT"

test "$(find "$OUT/lossless" -maxdepth 1 -type f -name '*.pt' | wc -l)" -eq 2
test "$(find "$OUT/video" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -eq 2
"$V/bin/python" - "$OUT/lossless/clip_0000.pt" "$OUT/lossless/clip_0001.pt" <<'PY'
import sys,torch
for path in sys.argv[1:]:
    x=torch.load(path,map_location='cpu',weights_only=True)
    assert x.shape==(6,3,33,448,800) and x.dtype==torch.uint8
    print('lossless_smoke_ok',path,tuple(x.shape),x.dtype)
PY
