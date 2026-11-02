#!/bin/bash
#SBATCH --job-name=epona_common
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=/path/to/pandwm_root/external_runs/epona/logs/%x_%A_%a.log
set -euo pipefail

SEED=${SEED:?export SEED}
SHARD_SIZE=${SHARD_SIZE:-5}
SMOKE=${SMOKE:-0}
case "$SMOKE" in
  0) PREDICTION_FRAMES=16; RUN_NAME=common_target_10ctx_16pred ;;
  1) PREDICTION_FRAMES=6; RUN_NAME=smoke_10ctx_6pred ;;
  2) PREDICTION_FRAMES=16; RUN_NAME=smoke_10ctx_16pred_temporal6 ;;
  *) echo "SMOKE must be 0, 1, or 2" >&2; exit 2 ;;
esac
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START=$((TASK_ID * SHARD_SIZE))
END=$((START + SHARD_SIZE))
if [ "$END" -gt 149 ]; then END=149; fi
if [ "$START" -ge 149 ]; then
  echo "empty shard task=$TASK_ID start=$START"
  exit 0
fi

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
E=$H/external_baselines/Epona
M=$H/external_models/epona
OUT=$H/external_runs/epona/$RUN_NAME/seed_${SEED}/lossless
REAL=$H/horizondrive_runs/common/job_608501/lossless/real
COMMIT=69b24c55f5ab8b3ffde8fa55e9f833bfe64d2c68

for path in \
  "$E/configs/dit_config_dcae_nuscenes.py" \
  "$M/epona_nuplan+nusc.pkl" \
  "$M/dcae_td_20000.pkl" \
  "$M/SHA256SUMS" \
  "$H/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl" \
  "$H/pandwm_data/rgb" \
  "$REAL"; do
  test -e "$path" || { echo "missing: $path" >&2; exit 2; }
done

mkdir -p "$OUT/fake" "$OUT/manifests" "$H/external_runs/epona/logs"
if [ ! -e "$OUT/real" ]; then
  ln -s "$REAL" "$OUT/real" 2>/dev/null || test -L "$OUT/real"
fi
CKPT_SHA=$(awk '$2 == "epona_nuplan+nusc.pkl" {print $1}' "$M/SHA256SUMS")
VAE_SHA=$(awk '$2 == "dcae_td_20000.pkl" {print $1}' "$M/SHA256SUMS")
test "${#CKPT_SHA}" -eq 64
test "${#VAE_SHA}" -eq 64

source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$E:$P/src
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host=$(hostname) seed=$SEED shard=$TASK_ID range=$START:$END"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
cd "$E"
"$H/horizondrive_venv/bin/python" "$P/scripts/epona/eval_common.py" \
  --config "$E/configs/dit_config_dcae_nuscenes.py" \
  --checkpoint "$M/epona_nuplan+nusc.pkl" \
  --vae "$M/dcae_td_20000.pkl" \
  --ann "$H/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl" \
  --pano-rgb "$H/pandwm_data/rgb" \
  --canonical-real-dir "$REAL" \
  --output "$OUT" \
  --seed "$SEED" \
  --start-index "$START" \
  --end-index "$END" \
  --prediction-frames "$PREDICTION_FRAMES" \
  --upstream-commit "$COMMIT" \
  --checkpoint-sha256 "$CKPT_SHA" \
  --vae-sha256 "$VAE_SHA"
