#!/bin/bash
#SBATCH --job-name=pandwm_boxtrack
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
#
# Paired production run (submit both with the same settings):
#   sbatch --export=ALL,ARM=control,STEPS=600 train_box_tracks.sbatch
#   sbatch --export=ALL,ARM=track,STEPS=600 train_box_tracks.sbatch
# Engineering smoke on the already-present val files (never a paper result):
#   sbatch --export=ALL,ARM=track,STEPS=2,CAL=1 train_box_tracks.sbatch
set -euo pipefail

ARM=${ARM:-track}
STEPS=${STEPS:-600}
SEED=${SEED:-0}
CAL=${CAL:-0}
LOSS_WEIGHT=${LOSS_WEIGHT:-0.05}
LR=${LR:-2e-6}
MAX_TRACKS=${MAX_TRACKS:-96}
BAD_TIMESTAMP_US=${BAD_TIMESTAMP_US:-1531884295049111}
H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
CFG_SRC=$P/configs/ctsd/pandwm/h20/cd_eval_hpc3.json
STUDENT=$H/pandwm_data/cd_10000_ema.pth
TEACHER=$H/pandwm_data/stage1_causal_20000.pth

case "$ARM" in control|track) ;; *) echo "ARM must be control or track" >&2; exit 2;; esac
case "$CAL" in 0|1) ;; *) echo "CAL must be 0 or 1" >&2; exit 2;; esac
if [ "$CAL" = 1 ]; then
  TRAIN_PKL=$H/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl
  TRAIN_RGB=$H/pandwm_data/rgb
  RUN=$H/pandwm_runs/boxtrack_cal_${ARM}_s${STEPS}_seed${SEED}_${SLURM_JOB_ID}
  CKPT_EVERY=100000000
else
  TRAIN_PKL=$H/pandwm_data/pickle/nuscenes_interp_12Hz_infos_train_smoke.pkl
  TRAIN_RGB=$H/pandwm_data/train/rgb
  RUN=$H/pandwm_runs/boxtrack_${ARM}_s${STEPS}_seed${SEED}
  CKPT_EVERY=${CKPT_EVERY:-300}
fi
BBOX=$H/pandwm_data/bbox
HDMAP=$H/pandwm_data/hdmap

for path in "$CFG_SRC" "$STUDENT" "$TEACHER" "$TRAIN_PKL"; do
  test -f "$path" || { echo "missing file: $path" >&2; exit 2; }
done
for path in "$TRAIN_RGB" "$BBOX" "$HDMAP"; do
  test -d "$path" || { echo "missing directory: $path" >&2; exit 2; }
done
test ! -e "$RUN" || { echo "refusing to overwrite $RUN" >&2; exit 2; }
mkdir -p "$RUN/log" "$RUN/torchinductor_cache"
CFG=$RUN/log/box_track_config.json

source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$P/src:$H/pandwm_code/TATS_fvd
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$RUN/torchinductor_cache
export DWM_MEM_PROBE_STEP=${DWM_MEM_PROBE_STEP:-1}
export DWM_SKIP_OPTIMIZER_STATE=${DWM_SKIP_OPTIMIZER_STATE:-1}
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 500))
unset HF_AUTO_UPLOAD_REPO || true

"$H/pandwm_venv/bin/python" "$P/scripts/hpc3/make_box_track_config.py" \
  --base "$CFG_SRC" --output "$CFG" --arm "$ARM" --seed "$SEED" \
  --student "$STUDENT" --teacher "$TEACHER" \
  --train-pkl "$TRAIN_PKL" --train-rgb "$TRAIN_RGB" \
  --bbox-dir "$BBOX" --hdmap-dir "$HDMAP" \
  --loss-weight "$LOSS_WEIGHT" --max-tracks "$MAX_TRACKS" --lr "$LR" \
  --bad-timestamp-us "$BAD_TIMESTAMP_US"

echo "host=$(hostname) arm=$ARM steps=$STEPS seed=$SEED cal=$CAL"
echo "train_pkl=$TRAIN_PKL train_rgb=$TRAIN_RGB run=$RUN"
echo "excluded_corrupt_timestamp_us=$BAD_TIMESTAMP_US checkpoint_every=$CKPT_EVERY"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
cd "$P"
"$H/pandwm_venv/bin/python" src/dwm_pano/train.py \
  -c "$CFG" -o "$RUN" \
  --max-steps "$STEPS" \
  --log-steps 1 \
  --preview-steps 100000000 \
  --checkpointing-steps "$CKPT_EVERY" \
  --evaluation-steps 100000000

echo "completed arm=$ARM run=$RUN"
