#!/usr/bin/env bash
#SBATCH --job-name=pandwm_srr_train
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%A_%a.log
set -euo pipefail
ARM=${ARM:?set ARM=control or recovery}
case "$ARM" in control|recovery) ;; *) exit 2 ;; esac
STEPS=${STEPS:-100}
RUN_TAG=${RUN_TAG:-srr_pilot_v1}
CKPT_STEPS=${CKPT_STEPS:-$STEPS}
DATA_ROOT=/path/to/pandwm_root
CODE_ROOT=$DATA_ROOT/pandwm_code/PanDWM_long_horizon_20260928
RUN_ROOT=$DATA_ROOT/pandwm_runs/long_horizon_20260928
RUN=$RUN_ROOT/$RUN_TAG/$ARM
CONFIG=${CONFIG_OVERRIDE:-$RUN_ROOT/srr_config/$ARM.json}
test ! -e "$RUN" || { echo "refusing to overwrite $RUN" >&2; exit 2; }
mkdir -p "$RUN/log" "$RUN/torchinductor_cache"
source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$CODE_ROOT/src:$DATA_ROOT/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$RUN/torchinductor_cache
export DWM_MEM_PROBE_STEP=1 DWM_SKIP_OPTIMIZER_STATE=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
# A fixed JOBID-derived port collided (EADDRINUSE) on a shared node;
# ask the kernel for a free one instead.
export MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
unset HF_AUTO_UPLOAD_REPO || true
cd "$CODE_ROOT"
"$DATA_ROOT/pandwm_venv/bin/python" src/dwm_pano/train.py \
  -c "$CONFIG" -o "$RUN" \
  --max-steps "$STEPS" --log-steps 1 \
  --preview-steps 100000000 --evaluation-steps 100000000 \
  --checkpointing-steps "$CKPT_STEPS"
test -f "$RUN/checkpoints/${STEPS}.pth"
echo "completed $ARM $RUN/checkpoints/${STEPS}.pth"
