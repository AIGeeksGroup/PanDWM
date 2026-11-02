#!/bin/bash
#SBATCH --job-name=pandwm_sfcd
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=7-00:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

ARM=${ARM:?ARM must be warmup|control|self}
STEPS=${STEPS:-8000}
SEED=${SEED:-0}
RUN_TAG=${RUN_TAG:-production}
CURRICULUM_STEPS=${CURRICULUM_STEPS:-2000}
case "$ARM" in warmup|control|self) ;; *) exit 2 ;; esac

DATA_ROOT=/path/to/pandwm_root
CODE_ROOT=$DATA_ROOT/pandwm_code/PanDWM_self_forcing_20260915
BASE_CONFIG=$DATA_ROOT/pandwm_code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_hpc3.json
STAGE1=$DATA_ROOT/pandwm_data/stage1_causal_20000.pth
WARMUP=$DATA_ROOT/pandwm_runs/self_forcing_stage2_20260915/production/warmup/checkpoints/2000_ema.pth
RUN=$DATA_ROOT/pandwm_runs/self_forcing_stage2_20260915/$RUN_TAG/$ARM
PARALLEL_SELF_MARKER=$DATA_ROOT/pandwm_runs/self_forcing_stage2_20260915/production/.self_parallel_owner
# The original campaign job was submitted as warmup -> control -> self.  When
# Self-Forcing is delegated to its own one-GPU Slurm job, let that original
# campaign finish after control instead of launching a duplicate writer for
# production/self.  The evaluation job is separately made dependent on both
# jobs, so this skip cannot start evaluation early.
if [ "$ARM" = self ] && [ "$RUN_TAG" = production ] && \
   [ "${SELF_PARALLEL_OWNER:-0}" != 1 ] && \
   [ -f "$PARALLEL_SELF_MARKER" ]; then
  echo "Self-Forcing delegated to parallel job: $(cat "$PARALLEL_SELF_MARKER")"
  exit 0
fi
STUDENT=$STAGE1
if [ "$ARM" != warmup ]; then
  case "$RUN_TAG" in
    smoke*) ;;
    *) STUDENT=$WARMUP ;;
  esac
fi
TRAIN_PKL=$DATA_ROOT/pandwm_data/pickle/nuscenes_interp_12Hz_infos_train_smoke.pkl
TRAIN_RGB=$DATA_ROOT/pandwm_data/train/rgb
BBOX=$DATA_ROOT/pandwm_data/bbox
HDMAP=$DATA_ROOT/pandwm_data/hdmap
META=$DATA_ROOT/pandwm_data/nuscenes_meta/v1.0-trainval

for path in "$CODE_ROOT/scripts/hpc3/make_self_forcing_config.py" \
            "$BASE_CONFIG" "$STUDENT" "$STAGE1" "$TRAIN_PKL" \
            "$META/sample.json"; do
  test -f "$path" || { echo "missing file: $path" >&2; exit 2; }
done
for path in "$TRAIN_RGB" "$BBOX" "$HDMAP"; do
  test -d "$path" || { echo "missing directory: $path" >&2; exit 2; }
done
test ! -e "$RUN" || { echo "refusing to overwrite $RUN" >&2; exit 2; }
mkdir -p "$RUN/log" "$RUN/torchinductor_cache"
CONFIG=$RUN/log/self_forcing_config.json

source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$CODE_ROOT/src:$DATA_ROOT/pandwm_code/TATS_fvd
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$RUN/torchinductor_cache
export DWM_MEM_PROBE_STEP=1 DWM_SKIP_OPTIMIZER_STATE=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 500))
unset HF_AUTO_UPLOAD_REPO || true

"$DATA_ROOT/pandwm_venv/bin/python" \
  "$CODE_ROOT/scripts/hpc3/make_self_forcing_config.py" \
  --base "$BASE_CONFIG" --output "$CONFIG" --arm "$ARM" --seed "$SEED" \
  --student "$STUDENT" --teacher "$STAGE1" \
  --train-pkl "$TRAIN_PKL" --train-rgb "$TRAIN_RGB" \
  --bbox-dir "$BBOX" --hdmap-dir "$HDMAP" --meta-dir "$META" \
  --curriculum-steps "$CURRICULUM_STEPS"

echo "host=$(hostname) arm=$ARM steps=$STEPS seed=$SEED"
echo "student=$STUDENT run=$RUN config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
cd "$CODE_ROOT"
"$DATA_ROOT/pandwm_venv/bin/python" src/dwm_pano/train.py \
  -c "$CONFIG" -o "$RUN" --max-steps "$STEPS" --log-steps 1 \
  --preview-steps 100000000 --evaluation-steps 100000000 \
  --checkpointing-steps "$([ "$STEPS" -le 2000 ] && echo "$STEPS" || echo 2000)"
test -f "$RUN/checkpoints/${STEPS}_ema.pth"
echo "completed arm=$ARM checkpoint=$RUN/checkpoints/${STEPS}_ema.pth"
