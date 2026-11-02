#!/usr/bin/env bash
#SBATCH --job-name=pandwm_noise
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

ARM=${ARM:?ARM=control or noise}
case "$ARM" in control|noise) ;; *) exit 2 ;; esac
STEPS=${STEPS:-250}
RUN_TAG=${RUN_TAG:-noise_pilot}
DATA_ROOT=/path/to/pandwm_root
CODE_ROOT=$DATA_ROOT/pandwm_code/PanDWM_long_horizon_20260928
BASE=$DATA_ROOT/pandwm_code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_hpc3.json
STUDENT=$DATA_ROOT/pandwm_data/cd_10000_ema.pth
TEACHER=$DATA_ROOT/pandwm_data/stage1_causal_20000.pth
RUN=$DATA_ROOT/pandwm_runs/long_horizon_20260928/$RUN_TAG/$ARM
for file in "$BASE" "$STUDENT" "$TEACHER" "$CODE_ROOT/make_noise_training_config.py"; do
  test -f "$file" || { echo "missing $file" >&2; exit 2; }
done
test ! -e "$RUN" || { echo "refusing to overwrite $RUN" >&2; exit 2; }
mkdir -p "$RUN/log" "$RUN/torchinductor_cache"
source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$CODE_ROOT/src:$DATA_ROOT/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$RUN/torchinductor_cache
export DWM_MEM_PROBE_STEP=1 DWM_SKIP_OPTIMIZER_STATE=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 10000))
unset HF_AUTO_UPLOAD_REPO || true
"$DATA_ROOT/pandwm_venv/bin/python" "$CODE_ROOT/make_noise_training_config.py" \
  --base "$BASE" --output "$RUN/log/config.json" --arm "$ARM" \
  --student "$STUDENT" --teacher "$TEACHER" --data-root "$DATA_ROOT" \
  --seed 0 --lr 2e-7 --max-sigma 0.05 --probability 0.5
cd "$CODE_ROOT"
"$DATA_ROOT/pandwm_venv/bin/python" src/dwm_pano/train.py \
  -c "$RUN/log/config.json" -o "$RUN" --max-steps "$STEPS" \
  --log-steps 1 --preview-steps 100000000 \
  --evaluation-steps 100000000 --checkpointing-steps "$STEPS"
test -f "$RUN/checkpoints/${STEPS}_ema.pth"
echo "completed $ARM $RUN/checkpoints/${STEPS}_ema.pth"
