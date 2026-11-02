#!/usr/bin/env bash
#SBATCH --job-name=pandwm_cfdmd_A
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x_%j.log
set -euo pipefail
source /path/to/pandwm_root/pandwm_venv/bin/activate
CF=/path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008
DEMO=/path/to/pandwm_root/pandwm_code/PanDWM_long_horizon_20260928
export CAMPAIGN_DIR=/path/to/pandwm_root/pandwm_runs/cfdmd_20261008_A
export PAUSE_FILE=$CAMPAIGN_DIR/PAUSE_REQUESTED
export PYTHONPATH=$CF/src:$DEMO/vendor:$DEMO/src:/path/to/pandwm_root/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/path/to/pandwm_root/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR || true
mkdir -p "$CAMPAIGN_DIR"
trap 'touch "$PAUSE_FILE"; echo "PAUSE_SIGNAL_RECEIVED"' USR1 TERM
if [[ -e "$PAUSE_FILE" ]]; then
    echo "Campaign is paused; remove PAUSE_REQUESTED only when resume is requested."
    exit 0
fi
echo "PLAN_A_START $(date -Is) node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$CF/scripts/test_rank_optimizer_resume.py"
for ARM in gan; do
    OUT=$CAMPAIGN_DIR/$ARM
    mkdir -p "$OUT/checkpoints" "$OUT/training_state" "$OUT/log"
    CONFIG=$CF/configs/cfdmd_A_${ARM}_hpc3_8gpu.json
    STEP=$(python3 - "$OUT" <<'PY'
import sys
from dwm_pano_cfdmd.checkpoint_io import latest_step
step = latest_step(sys.argv[1], 8)
print("none" if step is None else step)
PY
)
    if [[ "$STEP" != none ]] && (( STEP >= 2000 )); then
        echo "ARM_ALREADY_COMPLETE $ARM step=$STEP"
        continue
    fi
    if [[ -e "$PAUSE_FILE" ]]; then echo "CAMPAIGN_PAUSED"; exit 0; fi
    cp "$CONFIG" "$OUT/log/config_used_job_${SLURM_JOB_ID}.json"
    EXTRA=()
    if [[ "$STEP" != none ]]; then EXTRA=(--resume-from "$STEP" --resume-data-offset "$STEP"); fi
    echo "ARM_START $ARM resume=$STEP $(date -Is)"
    set +e
    python -m torch.distributed.run --standalone --nproc_per_node=8 \
        "$CF/scripts/train_resumable.py" -c "$CONFIG" -o "$OUT" \
        --log-steps 10 --preview-steps 100000000 --evaluation-steps 100000000 \
        --checkpointing-steps 100000000 --max-steps 2000 "${EXTRA[@]}" &
    TRAIN_PID=$!
    wait "$TRAIN_PID"
    RESULT=$?
    # A trap can interrupt wait while the child is still checkpointing.
    while kill -0 "$TRAIN_PID" 2>/dev/null; do wait "$TRAIN_PID"; RESULT=$?; done
    set -e
    if [[ -e "$PAUSE_FILE" ]]; then
        echo "CAMPAIGN_PAUSED $ARM checkpoint safely requested; rc=$RESULT $(date -Is)"
        exit 0
    fi
    if (( RESULT != 0 )); then echo "ARM_FAILED $ARM rc=$RESULT"; exit "$RESULT"; fi
    echo "ARM_COMPLETE $ARM $(date -Is)"
done
echo "PLAN_A_COMPLETE $(date -Is)"
