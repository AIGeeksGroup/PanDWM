#!/usr/bin/env bash
# 2026-10-16 (prepared, not submitted): branch L = v8.6 (src/dwm_pano_cfdmd/pipeline_v8_6.py) = branch I (v8.4) with
# the mm3 low-pass floor of the rollout-window term raised 0.9 -> 0.97 (mm3_lf_floor_rollout; the first-window term
# keeps 0.9), from branch I step 4250 (seeds/I_4250).  Copy of scripts/train_v8_4_K.sbatch with ARM=L, campaign
# cfdmd_20261016_L, config configs/cfdmd_L_gan_v8_6_hpc3_8gpu.json, FROM=4250, MAX_STEPS default 4500 (same data and
# k sequence as I 4500 / K 4500, so the V84 monitors pair step by step); ACD1-24 added to --exclude (10-15 NCCL
# failures).  Reason: report section 30 appendix (window 1 / window 7 of the student and the teacher).
#   sbatch --job-name=pandwm_cfdmd_L scripts/train_v8_6_L.sbatch
#SBATCH --job-name=pandwm_cfdmd_L
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
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -euo pipefail
ARM=L
MAX_STEPS=${MAX_STEPS:-4500}
FROM=4250
source /path/to/pandwm_root/pandwm_venv/bin/activate
CF=/path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008
DEMO=/path/to/pandwm_root/pandwm_code/PanDWM_long_horizon_20260928
R=/path/to/pandwm_root/pandwm_runs
export CAMPAIGN_DIR=$R/cfdmd_20261016_L
CONFIG=$CF/configs/cfdmd_L_gan_v8_6_hpc3_8gpu.json
SEED_DIR=$R/seeds/I_4250/gan
export PAUSE_FILE=$CAMPAIGN_DIR/PAUSE_REQUESTED
export PYTHONPATH=$CF/src:$DEMO/vendor:$DEMO/src:/path/to/pandwm_root/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/path/to/pandwm_root/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR || true
trap 'touch "$PAUSE_FILE"; echo "PAUSE_SIGNAL_RECEIVED"' USR1 TERM
if [[ -e "$PAUSE_FILE" ]]; then
    echo "Campaign is paused; remove PAUSE_REQUESTED only when resume is requested."
    exit 0
fi
OUT=$CAMPAIGN_DIR/gan
latest() {
python3 - "$OUT" <<'PY'
import sys
from dwm_pano_cfdmd.checkpoint_io import latest_step
step = latest_step(sys.argv[1], 8)
print("none" if step is None else step)
PY
}
STEP=$(latest)
if [[ "$STEP" == none ]]; then
    [[ -e "$SEED_DIR/checkpoints/$FROM.complete.json" ]] || { echo "SEED_MISSING $SEED_DIR step $FROM"; exit 1; }
    python3 "$CF/scripts/seed_branch.py" "$SEED_DIR" "$OUT" $FROM
    STEP=$(latest)
fi
if [[ "$STEP" == none ]] || (( STEP < FROM )); then echo "BRANCH_NOT_SEEDED: $OUT step=$STEP"; exit 1; fi
if (( STEP >= MAX_STEPS )); then echo "ARM_ALREADY_COMPLETE gan step=$STEP"; exit 0; fi
# The 179-frame dataset is far smaller than the 51-frame one (11433 vs 100295 clips; 1430 sampler items per rank), and
# vendor_snapshot/train.py requires --resume-data-offset == resume step % sampler length (4000 -> 1140).
OFFSET=$(python3 "$CF/scripts/dataset_len.py" "$CONFIG" 8 "$STEP" | awk '/RESUME_DATA_OFFSET/ {print $2}')
[[ "$OFFSET" =~ ^[0-9]+$ ]] || { echo "DATA_OFFSET_FAILED"; exit 1; }
echo "PLAN_${ARM}_START $(date -Is) node=$(hostname) job=$SLURM_JOB_ID resume=$STEP max=$MAX_STEPS data_offset=$OFFSET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$CF/scripts/test_rank_optimizer_resume.py"
cp "$CONFIG" "$OUT/log/config_used_job_${SLURM_JOB_ID}.json"
echo "ARM_START gan resume=$STEP config=$(basename "$CONFIG") $(date -Is)"
set +e
python -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$CF/scripts/train_resumable_v7.py" -c "$CONFIG" -o "$OUT" \
    --log-steps 10 --preview-steps 100000000 --evaluation-steps 100000000 \
    --checkpointing-steps 100000000 --max-steps "$MAX_STEPS" --resume-from "$STEP" --resume-data-offset "$OFFSET" &
TRAIN_PID=$!
wait "$TRAIN_PID"
RESULT=$?
# A trap can interrupt wait while the child is still checkpointing.
while kill -0 "$TRAIN_PID" 2>/dev/null; do wait "$TRAIN_PID"; RESULT=$?; done
set -e
if [[ -e "$PAUSE_FILE" ]]; then
    echo "CAMPAIGN_PAUSED gan checkpoint safely requested; rc=$RESULT $(date -Is)"
    exit 0
fi
if (( RESULT == 0 )); then echo "ARM_COMPLETE gan $(date -Is)"; echo "PLAN_${ARM}_COMPLETE $(date -Is)"; exit 0; fi
echo "ARM_FAILED gan rc=$RESULT"; exit "$RESULT"
