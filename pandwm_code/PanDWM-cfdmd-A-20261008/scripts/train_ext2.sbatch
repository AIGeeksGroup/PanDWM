#!/usr/bin/env bash
# 2026-10-12 18:00: second extension of branch H, from step 3000 to MAX_STEPS (default 4000).  Copy of
# scripts/train_ext.sbatch (left untouched while job 710735 runs from it) with three changes:
#   - config configs/cfdmd_Hx2_gan_v8_3_hpc3_8gpu.json (target_steps 4000; with 3000 every later step would save);
#   - it needs the branch at >= 3000 (the first extension finished) and seeds seeds/H_3000 before training on;
#   - markers PLAN_HX2_START / PLAN_HX2_COMPLETE.
#   sbatch --job-name=pandwm_cfdmd_Hx2 --dependency=afterok:710735 scripts/train_ext2.sbatch
#SBATCH --job-name=pandwm_cfdmd_ext
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
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -euo pipefail
ARM=H
MAX_STEPS=${MAX_STEPS:-4000}
FROM=3000
source /path/to/pandwm_root/pandwm_venv/bin/activate
CF=/path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008
DEMO=/path/to/pandwm_root/pandwm_code/PanDWM_long_horizon_20260928
R=/path/to/pandwm_root/pandwm_runs
case "$ARM" in
    H) export CAMPAIGN_DIR=$R/cfdmd_20261010_H; CONFIG=$CF/configs/cfdmd_Hx2_gan_v8_3_hpc3_8gpu.json ;;
    *) echo "UNKNOWN_ARM $ARM"; exit 1 ;;
esac
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
STEP=$(python3 - "$OUT" <<'PY'
import sys
from dwm_pano_cfdmd.checkpoint_io import latest_step
step = latest_step(sys.argv[1], 8)
print("none" if step is None else step)
PY
)
if [[ "$STEP" == none ]]; then
    echo "BRANCH_NOT_SEEDED: no committed checkpoint in $OUT"
    exit 1
fi
if (( STEP < FROM )); then echo "EXT_NOT_READY: $ARM is at step $STEP, the run to $FROM has not finished"; exit 1; fi
if (( STEP >= MAX_STEPS )); then echo "ARM_ALREADY_COMPLETE gan step=$STEP"; exit 0; fi
if (( STEP == FROM )) && [[ ! -e "$R/seeds/${ARM}_$FROM" ]]; then
    python3 "$CF/scripts/seed_branch.py" "$OUT" "$R/seeds/${ARM}_$FROM/gan" $FROM
fi
echo "PLAN_${ARM}X2_START $(date -Is) node=$(hostname) job=$SLURM_JOB_ID resume=$STEP max=$MAX_STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$CF/scripts/test_rank_optimizer_resume.py"
cp "$CONFIG" "$OUT/log/config_used_job_${SLURM_JOB_ID}.json"
echo "ARM_START gan resume=$STEP config=$(basename "$CONFIG") $(date -Is)"
set +e
python -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$CF/scripts/train_resumable_v7.py" -c "$CONFIG" -o "$OUT" \
    --log-steps 10 --preview-steps 100000000 --evaluation-steps 100000000 \
    --checkpointing-steps 100000000 --max-steps "$MAX_STEPS" --resume-from "$STEP" --resume-data-offset "$STEP" &
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
if (( RESULT == 0 )); then echo "ARM_COMPLETE gan $(date -Is)"; echo "PLAN_${ARM}X2_COMPLETE $(date -Is)"; exit 0; fi
echo "ARM_FAILED gan rc=$RESULT"; exit "$RESULT"
