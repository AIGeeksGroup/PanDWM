#!/usr/bin/env bash
# 2026-10-21 ablation (ABLATION_20261021.md): one variant of the final recipe (v8.4, branch I) trained for 500
# iterations (global steps) from the consistency-distilled student cd_10000_ema.pth, fresh optimizers / heads / EMA.
# Copy of PanDWM-cfdmd-A-20261008/scripts/train_v8_4.sbatch with these changes only:
#   - ARM selects configs/abl_<ARM>_v8_4_hpc3_8gpu.json (make_ablation_configs.py; diffs pre-registered);
#   - campaign pandwm_runs/abl_20261021_<ARM>; NO seeding from a staged checkpoint: the first run starts at step 0 from
#     the config's generator_ckpt (cd_10000_ema.pth), fake / real score from stage1_causal_20000.pth;
#   - a re-run (requeue after a node failure) resumes from the last committed full checkpoint of THIS campaign with
#     --resume-data-offset = step % sampler length (scripts/dataset_len.py, as in train_v8_4.sbatch);
#   - refuses to touch a campaign dir it did not create (marker ABLATION_OWNER.json);
#   - --mem 700G (branch L was OOM-killed at 480G; M ran at 700G); host memory only, no effect on the training math;
#   - ends with ABL_TRAIN_DONE only if milestones/500_ema.pth exists (afterok gate for the eval chain).
#   ARM=A_ctrl sbatch --job-name=pandwm_abl_A scripts/train_abl.sbatch
#SBATCH --job-name=pandwm_abl
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -euo pipefail
ARM=${ARM:?ARM}
MAX_STEPS=${MAX_STEPS:-500}
source /path/to/pandwm_root/pandwm_venv/bin/activate
CF=/path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008
ABL=/path/to/pandwm_root/pandwm_code/PanDWM-ablation-20261021
DEMO=/path/to/pandwm_root/pandwm_code/PanDWM_long_horizon_20260928
R=/path/to/pandwm_root/pandwm_runs
export CAMPAIGN_DIR=$R/abl_20261021_${ARM}
CONFIG=$ABL/configs/abl_${ARM}_v8_4_hpc3_8gpu.json
[[ -f "$CONFIG" ]] || { echo "CONFIG_MISSING $CONFIG"; exit 1; }
export PAUSE_FILE=$CAMPAIGN_DIR/PAUSE_REQUESTED
export PYTHONPATH=$CF/src:$DEMO/vendor:$DEMO/src:/path/to/pandwm_root/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/path/to/pandwm_root/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR || true
OUT=$CAMPAIGN_DIR/gan
OWNER=$CAMPAIGN_DIR/ABLATION_OWNER.json
if [[ -e "$CAMPAIGN_DIR" && ! -e "$OWNER" ]]; then
    echo "REFUSE: $CAMPAIGN_DIR exists and was not created by train_abl.sbatch"; exit 1
fi
if [[ ! -e "$CAMPAIGN_DIR" ]]; then
    mkdir -p "$OUT/checkpoints" "$OUT/training_state" "$OUT/log"
    printf '{"arm": "%s", "config": "%s", "config_md5": "%s", "created_by_job": "%s", "created_at": "%s"}\n' \
        "$ARM" "$CONFIG" "$(md5sum "$CONFIG" | cut -d' ' -f1)" "$SLURM_JOB_ID" "$(date -Is)" > "$OWNER"
fi
trap 'touch "$PAUSE_FILE"; echo "PAUSE_SIGNAL_RECEIVED"' USR1 TERM
if [[ -e "$PAUSE_FILE" ]]; then
    echo "Campaign is paused; remove PAUSE_REQUESTED only when resume is requested."
    exit 1
fi
latest() {
python3 - "$OUT" <<'PY'
import sys
from dwm_pano_cfdmd.checkpoint_io import latest_step
step = latest_step(sys.argv[1], 8)
print("none" if step is None else step)
PY
}
done_check() {
    if [[ -s "$OUT/milestones/${MAX_STEPS}_ema.pth" && -e "$OUT/milestones/${MAX_STEPS}.complete.json" ]]; then
        echo "ABL_TRAIN_DONE arm=$ARM ema=$OUT/milestones/${MAX_STEPS}_ema.pth size=$(stat -c %s "$OUT/milestones/${MAX_STEPS}_ema.pth") $(date -Is)"
        return 0
    fi
    return 1
}
STEP=$(latest)
if [[ "$STEP" != none ]] && (( STEP >= MAX_STEPS )); then
    echo "ARM_ALREADY_COMPLETE step=$STEP"; done_check && exit 0; echo "MILESTONE_MISSING"; exit 1
fi
EXTRA=()
if [[ "$STEP" != none ]]; then
    OFFSET=$(python3 "$CF/scripts/dataset_len.py" "$CONFIG" 8 "$STEP" | awk '/RESUME_DATA_OFFSET/ {print $2}')
    [[ "$OFFSET" =~ ^[0-9]+$ ]] || { echo "DATA_OFFSET_FAILED"; exit 1; }
    EXTRA=(--resume-from "$STEP" --resume-data-offset "$OFFSET")
fi
echo "PLAN_ABL_START arm=$ARM $(date -Is) node=$(hostname) job=$SLURM_JOB_ID resume=$STEP max=$MAX_STEPS ${EXTRA[*]} config=$CONFIG md5=$(md5sum "$CONFIG" | cut -d' ' -f1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$CF/scripts/test_rank_optimizer_resume.py"
cp "$CONFIG" "$OUT/log/config_used_job_${SLURM_JOB_ID}.json"
echo "ARM_START gan resume=$STEP config=$(basename "$CONFIG") $(date -Is)"
set +e
python -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$CF/scripts/train_resumable_v7.py" -c "$CONFIG" -o "$OUT" \
    --log-steps 10 --preview-steps 100000000 --evaluation-steps 100000000 \
    --checkpointing-steps 100000000 --max-steps "$MAX_STEPS" "${EXTRA[@]}" &
TRAIN_PID=$!
wait "$TRAIN_PID"
RESULT=$?
# A trap can interrupt wait while the child is still checkpointing.
while kill -0 "$TRAIN_PID" 2>/dev/null; do wait "$TRAIN_PID"; RESULT=$?; done
set -e
if [[ -e "$PAUSE_FILE" ]]; then
    echo "CAMPAIGN_PAUSED rc=$RESULT $(date -Is)"; exit 1
fi
if (( RESULT == 0 )) && done_check; then echo "ARM_COMPLETE $(date -Is)"; exit 0; fi
echo "ARM_FAILED rc=$RESULT"; exit 1
