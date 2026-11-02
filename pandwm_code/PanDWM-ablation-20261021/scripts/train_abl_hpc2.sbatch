#!/usr/bin/env bash
# 2026-10-21 ablation, HPC2 (A800 whole node) copy of train_abl.sbatch for ONE arm (D_notemp), ABLATION_20261021.md 2.2.
# Differences from train_abl.sbatch (HPC3): partition / QOS i64m1tga800u8ka, --gres=gpu:a800:8 --exclusive,
# env ~/pandwm_env.sh, code = md5-checked snapshot in ~/pandwm_code/PanDWM-ablation-20261021/cfdmd (copied from the HPC3
# tree), config = the HPC3 config with the path prefix /path/to/pandwm_root -> /path/to/pandwm_root only, and a
# SMOKE first: SMOKE_STEPS (3) steps of the same config into a scratch campaign (abl_20261021_smoke_<ARM>_hpc2); only
# if it exits 0 with a log row the real run starts, in the same allocation.  The smoke dir is moved to ~/_trash after.
#   ARM=D_notemp sbatch scripts/train_abl_hpc2.sbatch
#SBATCH --job-name=pandwm_abl_D_hpc2
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:a800:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
ARM=${ARM:?ARM}
MAX_STEPS=${MAX_STEPS:-500}
SMOKE_STEPS=${SMOKE_STEPS:-3}
source $HOME/pandwm_env.sh
ABL=$HOME/pandwm_code/PanDWM-ablation-20261021
CF=$ABL/cfdmd
DEMO=$HOME/pandwm_code/PanDWM_long_horizon_20260928
R=$HOME/pandwm_runs
CONFIG=$ABL/configs/abl_${ARM}_v8_4_hpc2_8gpu.json
[[ -f "$CONFIG" ]] || { echo "CONFIG_MISSING $CONFIG"; exit 1; }
export PYTHONPATH=$CF/src:$DEMO/vendor:$DEMO/src:$HOME/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=$HOME/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR || true
echo "PLAN_ABL_HPC2 arm=$ARM $(date -Is) node=$(hostname) job=$SLURM_JOB_ID config=$CONFIG md5=$(md5sum "$CONFIG" | cut -d' ' -f1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( cd "$CF" && md5sum -c "$ABL/cfdmd_md5_hpc3.txt" --quiet ) || { echo "CODE_SNAPSHOT_MISMATCH"; exit 1; }
echo "CODE_SNAPSHOT_OK (md5 = HPC3 tree)"
latest() {
python3 - "$1" <<'PY'
import sys
from dwm_pano_cfdmd.checkpoint_io import latest_step
step = latest_step(sys.argv[1], 8)
print("none" if step is None else step)
PY
}
run_train() {  # $1 campaign dir, $2 max steps, $3 log steps, rest: extra args
    local camp=$1 max=$2 logs=$3; shift 3
    export CAMPAIGN_DIR=$camp PAUSE_FILE=$camp/PAUSE_REQUESTED
    local out=$camp/gan
    cp "$CONFIG" "$out/log/config_used_job_${SLURM_JOB_ID}.json"
    set +e
    python -m torch.distributed.run --standalone --nproc_per_node=8 \
        "$CF/scripts/train_resumable_v7.py" -c "$CONFIG" -o "$out" \
        --log-steps "$logs" --preview-steps 100000000 --evaluation-steps 100000000 \
        --checkpointing-steps 100000000 --max-steps "$max" "$@" &
    TRAIN_PID=$!
    wait "$TRAIN_PID"; local rc=$?
    while kill -0 "$TRAIN_PID" 2>/dev/null; do wait "$TRAIN_PID"; rc=$?; done
    set -e
    return $rc
}
trap 'touch "${PAUSE_FILE:-/dev/null}"; echo "PAUSE_SIGNAL_RECEIVED"' USR1 TERM
CAMP=$R/abl_20261021_${ARM}_hpc2
OUT=$CAMP/gan
OWNER=$CAMP/ABLATION_OWNER.json
if [[ -e "$CAMP" && ! -e "$OWNER" ]]; then echo "REFUSE: $CAMP exists and was not created by this script"; exit 1; fi
if [[ ! -e "$CAMP" ]]; then
    mkdir -p "$OUT/checkpoints" "$OUT/training_state" "$OUT/log"
    printf '{"arm": "%s", "host": "hpc2", "config": "%s", "config_md5": "%s", "created_by_job": "%s", "created_at": "%s"}\n' \
        "$ARM" "$CONFIG" "$(md5sum "$CONFIG" | cut -d' ' -f1)" "$SLURM_JOB_ID" "$(date -Is)" > "$OWNER"
fi
export CAMPAIGN_DIR=$CAMP PAUSE_FILE=$CAMP/PAUSE_REQUESTED
python -m torch.distributed.run --standalone --nproc_per_node=8 "$CF/scripts/test_rank_optimizer_resume.py"
# ---- smoke (once, before the real run starts) ----
if [[ ! -e "$CAMP/SMOKE_PASSED" ]]; then
    SM=$R/abl_20261021_smoke_${ARM}_hpc2
    [[ -e "$SM" ]] && { mkdir -p ~/_trash; mv "$SM" ~/_trash/$(basename "$SM")_$(date +%s); }
    mkdir -p "$SM/gan/checkpoints" "$SM/gan/training_state" "$SM/gan/log"
    echo "SMOKE_START steps=$SMOKE_STEPS $(date -Is)"
    t0=$(date +%s)
    if ! run_train "$SM" "$SMOKE_STEPS" 1; then echo "SMOKE_FAIL $(date -Is)"; exit 1; fi
    echo "SMOKE_PASS steps=$SMOKE_STEPS wall=$(( $(date +%s) - t0 ))s $(date -Is)"
    echo "job=$SLURM_JOB_ID $(date -Is)" > "$CAMP/SMOKE_PASSED"
    mkdir -p ~/_trash; mv "$SM" ~/_trash/$(basename "$SM")_passed_$(date +%s)
fi
[[ -e "$CAMP/PAUSE_REQUESTED" ]] && { echo "Campaign is paused"; exit 1; }
STEP=$(latest "$OUT")
EXTRA=()
if [[ "$STEP" != none ]]; then
    (( STEP >= MAX_STEPS )) && { echo "ARM_ALREADY_COMPLETE step=$STEP"; exit 0; }
    OFFSET=$(python3 "$CF/scripts/dataset_len.py" "$CONFIG" 8 "$STEP" | awk '/RESUME_DATA_OFFSET/ {print $2}')
    [[ "$OFFSET" =~ ^[0-9]+$ ]] || { echo "DATA_OFFSET_FAILED"; exit 1; }
    EXTRA=(--resume-from "$STEP" --resume-data-offset "$OFFSET")
fi
echo "ARM_START arm=$ARM resume=$STEP max=$MAX_STEPS ${EXTRA[*]} $(date -Is)"
if run_train "$CAMP" "$MAX_STEPS" 10 "${EXTRA[@]}" && [[ -s "$OUT/milestones/${MAX_STEPS}_ema.pth" ]]; then
    echo "ABL_TRAIN_DONE arm=$ARM ema=$OUT/milestones/${MAX_STEPS}_ema.pth size=$(stat -c %s "$OUT/milestones/${MAX_STEPS}_ema.pth") $(date -Is)"
    sha256sum "$OUT/milestones/${MAX_STEPS}_ema.pth" | tee "$OUT/milestones/${MAX_STEPS}_ema.pth.sha256"
    exit 0
fi
echo "ARM_FAILED $(date -Is)"; exit 1
