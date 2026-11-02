#!/usr/bin/env bash
# 2026-10-21 D2 refiner training (PanDWM-d2-refiner-20261021/D2_PLAN_20261021.md Tasks 8 / 9 / 11).
#   TAG=pilot STEPS=1000 CKPT_STEPS=500 sbatch --job-name=pandwm_d2_pilot train_d2.sbatch
#   SMOKE=1 sbatch --gres=gpu:2 --cpus-per-task=16 --mem=240G --time=01:00:00 --job-name=pandwm_d2_smoke train_d2.sbatch
#   RESUME_FROM=<step> TAG=<existing run> STEPS=<new max> ... : resume in place (no overwrite check)
#   CONFIG_FILE=$D2/configs/d2_full_hpc3_8gpu.json : the section 43.3 continuation (render-init samples)
#SBATCH --job-name=pandwm_d2
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/d2_refiner_20261021/logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58,ACD1-59
set -uo pipefail
DR=/path/to/pandwm_root
D2=$DR/pandwm_code/PanDWM_d2_refiner_20261021
RUNROOT=$DR/pandwm_runs/d2_refiner_20261021/runs
PY=$DR/pandwm_venv/bin/python
SMOKE=${SMOKE:-0}; STEPS=${STEPS:-1000}; CKS=${CKPT_STEPS:-500}; LOGS=${LOG_STEPS:-10}; TAG=${TAG:-pilot}
RESUME=${RESUME_FROM:-}
export PYTHONPATH=$D2/src:$D2/src_base:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_HOME=$DR/.cache/torch
export MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID} % 20000 ))
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR || true
cd "$D2" || exit 1
CONFIG=${CONFIG_FILE:-$D2/configs/d2_pilot_hpc3_8gpu.json}
NGPU=8
if [[ $SMOKE == 1 ]]; then
    NGPU=2; STEPS=4; CKS=100000000; LOGS=1; TAG=smoke_${SLURM_JOB_ID}
    CONFIG=$D2/configs/d2_smoke_hpc3_2gpu.json
    export DWM_AUDIT_STEPS=4 DWM_MEM_PROBE_STEP=2
    "$PY" tests/test_d2_logic.py || { echo "UNIT_TEST_FAILED"; exit 1; }
fi
RUN=$RUNROOT/$TAG
RES_ARGS=""
if [[ -n $RESUME ]]; then
    [[ -f $RUN/checkpoints/$RESUME.pth ]] || { echo "NO_RESUME_CKPT $RUN/checkpoints/$RESUME.pth"; exit 2; }
    RES_ARGS="--resume-from $RESUME"
else
    [[ -e $RUN/checkpoints ]] && { echo "REFUSE_OVERWRITE $RUN"; exit 2; }
fi
mkdir -p "$RUN/log"
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}
cp "$CONFIG" "$RUN/log/config_job_${SLURM_JOB_ID:-x}.json"
echo "D2_TRAIN_START $(date -Is) node=$(hostname) steps=$STEPS ngpu=$NGPU run=$RUN resume=${RESUME:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -2
"$PY" -m torch.distributed.run --standalone --nproc_per_node=$NGPU src_base/dwm_pano/train.py -c "$CONFIG" -o "$RUN" \
    --max-steps "$STEPS" --log-steps "$LOGS" --preview-steps 100000000 --evaluation-steps 100000000 \
    --checkpointing-steps "$CKS" $RES_ARGS
rc=$?
ls -la "$RUN/checkpoints" 2>/dev/null | tail -4
echo "D2_TRAIN_END rc=$rc $(date -Is)"
exit $rc
