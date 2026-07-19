#!/usr/bin/env bash
# Train the GS decoder downstream of the stage-2 CD model (v0 overfit).
#
# Usage:
#   JOBID=<alloc> bash scripts/train_ctsd_unimlvg_stage3_tirda_causal_nwa_consistency_distillation_gs_decoder.sh
#   JOBID=<alloc> RESUME_OUT=<run_dir> RESUME_FROM=N bash scripts/train_ctsd_unimlvg_stage3_tirda_causal_nwa_consistency_distillation_gs_decoder.sh
set -euo pipefail

# Clear any outer srun-step env so the inner srun reuses our jobid cleanly.
unset $(compgen -v | grep -E '^SLURM_(STEP|NTASKS|NPROCS|NNODES|NODEID|LOCALID|PROCID|GTIDS|JOB_NODELIST|NODELIST|TASKS_PER_NODE|TOPOLOGY)') 2>/dev/null || true

OPENDWM=/scratch/fs201130/jn20658/tenants/clientB/workspace/OpenDWM
PYBIN=/scratch/fs201130/jn20658/tenants/clientB/workspace/miniconda3/envs/opendwm/bin

CFG=$OPENDWM/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_causal_nwa_consistency_distillation_gs_decoder.json
RUN_BASE=/data/fs201130/jn20658/tenants/clientB/OpenDWM/runs

JOBID=${JOBID:?set a live slurm allocation id}
NODELIST=${NODELIST:-}
MASTER_NODE=${MASTER_NODE:-localhost}
MASTER_PORT=${MASTER_PORT:-29511}
NNODES=${NNODES:-1}
NGPU_PER_NODE=${NGPU_PER_NODE:-1}
NCPUS_PER_TASK=${NCPUS_PER_TASK:-8}

if [ -z "${RESUME_FROM:-}" ]; then
    OUT=$RUN_BASE/$(basename "$CFG" .json)_$(date +%Y%m%d_%H%M%S)
    mkdir -p "$OUT/checkpoints" "$OUT/optimizer" "$OUT/preview" "$OUT/log"
else
    OUT=${RESUME_OUT:?must set RESUME_OUT=<run_dir> for resume}
fi
LOG=$OUT/log/$(basename "$CFG" .json)_$(date +%Y%m%d_%H%M%S).log
cp "$CFG" "$OUT/log/config_used.json"

echo "==== train launch ($(basename "$CFG" .json)) ====" | tee "$LOG"
echo "jobid:  $JOBID  ($NNODES x $NGPU_PER_NODE GPU)"    | tee -a "$LOG"
echo "config: $CFG"                                      | tee -a "$LOG"
echo "output: $OUT"                                      | tee -a "$LOG"
echo "resume: ${RESUME_FROM:-<none>}"                    | tee -a "$LOG"

# `--gpu-bind=none` keeps the node's GPUs visible to the step (gres cgroup
# binding on these allocations breaks CUDA init, see the DMD launcher).
srun --jobid="$JOBID" --overlap ${NODELIST:+--nodelist=$NODELIST} \
     --nodes="$NNODES" --ntasks="$NNODES" \
     --gpus-per-node="$NGPU_PER_NODE" --gpu-bind=none \
     --cpus-per-task="$NCPUS_PER_TASK" --cpu-bind=none \
     --export=ALL,\
PYTHONPATH=$OPENDWM/src,\
PYTHONUNBUFFERED=1,\
OMP_NUM_THREADS=1,\
TORCH_HOME=/local/jn20658/torch_home,\
PATH=$PYBIN:$PATH,\
WANDB_API_KEY=<YOUR_WANDB_API_KEY>,\
WANDB_DIR=$OUT,WANDB_CACHE_DIR=$OUT/wandb_cache \
     "$PYBIN/torchrun" \
        --nnodes="$NNODES" --nproc-per-node="$NGPU_PER_NODE" \
        --rdzv-id="pandwm_gs_$(date +%s)" --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_NODE:$MASTER_PORT" \
        "$OPENDWM/src/dwm_pano/train.py" \
        -c "$CFG" -o "$OUT" \
        --log-steps 20 --preview-steps 200 \
        --checkpointing-steps 1000 --evaluation-steps 0 \
        --wandb --wandb-project pandwm-unimlvg \
        --wandb-run-name "$(basename "$CFG" .json)_$(date +%Y%m%d_%H%M)" \
        ${RESUME_FROM:+--resume-from $RESUME_FROM} \
     2>&1 | tee -a "$LOG"
