#!/usr/bin/env bash
# Train bidirectional pano teacher (Stage 0).
#
# Usage:
#   bash scripts/train_ctsd_unimlvg_stage3_tirda_bidirectional_nwa.sh
#   RESUME_OUT=<run_dir> RESUME_FROM=N bash scripts/train_ctsd_unimlvg_stage3_tirda_bidirectional_nwa.sh
set -euo pipefail

# Clear any outer srun-step env so the inner srun reuses our jobid cleanly.
unset $(compgen -v | grep -E '^SLURM_(STEP|NTASKS|NPROCS|NNODES|NODEID|LOCALID|PROCID|GTIDS|JOB_NODELIST|NODELIST|TASKS_PER_NODE|TOPOLOGY)') 2>/dev/null || true

OPENDWM=/scratch/fs201130/jn20658/tenants/clientB/workspace/OpenDWM
PYBIN=/scratch/fs201130/jn20658/tenants/clientB/workspace/miniconda3/envs/opendwm/bin
PANDWM=/scratch/fs201130/jn20658/tenants/clientB/workspace/PanDWM
ONEBEV_STITCH=/scratch/fs201130/jn20658/tenants/clientB/workspace/OneBEV/tools/stitch

CFG=$OPENDWM/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_bidirectional_nwa.json
RUN_BASE=/data/fs201130/jn20658/tenants/clientB/OpenDWM/runs

JOBID=${JOBID:-15151}
NODELIST=${NODELIST:-n3001-[004,007]}
MASTER_NODE=${MASTER_NODE:-n3001-004}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-2}
NGPU_PER_NODE=${NGPU_PER_NODE:-4}
NCPUS_PER_TASK=${NCPUS_PER_TASK:-64}

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
echo "nodes:  $NODELIST (master=$MASTER_NODE)"           | tee -a "$LOG"
echo "config: $CFG"                                      | tee -a "$LOG"
echo "output: $OUT"                                      | tee -a "$LOG"
echo "resume: ${RESUME_FROM:-<none>}"                    | tee -a "$LOG"

srun --jobid="$JOBID" --overlap --nodelist="$NODELIST" \
     --nodes="$NNODES" --ntasks="$NNODES" --gres=gpu:"$NGPU_PER_NODE" --cpus-per-task="$NCPUS_PER_TASK" \
     --export=ALL,\
PYTHONPATH=$OPENDWM/src:$OPENDWM/externals/TATS/tats/fvd:$PANDWM,\
ONEBEV_STITCH_PATH=$ONEBEV_STITCH,\
OMP_NUM_THREADS=1,TOKENIZERS_PARALLELISM=false,\
PATH=$PYBIN:$PATH,\
WANDB_API_KEY=<YOUR_WANDB_API_KEY>,\
WANDB_DIR=$OUT,WANDB_CACHE_DIR=$OUT/wandb_cache,\
HF_AUTO_UPLOAD_REPO=zeyuren2002/nuscene_360_train,\
HF_TOKEN=<YOUR_HF_TOKEN>,\
HF_BIN=$PYBIN/hf,\
HF_HUB_ENABLE_HF_TRANSFER=1,\
HF_HUB_DISABLE_XET=1,\
TORCHINDUCTOR_CACHE_DIR=/data/fs201130/jn20658/tenants/clientB/OpenDWM/torchinductor_cache \
     "$PYBIN/torchrun" \
        --nnodes="$NNODES" --nproc-per-node="$NGPU_PER_NODE" \
        --rdzv-id="pandwm_$(date +%s)" --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_NODE:$MASTER_PORT" \
        "$OPENDWM/src/dwm_pano/train.py" \
        -c "$CFG" -o "$OUT" \
        --log-steps 100 --preview-steps 1000 \
        --checkpointing-steps 2000 --evaluation-steps 2000 \
        --wandb --wandb-project pandwm-unimlvg \
        --wandb-run-name "$(basename "$CFG" .json)_$(date +%Y%m%d_%H%M)" \
        ${RESUME_FROM:+--resume-from $RESUME_FROM} \
     2>&1 | tee -a "$LOG"
