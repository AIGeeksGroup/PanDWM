#!/usr/bin/env bash
# Eval PanDWM bidirectional ckpt with the matching eval16gpu config.
#
# Usage:
#   bash scripts/eval_ctsd_unimlvg_stage3_tirda_bidirectional_nwa.sh
#   STEP=12000 OUT=<run_dir> bash scripts/eval_ctsd_unimlvg_stage3_tirda_bidirectional_nwa.sh
set -euo pipefail

# Clear any outer srun-step env so the inner srun reuses our jobid cleanly.
unset $(compgen -v | grep -E '^SLURM_(STEP|NTASKS|NPROCS|NNODES|NODEID|LOCALID|PROCID|GTIDS|JOB_NODELIST|NODELIST|TASKS_PER_NODE|TOPOLOGY)') 2>/dev/null || true

OPENDWM=/scratch/fs201130/jn20658/tenants/clientB/workspace/OpenDWM
PYBIN=/scratch/fs201130/jn20658/tenants/clientB/workspace/miniconda3/envs/opendwm/bin
PANDWM=/scratch/fs201130/jn20658/tenants/clientB/workspace/PanDWM
ONEBEV_STITCH=/scratch/fs201130/jn20658/tenants/clientB/workspace/OneBEV/tools/stitch

CFG=$OPENDWM/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_bidirectional_nwa_eval16gpu.json
OUT=${OUT:-/data/fs201130/jn20658/tenants/clientB/OpenDWM/runs/train_ctsd_unimlvg_stage3_tirda_bidirectional_nwa_20260504_192921}
STEP=${STEP:-12000}

JOBID=${JOBID:-15151}
NODELIST=${NODELIST:-n3001-[004,007]}
MASTER_NODE=${MASTER_NODE:-n3001-004}
MASTER_PORT=${MASTER_PORT:-29501}
NNODES=${NNODES:-2}
NGPU_PER_NODE=${NGPU_PER_NODE:-4}
NCPUS_PER_TASK=${NCPUS_PER_TASK:-64}

LOG=$OPENDWM/outputs/eval_$(basename "$CFG" .json)_step${STEP}_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$(dirname "$LOG")"

echo "==== eval ($(basename "$CFG" .json), step=$STEP) ====" | tee "$LOG"
echo "jobid:  $JOBID  ($NNODES x $NGPU_PER_NODE GPU)"        | tee -a "$LOG"
echo "nodes:  $NODELIST (master=$MASTER_NODE)"               | tee -a "$LOG"
echo "config: $CFG"                                          | tee -a "$LOG"
echo "output: $OUT"                                          | tee -a "$LOG"
echo "ckpt:   $OUT/checkpoints/${STEP}.pth"                  | tee -a "$LOG"

srun --jobid="$JOBID" --overlap --nodelist="$NODELIST" \
     --nodes="$NNODES" --ntasks="$NNODES" --gres=gpu:"$NGPU_PER_NODE" --cpus-per-task="$NCPUS_PER_TASK" \
     --export=ALL,\
PYTHONPATH=$OPENDWM/src:$OPENDWM/externals/TATS/tats/fvd:$PANDWM,\
ONEBEV_STITCH_PATH=$ONEBEV_STITCH,\
OMP_NUM_THREADS=1,TOKENIZERS_PARALLELISM=false,\
PATH=$PYBIN:$PATH,\
TORCHINDUCTOR_CACHE_DIR=/data/fs201130/jn20658/tenants/clientB/OpenDWM/torchinductor_cache \
     "$PYBIN/torchrun" \
        --nnodes="$NNODES" --nproc-per-node="$NGPU_PER_NODE" \
        --rdzv-id="eval_$(date +%s)" --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_NODE:$MASTER_PORT" \
        "$OPENDWM/src/dwm_pano/evaluate.py" \
        -c "$CFG" -o "$OUT" --resume-from "$STEP" \
     2>&1 | tee -a "$LOG"
