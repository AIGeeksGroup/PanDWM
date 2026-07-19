#!/usr/bin/env bash
# 4-step distilled inference on a Stage 2 ckpt.
# Saves <OUT>/preview/<STEP>.mp4 for visual comparison with the training-loop
# 50-step preview (which doesn't reflect the few-step regime).
#
# Usage:
#   bash scripts/preview_..._4step.sh                       # STEP=4000 default
#   STEP=6000 bash scripts/preview_..._4step.sh
#   STEP=6000 EMA=1 bash scripts/preview_..._4step.sh       # use ema ckpt
set -euo pipefail

unset $(compgen -v | grep -E '^SLURM_(STEP|NTASKS|NPROCS|NNODES|NODEID|LOCALID|PROCID|GTIDS|JOB_NODELIST|NODELIST|TASKS_PER_NODE|TOPOLOGY)') 2>/dev/null || true

OPENDWM=/scratch/fs201130/jn20658/tenants/clientB/workspace/OpenDWM
PYBIN=/scratch/fs201130/jn20658/tenants/clientB/workspace/miniconda3/envs/opendwm/bin
PANDWM=/scratch/fs201130/jn20658/tenants/clientB/workspace/PanDWM
ONEBEV_STITCH=/scratch/fs201130/jn20658/tenants/clientB/workspace/OneBEV/tools/stitch

CFG=$OPENDWM/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_causal_nwa_consistency_distillation.json
RUN=${RUN:-/data/fs201130/jn20658/tenants/clientB/OpenDWM/runs/ctsd_unimlvg_stage3_tirda_causal_nwa_consistency_distillation_20260521_023203}
STEP=${STEP:-4000}
EMA=${EMA:-0}
SCENE_INDEX=${SCENE_INDEX:-0}
OUT=${OUT:-$RUN/preview_4step}

JOBID=${JOBID:-15150}
NODELIST=${NODELIST:-n3001-013}
MASTER_NODE=${MASTER_NODE:-n3001-013}
MASTER_PORT=${MASTER_PORT:-29503}
NNODES=${NNODES:-1}
NGPU_PER_NODE=${NGPU_PER_NODE:-4}
NCPUS_PER_TASK=${NCPUS_PER_TASK:-22}

mkdir -p "$OUT"
LOG=$OPENDWM/outputs/preview_4step_step${STEP}$([ "$EMA" = "1" ] && echo "_ema")_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$(dirname "$LOG")"

echo "==== 4-step preview (step=$STEP, ema=$EMA) ====" | tee "$LOG"
echo "jobid:  $JOBID  ($NNODES x $NGPU_PER_NODE GPU)"  | tee -a "$LOG"
echo "nodes:  $NODELIST (master=$MASTER_NODE)"         | tee -a "$LOG"
echo "config: $CFG"                                    | tee -a "$LOG"
echo "run:    $RUN"                                    | tee -a "$LOG"
echo "ckpt:   $RUN/checkpoints/${STEP}$([ "$EMA" = "1" ] && echo "_ema").pth" | tee -a "$LOG"
echo "out:    $OUT/preview/${STEP}.mp4"                | tee -a "$LOG"

EMA_FLAG=""
if [ "$EMA" = "1" ]; then EMA_FLAG="--ema"; fi

# New srun pattern (alloc 15150: 4 task-slots × 1 GPU × 22 CPU per node).
srun --jobid="$JOBID" --overlap --nodelist="$NODELIST" \
     --nodes="$NNODES" --ntasks="$((NNODES * NGPU_PER_NODE))" \
     --ntasks-per-node="$NGPU_PER_NODE" --distribution=block:block \
     --gpus-per-node="$NGPU_PER_NODE" --gpu-bind=none \
     --cpus-per-task="$NCPUS_PER_TASK" --cpu-bind=none \
     --export=ALL,\
PYTHONPATH=$OPENDWM/src:$OPENDWM/externals/TATS/tats/fvd:$PANDWM,\
PYTHONUNBUFFERED=1,\
ONEBEV_STITCH_PATH=$ONEBEV_STITCH,\
OMP_NUM_THREADS=1,TOKENIZERS_PARALLELISM=false,\
PATH=$PYBIN:$PATH,\
TORCHINDUCTOR_CACHE_DIR=/data/fs201130/jn20658/tenants/clientB/OpenDWM/torchinductor_cache,\
MASTER_ADDR=$MASTER_NODE,MASTER_PORT=$MASTER_PORT \
     bash -c "
        export LOCAL_RANK=\$SLURM_LOCALID
        export RANK=\$SLURM_PROCID
        export WORLD_SIZE=\$SLURM_NTASKS
        exec \"$PYBIN/python\" \"$OPENDWM/src/dwm_pano/preview_4step.py\" \
            -c \"$CFG\" -r \"$RUN\" -s \"$STEP\" -o \"$OUT\" \
            --scene-index \"$SCENE_INDEX\" $EMA_FLAG
     " \
     2>&1 | tee -a "$LOG"
