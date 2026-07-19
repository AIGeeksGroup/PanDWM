#!/usr/bin/env bash
# 4-step distilled FID/FVD eval on a Stage 2 ckpt.
# Runs the standard `evaluate_pipeline` over the full validation set with the
# pipeline's `inference_pipeline` hot-swapped to a 4-step
# (`denoising_step_list = [1000, 750, 500, 250]`) implementation — so FID/FVD
# reflect the distilled regime, not 50-step DDPM sampling.
#
# Usage:
#   bash scripts/eval_..._4step.sh                          # STEP=8000 default
#   STEP=10000 bash scripts/eval_..._4step.sh
#   STEP=10000 EMA=1 bash scripts/eval_..._4step.sh         # use ema ckpt
set -euo pipefail

unset $(compgen -v | grep -E '^SLURM_(STEP|NTASKS|NPROCS|NNODES|NODEID|LOCALID|PROCID|GTIDS|JOB_NODELIST|NODELIST|TASKS_PER_NODE|TOPOLOGY)') 2>/dev/null || true

OPENDWM=/scratch/fs201130/jn20658/tenants/clientB/workspace/OpenDWM
PYBIN=/scratch/fs201130/jn20658/tenants/clientB/workspace/miniconda3/envs/opendwm/bin
PANDWM=/scratch/fs201130/jn20658/tenants/clientB/workspace/PanDWM
ONEBEV_STITCH=/scratch/fs201130/jn20658/tenants/clientB/workspace/OneBEV/tools/stitch

CFG=$OPENDWM/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_causal_nwa_consistency_distillation.json
RUN=${RUN:-/data/fs201130/jn20658/tenants/clientB/OpenDWM/runs/ctsd_unimlvg_stage3_tirda_causal_nwa_consistency_distillation_20260521_023203}
STEP=${STEP:-8000}
EMA=${EMA:-0}

JOBID=${JOBID:-15150}
NODELIST=${NODELIST:-n3001-013}
MASTER_NODE=${MASTER_NODE:-n3001-013}
MASTER_PORT=${MASTER_PORT:-29504}
NNODES=${NNODES:-1}
NGPU_PER_NODE=${NGPU_PER_NODE:-4}
NCPUS_PER_TASK=${NCPUS_PER_TASK:-22}

SCRIPT_NAME=$(basename "$0" .sh)
LOG=$OPENDWM/outputs/${SCRIPT_NAME}_step${STEP}$([ "$EMA" = "1" ] && echo "_ema")_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$(dirname "$LOG")"

echo "==== 4-step eval (step=$STEP, ema=$EMA) ====" | tee "$LOG"
echo "jobid:  $JOBID  ($NNODES x $NGPU_PER_NODE GPU)" | tee -a "$LOG"
echo "nodes:  $NODELIST (master=$MASTER_NODE)"        | tee -a "$LOG"
echo "config: $CFG"                                   | tee -a "$LOG"
echo "run:    $RUN"                                   | tee -a "$LOG"
echo "ckpt:   $RUN/checkpoints/${STEP}$([ "$EMA" = "1" ] && echo "_ema").pth" | tee -a "$LOG"

EMA_FLAG=""
if [ "$EMA" = "1" ]; then EMA_FLAG="--ema"; fi

# Match the new train-script srun pattern: 1 srun-task per GPU, mapped via
# `--gpu-bind=none` (NCCL needs to see peer GPUs via NVML), block distribution
# so ranks are contiguous within a node, env vars translated to torch.distributed.
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
        exec \"$PYBIN/python\" \"$OPENDWM/src/dwm_pano/evaluate_4step.py\" \
            -c \"$CFG\" -r \"$RUN\" -s \"$STEP\" $EMA_FLAG
     " \
     2>&1 | tee -a "$LOG"
