#!/usr/bin/env bash
# 4-step FID/FVD eval on a Stage 3 (Asymmetric DMD) ckpt.
# Runs the standard `evaluate_pipeline` over the full validation set with the
# DMD pipeline's native 4-step `inference_pipeline` (no monkey-patch needed,
# unlike the Stage 2 CD eval).
#
# Usage:
#   bash scripts/eval_..._dmd_4step.sh                       # STEP=200 default
#   STEP=400 bash scripts/eval_..._dmd_4step.sh
#   STEP=400 EMA=1 bash scripts/eval_..._dmd_4step.sh        # use ema ckpt
#   RUN=<other_dmd_run_dir> STEP=N bash scripts/eval_..._dmd_4step.sh
set -euo pipefail

unset $(compgen -v | grep -E '^SLURM_(STEP|NTASKS|NPROCS|NNODES|NODEID|LOCALID|PROCID|GTIDS|JOB_NODELIST|NODELIST|TASKS_PER_NODE|TOPOLOGY)') 2>/dev/null || true

OPENDWM=/path/to/pandwm_root/workspace/OpenDWM
PYBIN=/path/to/pandwm_root/workspace/miniconda3/envs/opendwm/bin
PANDWM=/path/to/pandwm_root/workspace/PanDWM
ONEBEV_STITCH=/path/to/pandwm_root/workspace/OneBEV/tools/stitch

CFG=$OPENDWM/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_causal_nwa_dmd.json
RUN=${RUN:-/path/to/pandwm_root/OpenDWM/runs/ctsd_unimlvg_stage3_tirda_causal_nwa_dmd_20260524_062444}
STEP=${STEP:-200}
EMA=${EMA:-0}

JOBID=${JOBID:-15150}
NODELIST=${NODELIST:-n3001-013}
MASTER_NODE=${MASTER_NODE:-n3001-013}
MASTER_PORT=${MASTER_PORT:-29505}
NNODES=${NNODES:-1}
NGPU_PER_NODE=${NGPU_PER_NODE:-4}
NCPUS_PER_TASK=${NCPUS_PER_TASK:-22}

SCRIPT_NAME=$(basename "$0" .sh)
LOG=$OPENDWM/outputs/${SCRIPT_NAME}_step${STEP}$([ "$EMA" = "1" ] && echo "_ema")_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$(dirname "$LOG")"

echo "==== DMD 4-step eval (step=$STEP, ema=$EMA) ====" | tee "$LOG"
echo "jobid:  $JOBID  ($NNODES x $NGPU_PER_NODE GPU)"    | tee -a "$LOG"
echo "nodes:  $NODELIST (master=$MASTER_NODE)"           | tee -a "$LOG"
echo "config: $CFG"                                      | tee -a "$LOG"
echo "run:    $RUN"                                      | tee -a "$LOG"
echo "ckpt:   $RUN/checkpoints/${STEP}$([ "$EMA" = "1" ] && echo "_ema").pth" | tee -a "$LOG"

EMA_FLAG=""
if [ "$EMA" = "1" ]; then EMA_FLAG="--ema"; fi

# srun pattern (alloc 15150: 4 task-slots × 1 GPU × 22 CPU per node).
# `--gpu-bind=none` so NCCL can introspect peer GPUs via NVML; block:block so
# ranks are contiguous within a node; SLURM_LOCALID/PROCID/NTASKS translated
# to torch.distributed env vars.
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
TORCHINDUCTOR_CACHE_DIR=/path/to/pandwm_root/OpenDWM/torchinductor_cache,\
MASTER_ADDR=$MASTER_NODE,MASTER_PORT=$MASTER_PORT \
     bash -c "
        export LOCAL_RANK=\$SLURM_LOCALID
        export RANK=\$SLURM_PROCID
        export WORLD_SIZE=\$SLURM_NTASKS
        exec \"$PYBIN/python\" \"$OPENDWM/src/dwm_pano/evaluate_4step_dmd.py\" \
            -c \"$CFG\" -r \"$RUN\" -s \"$STEP\" $EMA_FLAG
     " \
     2>&1 | tee -a "$LOG"
