#!/usr/bin/env bash
# Eval PanDWM causal ckpt with the matching eval16gpu config.
#
# Usage:
#   bash scripts/eval_ctsd_unimlvg_stage3_tirda_causal_nwa.sh
#   STEP=12000 OUT=<run_dir> bash scripts/eval_ctsd_unimlvg_stage3_tirda_causal_nwa.sh
set -euo pipefail

# Clear any outer srun-step env so the inner srun reuses our jobid cleanly.
unset $(compgen -v | grep -E '^SLURM_(STEP|NTASKS|NPROCS|NNODES|NODEID|LOCALID|PROCID|GTIDS|JOB_NODELIST|NODELIST|TASKS_PER_NODE|TOPOLOGY)') 2>/dev/null || true

OPENDWM=/path/to/pandwm_root/workspace/OpenDWM
PYBIN=/path/to/pandwm_root/workspace/miniconda3/envs/opendwm/bin
PANDWM=/path/to/pandwm_root/workspace/PanDWM
ONEBEV_STITCH=/path/to/pandwm_root/workspace/OneBEV/tools/stitch

CFG=$OPENDWM/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_causal_nwa_eval16gpu.json
OUT=${OUT:-/path/to/pandwm_root/OpenDWM/runs/train_ctsd_unimlvg_stage3_tirda_causal_nwa_20260501_182010}
STEP=${STEP:-12000}

JOBID=${JOBID:-15151}
NODELIST=${NODELIST:-n3001-[004,007]}
MASTER_NODE=${MASTER_NODE:-n3001-004}
MASTER_PORT=${MASTER_PORT:-29501}
NNODES=${NNODES:-2}
NGPU_PER_NODE=${NGPU_PER_NODE:-4}
NCPUS_PER_TASK=${NCPUS_PER_TASK:-22}

LOG=$OPENDWM/outputs/eval_$(basename "$CFG" .json)_step${STEP}_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$(dirname "$LOG")"

echo "==== eval ($(basename "$CFG" .json), step=$STEP) ====" | tee "$LOG"
echo "jobid:  $JOBID  ($NNODES x $NGPU_PER_NODE GPU)"        | tee -a "$LOG"
echo "nodes:  $NODELIST (master=$MASTER_NODE)"               | tee -a "$LOG"
echo "config: $CFG"                                          | tee -a "$LOG"
echo "output: $OUT"                                          | tee -a "$LOG"
echo "ckpt:   $OUT/checkpoints/${STEP}.pth"                  | tee -a "$LOG"

# New srun pattern (matches alloc topology of 4 task-slots × 1 GPU per node):
# `--gpu-bind=none` so NCCL can introspect peer GPUs, `block:block` for contiguous
# ranks, `--cpu-bind=none` to avoid NUMA-fragmented binding failures, env vars
# translated to torch.distributed (no torchrun).
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
        exec \"$PYBIN/python\" \"$OPENDWM/src/dwm_pano/evaluate.py\" \
            -c \"$CFG\" -o \"$OUT\" --resume-from \"$STEP\"
     " \
     2>&1 | tee -a "$LOG"
