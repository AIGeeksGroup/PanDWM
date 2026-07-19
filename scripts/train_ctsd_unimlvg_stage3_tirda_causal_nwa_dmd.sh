#!/usr/bin/env bash
# Train causal pano DMD distillation (Stage 3 — Causal-Forcing++ DMD).
#
# Usage:
#   bash scripts/train_ctsd_unimlvg_stage3_tirda_causal_nwa_dmd.sh
#   RESUME_OUT=<run_dir> RESUME_FROM=N bash scripts/train_ctsd_unimlvg_stage3_tirda_causal_nwa_dmd.sh
set -euo pipefail

# Clear all outer SLURM_* env so the inner srun spawns a fresh step with our
# explicit flags. A narrow regex leaves SLURM_DISTRIBUTION / SLURM_CPU_BIND /
# SLURM_STEP_GPUS etc, which override our --distribution / --cpu-bind / GPU
# allocation and break CUDA visibility on the new step.
unset $(compgen -v | grep '^SLURM_') 2>/dev/null || true

OPENDWM=/scratch/fs201130/jn20658/tenants/clientB/workspace/OpenDWM
PYBIN=/scratch/fs201130/jn20658/tenants/clientB/workspace/miniconda3/envs/opendwm/bin
PANDWM=/scratch/fs201130/jn20658/tenants/clientB/workspace/PanDWM
ONEBEV_STITCH=/scratch/fs201130/jn20658/tenants/clientB/workspace/OneBEV/tools/stitch

CFG=$OPENDWM/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_causal_nwa_dmd.json
RUN_BASE=/data/fs201130/jn20658/tenants/clientB/OpenDWM/runs

JOBID=${JOBID:-17988}
NODELIST=${NODELIST:-n3001-[003,004]}
MASTER_NODE=${MASTER_NODE:-n3001-003}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-2}
NGPU_PER_NODE=${NGPU_PER_NODE:-4}
NCPUS_PER_TASK=${NCPUS_PER_TASK:-22}

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

# Alloc 17988 is a slot topology: 4 task-slots × 1 GPU × 22 CPU per node, with
# gres/gpu:single:1 binding. torchrun (1 task/node, 4 GPUs visible) does NOT
# work here — each worker only sees 1 GPU (device>=num_gpus crash). So spawn
# NNODES*NGPU_PER_NODE srun tasks (NGPU_PER_NODE per node); `--gpu-bind=none`
# keeps all 4 GPUs visible to every task so NCCL can introspect peer GPUs via
# NVML during FSDP bootstrap, and `LOCAL_RANK=SLURM_LOCALID` then selects the
# right GPU. Translate SLURM_* env into what torch.distributed expects and
# invoke python directly (no torchrun: srun already spawns the right procs).
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
WANDB_API_KEY=<YOUR_WANDB_API_KEY>,\
WANDB_DIR=$OUT,WANDB_CACHE_DIR=$OUT/wandb_cache,\
HF_AUTO_UPLOAD_REPO=zeyuren2002/nuscene_360_train,\
CKPT_OUT_DIR=/local/$(basename "$OUT")/checkpoints,\
HF_TOKEN=<YOUR_HF_TOKEN>,\
HF_BIN=$PYBIN/hf,\
HF_HUB_ENABLE_HF_TRANSFER=1,\
HF_HUB_DISABLE_XET=1,\
TORCHINDUCTOR_CACHE_DIR=/data/fs201130/jn20658/tenants/clientB/OpenDWM/torchinductor_cache,\
MASTER_ADDR=$MASTER_NODE,MASTER_PORT=$MASTER_PORT \
     bash -c "
        export LOCAL_RANK=\$SLURM_LOCALID
        export RANK=\$SLURM_PROCID
        export WORLD_SIZE=\$SLURM_NTASKS
        exec \"$PYBIN/python\" \"$OPENDWM/src/dwm_pano/train.py\" \
            -c \"$CFG\" -o \"$OUT\" \
            --log-steps 10 --preview-steps 500 \
            --checkpointing-steps 200 --evaluation-steps 0 \
            --wandb --wandb-project pandwm-unimlvg \
            --wandb-run-name \"$(basename "$CFG" .json)_$(date +%Y%m%d_%H%M)\" \
            ${RESUME_FROM:+--resume-from $RESUME_FROM}
     " \
     2>&1 | tee -a "$LOG"
