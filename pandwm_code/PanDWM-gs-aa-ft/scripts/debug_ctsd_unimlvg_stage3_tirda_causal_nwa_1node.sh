#!/usr/bin/env bash
# 1-node debug — from-scratch warm-start, verify full data + train + save +
# HF upload + prune chain at small intervals on jobid 12803 / n3001-016.
#
# Compressed schedule (vs production 4-node):
#   log:       10 steps   (loss/dtype trace)
#   audit:     20 steps   (DWM_AUDIT_STEPS — prints finite/dtype per step)
#   mem-probe: step 10    (one-shot allocated/reserved/peak)
#   preview:   500 steps  (rarely fires in a short debug)
#   ckpt:      100 steps  (one full save→HF upload→prune cycle)
#   eval:      off
#
# Usage:
#   bash scripts/debug_ctsd_unimlvg_stage3_tirda_causal_nwa_1node.sh
set -euo pipefail

unset $(compgen -v | grep -E '^SLURM_(STEP|NTASKS|NPROCS|NNODES|NODEID|LOCALID|PROCID|GTIDS|JOB_NODELIST|NODELIST|TASKS_PER_NODE|TOPOLOGY)') 2>/dev/null || true

OPENDWM=/path/to/pandwm_root/workspace/OpenDWM
PYBIN=/path/to/pandwm_root/workspace/miniconda3/envs/opendwm/bin
PANDWM=/path/to/pandwm_root/workspace/PanDWM
ONEBEV_STITCH=/path/to/pandwm_root/workspace/OneBEV/tools/stitch

CFG=$OPENDWM/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_causal_nwa_debug.json
RUN_BASE=/path/to/pandwm_root/OpenDWM/runs

JOBID=${JOBID:-15150}
NODELIST=${NODELIST:-n3001-013}
MASTER_NODE=${MASTER_NODE:-n3001-013}
MASTER_PORT=${MASTER_PORT:-29502}
NNODES=${NNODES:-1}
NGPU_PER_NODE=${NGPU_PER_NODE:-4}
NCPUS_PER_TASK=${NCPUS_PER_TASK:-64}

OUT=$RUN_BASE/debug_$(basename "$CFG" .json)_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT/checkpoints" "$OUT/optimizer" "$OUT/preview" "$OUT/log"
LOG=$OUT/log/$(basename "$CFG" .json)_$(date +%Y%m%d_%H%M%S).log
cp "$CFG" "$OUT/log/config_used.json"

echo "==== debug 1-node launch ($(basename "$CFG" .json)) ====" | tee "$LOG"
echo "jobid:  $JOBID  ($NNODES x $NGPU_PER_NODE GPU)"           | tee -a "$LOG"
echo "nodes:  $NODELIST (master=$MASTER_NODE)"                  | tee -a "$LOG"
echo "config: $CFG"                                             | tee -a "$LOG"
echo "output: $OUT"                                             | tee -a "$LOG"

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
TORCHINDUCTOR_CACHE_DIR=/path/to/pandwm_root/OpenDWM/torchinductor_cache,\
DWM_AUDIT_STEPS=20,DWM_MEM_PROBE_STEP=10 \
     "$PYBIN/torchrun" \
        --nnodes="$NNODES" --nproc-per-node="$NGPU_PER_NODE" \
        --rdzv-id="pandwm_debug_$(date +%s)" --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_NODE:$MASTER_PORT" \
        "$OPENDWM/src/dwm_pano/train.py" \
        -c "$CFG" -o "$OUT" \
        --log-steps 10 --preview-steps 500 \
        --checkpointing-steps 100 --evaluation-steps 0 \
        --wandb --wandb-project pandwm-unimlvg \
        --wandb-run-name "debug_$(basename "$CFG" .json)_$(date +%Y%m%d_%H%M)" \
     2>&1 | tee -a "$LOG"
