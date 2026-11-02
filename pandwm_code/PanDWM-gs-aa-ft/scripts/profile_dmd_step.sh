#!/usr/bin/env bash
# Profile ONE DMD cycle (1 generator + N critic steps) on the idle nodes of the
# alloc, without touching the running training. Mirrors the srun env of
# train_ctsd_unimlvg_stage3_tirda_causal_nwa_dmd.sh.
set -euo pipefail
unset $(compgen -v | grep '^SLURM_') 2>/dev/null || true

OPENDWM=/path/to/pandwm_root/workspace/OpenDWM
PYBIN=/path/to/pandwm_root/workspace/miniconda3/envs/opendwm/bin
PANDWM=/path/to/pandwm_root/workspace/PanDWM
ONEBEV_STITCH=/path/to/pandwm_root/workspace/OneBEV/tools/stitch

CFG=$OPENDWM/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_causal_nwa_dmd.json
OUT=/path/to/pandwm_root/OpenDWM/runs/_profile_tmp
mkdir -p "$OUT/log"

JOBID=${JOBID:-18374}
NODELIST=${NODELIST:-n3006-[013-014]}
MASTER_NODE=${MASTER_NODE:-n3006-013}
MASTER_PORT=${MASTER_PORT:-29600}
NNODES=${NNODES:-2}
NGPU_PER_NODE=${NGPU_PER_NODE:-4}
NCPUS_PER_TASK=${NCPUS_PER_TASK:-22}

echo "profile on $NODELIST (master=$MASTER_NODE), $NNODES x $NGPU_PER_NODE GPU"

srun --jobid="$JOBID" --overlap --nodelist="$NODELIST" \
     --nodes="$NNODES" --ntasks="$((NNODES * NGPU_PER_NODE))" \
     --ntasks-per-node="$NGPU_PER_NODE" --distribution=block:block \
     --gpus-per-node="$NGPU_PER_NODE" --gpu-bind=none \
     --cpus-per-task="$NCPUS_PER_TASK" --cpu-bind=none \
     --export=ALL,\
PYTHONPATH=$OPENDWM/src:$OPENDWM/externals/TATS/tats/fvd:$PANDWM:/path/to/pandwm_root/workspace/OpenDWM/flash-attention/hopper,\
PYTHONUNBUFFERED=1,\
ONEBEV_STITCH_PATH=$ONEBEV_STITCH,\
OMP_NUM_THREADS=1,TOKENIZERS_PARALLELISM=false,\
PATH=$PYBIN:$PATH,\
HF_HUB_ENABLE_HF_TRANSFER=1,HF_HUB_DISABLE_XET=1,\
TORCHINDUCTOR_CACHE_DIR=/path/to/pandwm_root/OpenDWM/torchinductor_cache,\
MASTER_ADDR=$MASTER_NODE,MASTER_PORT=$MASTER_PORT \
     bash -c "
        export LOCAL_RANK=\$SLURM_LOCALID
        export RANK=\$SLURM_PROCID
        export WORLD_SIZE=\$SLURM_NTASKS
        exec \"$PYBIN/python\" \"$OPENDWM/src/dwm_pano/profile_step.py\" \
            -c \"$CFG\" -o \"$OUT\" --warmup-steps ${WARMUP:-6} --profile-cycle 5 ${PROFILE_ARGS:-}
     "
