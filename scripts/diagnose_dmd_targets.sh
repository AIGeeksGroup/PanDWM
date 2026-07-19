#!/usr/bin/env bash
# Dump DMD generator/critic/teacher x0 predictions as decoded PNGs for a ckpt.
#   RUN=<dmd_run_dir> STEP=N OUT=<dir> JOBID=.. NODELIST=.. bash scripts/diagnose_dmd_targets.sh
set -euo pipefail
unset $(compgen -v | grep -E '^SLURM_(STEP|NTASKS|NPROCS|NNODES|NODEID|LOCALID|PROCID|GTIDS|JOB_NODELIST|NODELIST|TASKS_PER_NODE|TOPOLOGY)') 2>/dev/null || true

OPENDWM=/scratch/fs201130/jn20658/tenants/clientB/workspace/OpenDWM
PYBIN=/scratch/fs201130/jn20658/tenants/clientB/workspace/miniconda3/envs/opendwm/bin
PANDWM=/scratch/fs201130/jn20658/tenants/clientB/workspace/PanDWM
ONEBEV_STITCH=/scratch/fs201130/jn20658/tenants/clientB/workspace/OneBEV/tools/stitch

CFG=$OPENDWM/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_causal_nwa_dmd.json
RUN=${RUN:?set RUN=<dmd run dir>}
STEP=${STEP:?set STEP=N}
OUT=${OUT:?set OUT=<output dir>}

JOBID=${JOBID:?}; NODELIST=${NODELIST:?}; MASTER_NODE=${MASTER_NODE:-$NODELIST}
MASTER_PORT=${MASTER_PORT:-29571}; NNODES=${NNODES:-1}; NGPU_PER_NODE=${NGPU_PER_NODE:-4}
NCPUS_PER_TASK=${NCPUS_PER_TASK:-22}

srun --jobid="$JOBID" --overlap --nodelist="$NODELIST" \
     --nodes="$NNODES" --ntasks="$((NNODES * NGPU_PER_NODE))" \
     --ntasks-per-node="$NGPU_PER_NODE" --distribution=block:block \
     --gpus-per-node="$NGPU_PER_NODE" --gpu-bind=none \
     --cpus-per-task="$NCPUS_PER_TASK" --cpu-bind=none \
     --export=ALL,\
PYTHONPATH=$OPENDWM/src:$OPENDWM/externals/TATS/tats/fvd:$PANDWM,\
PYTHONUNBUFFERED=1,ONEBEV_STITCH_PATH=$ONEBEV_STITCH,\
OMP_NUM_THREADS=1,TOKENIZERS_PARALLELISM=false,PATH=$PYBIN:$PATH,\
TORCHINDUCTOR_CACHE_DIR=/data/fs201130/jn20658/tenants/clientB/OpenDWM/torchinductor_cache,\
MASTER_ADDR=$MASTER_NODE,MASTER_PORT=$MASTER_PORT \
     bash -c "
        export LOCAL_RANK=\$SLURM_LOCALID RANK=\$SLURM_PROCID WORLD_SIZE=\$SLURM_NTASKS
        exec \"$PYBIN/python\" \"$OPENDWM/src/dwm_pano/diagnose_dmd_targets.py\" \
            -c \"$CFG\" -r \"$RUN\" -s \"$STEP\" -o \"$OUT\"
     "
