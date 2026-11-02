#!/usr/bin/env bash
# 2026-10-15: student closed-loop history cache on train clips for the teacher fine-tune v3 (eval_v7/cache_student_history.py).
#   CKPT=<student ckpt> CACHE=<dir> sbatch --array=0-7%8 scripts/cache_student_history.sbatch      (SHARDS = array size)
#SBATCH --job-name=pandwm_scache
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs_eval149/%x_%A_%a.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
CF=$DR/pandwm_code/PanDWM-cfdmd-A-20261008
PY=$DR/pandwm_venv/bin/python
CK=${CKPT:-$DR/pandwm_runs/cfdmd_20261013_I/gan/milestones/4250_ema.pth}
CACHE=${CACHE:-$DR/pandwm_runs/teacher_v3_20261015/student_cache_w4_v84i4250}
SHARDS=${SHARDS:-8}; W=${MAX_TARGET_WINDOW:-4}; MAXI=${MAX_ITEMS:-0}
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=12 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR PANDWM_HISTORY_GUIDANCE || true
[[ -f "$CK" ]] || { echo "CKPT_MISSING $CK"; exit 1; }
cd "$C" || exit 1
echo "SCACHE_START $(date -Is) node=$(hostname) shard=${SLURM_ARRAY_TASK_ID:-0}/$SHARDS ckpt=$CK cache=$CACHE"
export MASTER_PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$PY" lowmem_launch.py cache_student_history.py --base-config "$C/cd_eval_frozen149_hpc3.json" \
  --train-config "$CF/configs/cfdmd_I_gan_v8_4_hpc3_8gpu.json" --checkpoint "$CK" --cache-dir "$CACHE" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --shards "$SHARDS" --max-target-window "$W" --max-items "$MAXI"
rc=$?
echo "SCACHE_END rc=$rc $(date -Is)"
exit $rc
