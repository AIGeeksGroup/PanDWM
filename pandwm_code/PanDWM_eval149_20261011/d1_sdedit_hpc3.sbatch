#!/usr/bin/env bash
#SBATCH --job-name=pandwm_d1sde
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
# 2026-10-20 D1 step 2 (PanDWM-gs4d-20261002/GS_GAP_ANALYSIS_20261020.md): SDEdit of shifted GS pano renders with the
# locked v84i4250 EMA student (d1_sdedit.py), environment of verify_clips_runner.sbatch.
#   CLIPS=2,60 SHIFTS=0,1,2 STARTS=2,3 sbatch d1_sdedit_hpc3.sbatch
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
CFG=$C/cd_eval_frozen149_hpc3.json
PY=$DR/pandwm_venv/bin/python
R=$DR/pandwm_runs
CK=$R/cfdmd_20261013_I/gan/milestones/4250_ema.pth
D1=$R/d1_sdedit_20261020
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID} % 20000 ))
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR PANDWM_HISTORY_GUIDANCE || true
echo "D1SDE_START clips=${CLIPS:?} shifts=${SHIFTS:-0,1,2} starts=${STARTS:-2,3} node=$(hostname) $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$C" || exit 1
"$PY" lowmem_launch.py d1_sdedit.py --base-config "$CFG" --checkpoint "$CK" --manifest dev149_manifest_20261018.json \
  --renders $D1/renders --out-root $D1/out --clips "$CLIPS" --shifts "${SHIFTS:-0,1,2}" --starts "${STARTS:-2,3}"
rc=$?
echo "D1SDE_END rc=$rc $(date -Is)"
exit $rc
