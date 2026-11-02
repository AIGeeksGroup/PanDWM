#!/usr/bin/env bash
# 2026-10-12: answer-leak test (eval_v7/leak_test_eval149.py) for the v8.2 step-2000 evaluation videos.  Same code
# tree, config, checkpoint, seed and settings as eval149_hpc3_gen.sbatch; 2 clips x 5 arms on 1 GPU (about 20 min).
# The test script is copied to $OUT first, so the frozen eval tree $C stays unchanged.
#   sbatch scripts/leak_test_hpc3.sbatch
#SBATCH --job-name=pandwm_leak
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=01:30:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/leak_test_20261012/leak_%j.log
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
CFG=$C/cd_eval_frozen149_hpc3.json
OUT=$DR/pandwm_runs/leak_test_20261012
PY=$DR/pandwm_venv/bin/python
R=$DR/pandwm_runs
CK=$R/cfdmd_20261010_G/gan/milestones/2000_ema.pth
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29871
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR PANDWM_HISTORY_GUIDANCE || true
echo "LEAK_START $(date -Is) node=$(hostname) ckpt=$CK"
[ -f "$OUT/leak_test_eval149.py" ] || { echo "LEAK_SCRIPT_MISSING"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$C"
"$PY" lowmem_launch.py "$OUT/leak_test_eval149.py" --base-config "$CFG" --checkpoint "$CK" --out "$OUT" \
  --ref-root "$R/eval149_hpc3_20261011/v82g2000" --clips 0,74 --seed 42
echo "LEAK_END $(date -Is) rc=$?"
