#!/usr/bin/env bash
# 2026-10-10: 3-GPU smoke of pipeline_v7 on HPC3 (not a training run).
# Phase 1: steps 0-10 with the adversarial terms on from step 0 (log every 2 steps, checkpoints at 5 and 10,
# milestone links); phase 2: resume from 10 to 12 (heads, head optimizer, counters).  GPU memory is sampled
# every 5 s by nvidia-smi into gpu_mem.csv.  Output: $OUT (own directory per job).
#SBATCH --job-name=pandwm_v7_smoke
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/v7_smoke_logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -euo pipefail
source /path/to/pandwm_root/pandwm_venv/bin/activate
CF=/path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008
DEMO=/path/to/pandwm_root/pandwm_code/PanDWM_long_horizon_20260928
export PYTHONPATH=$CF/src:$DEMO/vendor:$DEMO/src:/path/to/pandwm_root/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/path/to/pandwm_root/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR CAMPAIGN_DIR || true
ROOT=/path/to/pandwm_root/pandwm_runs/v7_smoke_${SLURM_JOB_ID}
OUT=$ROOT/gan
export PAUSE_FILE=$ROOT/PAUSE_REQUESTED
mkdir -p "$OUT/checkpoints" "$OUT/training_state" "$OUT/log"
CONFIG=$CF/configs/cfdmd_D_gan_v7_smoke_hpc3_3gpu.json
cp "$CONFIG" "$OUT/log/config_used_job_${SLURM_JOB_ID}.json"
echo "V7_SMOKE_START $(date -Is) node=$(hostname) job=$SLURM_JOB_ID out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader -l 5 > "$ROOT/gpu_mem.csv" &
SMI=$!
COMMON=(--log-steps 2 --preview-steps 100000000 --evaluation-steps 100000000 --checkpointing-steps 100000000)
set +e
python -m torch.distributed.run --standalone --nproc_per_node=3 "$CF/scripts/train_resumable_v7.py" \
    -c "$CONFIG" -o "$OUT" "${COMMON[@]}" --max-steps 10
RC1=$?
echo "V7_SMOKE_PHASE1_RC=$RC1 $(date -Is)"
RC2=skipped
if (( RC1 == 0 )); then
    python -m torch.distributed.run --standalone --nproc_per_node=3 "$CF/scripts/train_resumable_v7.py" \
        -c "$CONFIG" -o "$OUT" "${COMMON[@]}" --max-steps 12 --resume-from 10 --resume-data-offset 10
    RC2=$?
fi
set -e
kill "$SMI" 2>/dev/null || true
echo "V7_SMOKE_PHASE2_RC=$RC2 $(date -Is)"
python3 - "$ROOT/gpu_mem.csv" <<'PY'
import collections, sys
peak = collections.defaultdict(int)
for line in open(sys.argv[1]):
    parts = [x.strip() for x in line.split(",")]
    if len(parts) == 3 and parts[2].endswith("MiB"):
        peak[parts[1]] = max(peak[parts[1]], int(parts[2].split()[0]))
print("V7_SMOKE_PEAK_MEM_MIB", dict(peak))
PY
ls -la "$OUT/checkpoints" "$OUT/milestones" 2>/dev/null | tail -30
echo "V7_SMOKE_END $(date -Is)"
