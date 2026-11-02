#!/usr/bin/env bash
# 1-GPU history-blind discriminator probe (read-only on the training run): feature extraction, then the
# CPU/GPU probe fits on the saved features.
# Submit (the log directory must exist before sbatch opens the --output file):
#   mkdir -p /path/to/pandwm_root/pandwm_runs/cfdmd_20261008_A/probe_d_target_1585
#   cd /path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008
#   sbatch scripts/probe_d_target.sbatch
# Overrides via env: STEP, OUT, NCLIPS, NVAL, BUDGET_MIN, EXTRA_ARGS, FIT_ARGS.
#SBATCH --job-name=pandwm_probe_d
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/cfdmd_20261008_A/probe_d_target_1585/%x_%j.log
set -euo pipefail
source /path/to/pandwm_root/pandwm_venv/bin/activate
CF=/path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008
DEMO=/path/to/pandwm_root/pandwm_code/PanDWM_long_horizon_20260928
export PYTHONPATH=$CF/src:$DEMO/vendor:$DEMO/src:/path/to/pandwm_root/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/path/to/pandwm_root/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID}
# Never touch the training run: no pause file, no checkpoint redirection, no uploads.
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR PAUSE_FILE CAMPAIGN_DIR || true

STEP=${STEP:-1585}
OUT=${OUT:-/path/to/pandwm_root/pandwm_runs/cfdmd_20261008_A/probe_d_target_${STEP}}
NCLIPS=${NCLIPS:-48}
NVAL=${NVAL:-16}
BUDGET_MIN=${BUDGET_MIN:-135}
mkdir -p "$OUT"

echo "PROBE_START $(date -Is) node=$(hostname) job=${SLURM_JOB_ID:-none} step=$STEP out=$OUT"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || {
    echo "CUDA_SELF_CHECK_FAILED nvidia-smi node=$(hostname)"; exit 3; }
python - <<'PY' || { echo "CUDA_SELF_CHECK_FAILED torch node=$(hostname)"; exit 3; }
import socket, torch
assert torch.cuda.is_available() and torch.cuda.device_count() >= 1, "no CUDA device"
x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
y = (x @ x).float().sum()
torch.cuda.synchronize()
assert torch.isfinite(y), "non-finite matmul"
free, total = torch.cuda.mem_get_info()
assert free > 70 * 2**30, f"only {free / 2**30:.1f} GiB free"
print("CUDA_OK", socket.gethostname(), torch.cuda.get_device_name(0),
      f"free={free / 2**30:.1f}GiB", flush=True)
PY

set +e
python "$CF/scripts/probe_d_target.py" \
    --config "$CF/configs/cfdmd_A_gan_hpc3_8gpu.json" \
    --ckpt-dir /path/to/pandwm_root/pandwm_runs/cfdmd_20261008_A/gan/milestones \
    --step "$STEP" --generator raw --output-dir "$OUT" \
    --num-clips "$NCLIPS" --val-clips "$NVAL" --time-budget-min "$BUDGET_MIN" ${EXTRA_ARGS:-}
RC=$?
echo "EXTRACT_DONE $(date -Is) rc=$RC"
if [[ -f "$CF/scripts/probe_d_target_fit.py" ]]; then
    python "$CF/scripts/probe_d_target_fit.py" --input-dir "$OUT" --device cuda ${FIT_ARGS:-}
    echo "FIT_DONE $(date -Is) rc=$?"
else
    echo "FIT_SKIPPED: $CF/scripts/probe_d_target_fit.py not present"
fi
set -e
echo "PROBE_DONE $(date -Is) rc=$RC"
exit $RC
