#!/bin/bash
#SBATCH --job-name=common_summary
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.log
set -euo pipefail

HORIZON_SEED0_JOB=${HORIZON_SEED0_JOB:?}
HORIZON_SEED1000_JOB=${HORIZON_SEED1000_JOB:?}
PANDWM_SEED0_JOB=${PANDWM_SEED0_JOB:?}
PANDWM_SEED1000_JOB=${PANDWM_SEED1000_JOB:?}
H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
OUT_DIR=$H/pandwm_runs/common_multiseed_0_42_1000
mkdir -p "$OUT_DIR"

source /share/anaconda3/bin/activate pytorch271
cd "$P"
"$H/pandwm_venv/bin/python" \
  scripts/horizondrive/summarize_common_multiseed.py \
  --horizon "0=$H/horizondrive_runs/common/job_${HORIZON_SEED0_JOB}/common_front_fvd.json" \
  --horizon "42=$H/horizondrive_runs/common/job_608501/common_front_fvd.json" \
  --horizon "1000=$H/horizondrive_runs/common/job_${HORIZON_SEED1000_JOB}/common_front_fvd.json" \
  --pandwm "0=$P/scripts/hpc3/pandwm_common_${PANDWM_SEED0_JOB}.log" \
  --pandwm "42=$P/scripts/hpc3/pandwm_common_608695.log" \
  --pandwm "1000=$P/scripts/hpc3/pandwm_common_${PANDWM_SEED1000_JOB}.log" \
  --real-vs-real "$H/horizondrive_runs/common/job_608501/real_vs_real.json" \
  --out "$OUT_DIR/results.json" \
  --markdown-out "$OUT_DIR/table.md"

echo "completed: $OUT_DIR"
