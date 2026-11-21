#!/bin/bash
#SBATCH --job-name=omni_summary
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.log
# Summarize the OmniNWM results over seeds 0, 42 and 1000 (the qualitative figure is common/qualitative_figure.sbatch).
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/baselines
ROOT=$H/omninwm_runs
SUMMARY=$ROOT/common149_summary
R2R=$H/external_runs/epona/common_target_10ctx_16pred/real_vs_real.json

test ! -e "$SUMMARY"
mkdir -p "$SUMMARY"

"$H/pandwm_venv/bin/python" "$P/omninwm/summarize_common.py" \
  --result "0=$ROOT/common149_dump/seed_0/common_front_seed0.json" \
  --result "42=$ROOT/common149_dump/seed_42/common_front_seed42.json" \
  --result "1000=$ROOT/common149_dump/seed_1000/common_front_seed1000.json" \
  --real-vs-real "$R2R" \
  --out "$SUMMARY/omninwm_common149_multiseed.json" \
  --markdown-out "$SUMMARY/omninwm_common149_multiseed.md"

find "$SUMMARY" -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
