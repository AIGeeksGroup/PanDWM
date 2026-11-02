#!/bin/bash
#SBATCH --job-name=omni_figure
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
ROOT=$H/omninwm_runs
SUMMARY=$ROOT/common149_summary
FIGURES=$ROOT/common149_figures
HD=$H/horizondrive_runs/common/job_608501/lossless
PD=$H/external_runs/epona/common_target_10ctx_16pred/pandwm_seed42/lossless
EP=$H/external_runs/epona/common_target_10ctx_16pred/seed_42/lossless
VISTA=$H/vista_runs/common_seed42/fake
GEM=$H/gem_runs/common149_dump/seed_42/fake
OMNI=$ROOT/common149_dump/seed_42/fake
R2R=$H/external_runs/epona/common_target_10ctx_16pred/real_vs_real.json
AUDIT=$ROOT/checkpoint_audit.json

test ! -e "$SUMMARY"
test ! -e "$FIGURES"
mkdir -p "$SUMMARY"

"$H/pandwm_venv/bin/python" "$P/scripts/omninwm/summarize_common.py" \
  --result "0=$ROOT/common149_dump/seed_0/common_front_seed0.json" \
  --result "42=$ROOT/common149_dump/seed_42/common_front_seed42.json" \
  --result "1000=$ROOT/common149_dump/seed_1000/common_front_seed1000.json" \
  --real-vs-real "$R2R" \
  --checkpoint-audit "$AUDIT" \
  --out "$SUMMARY/omninwm_common149_multiseed.json" \
  --markdown-out "$SUMMARY/omninwm_common149_multiseed.md"

"$H/pandwm_venv/bin/python" \
  "$P/scripts/horizondrive/make_common_qualitative.py" \
  --horizon-real "$HD/real" \
  --horizon-fake "$HD/fake" \
  --pandwm-real "$PD/real" \
  --pandwm-fake "$PD/fake" \
  --epona-fake "$EP/fake" \
  --vista-fake "$VISTA" \
  --gem-fake "$GEM" \
  --omni-fake "$OMNI" \
  --item 0:1531281439800013 \
  --item 74:1535730460896709 \
  --item 148:1542799742547751 \
  --condition-frames 3 \
  --target-start 10 \
  --target-frames 16 \
  --epona-context-frames 10 \
  --vista-context-frames 1 \
  --gem-context-frames 1 \
  --omni-context-frames 0 \
  --future-indices 3,7,11,15 \
  --time-label-offset-frames 0 \
  --fps 10 \
  --out-dir "$FIGURES"

find "$SUMMARY" "$FIGURES" -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
