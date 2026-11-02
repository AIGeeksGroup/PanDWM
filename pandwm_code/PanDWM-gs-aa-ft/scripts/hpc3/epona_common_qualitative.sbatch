#!/bin/bash
#SBATCH --job-name=epona_qual
#SBATCH --partition=<partition>
#SBATCH --nodes=1
# The cluster's only available partitions require an explicit GPU request;
# rendering itself remains CPU-only.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=/path/to/pandwm_root/external_runs/epona/logs/%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
HD=$H/horizondrive_runs/common/job_608501/lossless
PD=$H/external_runs/epona/common_target_10ctx_16pred/pandwm_seed42/lossless
EP=$H/external_runs/epona/common_target_10ctx_16pred/seed_42/lossless
OUT=$H/external_runs/epona/common_target_10ctx_16pred/figures

for path in "$HD/real" "$HD/fake" "$PD/real" "$PD/fake" "$EP/fake"; do
  test -d "$path" || { echo "missing: $path" >&2; exit 2; }
  test "$(find "$path" -maxdepth 1 -name '*.pt' | wc -l)" -eq 149 || {
    echo "expected 149 tensors: $path" >&2
    exit 2
  }
done

source /share/anaconda3/bin/activate pytorch271
"$H/pandwm_venv/bin/python" \
  "$P/scripts/horizondrive/make_common_qualitative.py" \
  --horizon-real "$HD/real" \
  --horizon-fake "$HD/fake" \
  --pandwm-real "$PD/real" \
  --pandwm-fake "$PD/fake" \
  --epona-fake "$EP/fake" \
  --item 0:1531281439800013 \
  --item 74:1535730460896709 \
  --item 148:1542799742547751 \
  --condition-frames 3 \
  --target-start 10 \
  --target-frames 16 \
  --epona-context-frames 10 \
  --future-indices 3,7,11,15 \
  --time-label-offset-frames 0 \
  --fps 10 \
  --out-dir "$OUT"
