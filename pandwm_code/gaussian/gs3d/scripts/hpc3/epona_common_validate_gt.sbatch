#!/bin/bash
#SBATCH --job-name=epona_gt_check
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/path/to/pandwm_root/external_runs/epona/logs/%x_%j.log
# Check that the Epona, canonical and PanDWM ground-truth targets are aligned.
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/gaussian/gs3d
R=$H/external_runs/epona/common_target_10ctx_16pred

"$H/horizondrive_venv/bin/python" \
  "$P/scripts/epona/validate_common_targets.py" \
  --epona-manifest "$R/seed_42/lossless/manifests/0000_0149.json" \
  --canonical-real "$H/horizondrive_runs/common/job/lossless/real" \
  --pandwm-real "$R/pandwm_seed42/lossless/real" \
  --out "$R/target_gt_alignment.json"
