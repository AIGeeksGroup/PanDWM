#!/bin/bash
#SBATCH --job-name=epona_summary
#SBATCH --partition=<partition>
#SBATCH --nodes=1
# The cluster requires a GPU request even though this aggregation is CPU-only.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=/path/to/pandwm_root/external_runs/epona/logs/%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
R=$H/external_runs/epona/common_target_10ctx_16pred

"$H/horizondrive_venv/bin/python" "$P/scripts/epona/summarize_common.py" \
  --result HorizonDrive:0="$R/horizondrive_seed0.json" \
  --result HorizonDrive:42="$R/horizondrive_seed42.json" \
  --result HorizonDrive:1000="$R/horizondrive_seed1000.json" \
  --result PanDWM:0="$R/pandwm_seed0.json" \
  --result PanDWM:42="$R/pandwm_seed42.json" \
  --result PanDWM:1000="$R/pandwm_seed1000.json" \
  --result Epona:0="$R/seed_0/fid_fvd.json" \
  --result Epona:42="$R/seed_42/fid_fvd.json" \
  --result Epona:1000="$R/seed_1000/fid_fvd.json" \
  --real-vs-real "$R/real_vs_real.json" \
  --out "$R/results.json" \
  --markdown-out "$R/table.md"
