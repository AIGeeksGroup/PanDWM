#!/bin/bash
#SBATCH --job-name=vista_prepare_front
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
OUT=$H/vista_common_front149
test ! -e "$OUT" || { echo "refusing to overwrite $OUT" >&2; exit 2; }
export PYTHONPATH=$P/src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd "$P"
$H/pandwm_venv/bin/python scripts/vista/prepare_common_front.py \
  --common-pkl $H/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl \
  --pano-rgb-dir $H/pandwm_data/rgb \
  --out "$OUT" \
  --start-offset 9 \
  --frames 25 \
  --height 576 \
  --width 1024
find "$OUT" -type f | wc -l
du -sh "$OUT"
echo "completed out=$OUT"
