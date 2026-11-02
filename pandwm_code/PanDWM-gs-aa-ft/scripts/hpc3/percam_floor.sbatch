#!/bin/bash
#SBATCH --job-name=pandwm_floor
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.log
# Per-camera PanoPatch-FVD floor with no model involved. See percam_floor.py.
#   sbatch --export=ALL,ITEMS=149,WIDTHS=0_593 percam_floor.sbatch
#
# WIDTHS is underscore-separated on purpose: `--export` itself splits on
# commas, so `WIDTHS=0,593` silently delivers WIDTHS=0 and drops the second
# arm. Same trap the DSL variable in diag.sbatch already works around.
set -eu
ITEMS=${ITEMS:-149}; WIDTHS=${WIDTHS:-0_593}
WIDTHS=${WIDTHS//_/,}
case "$WIDTHS" in
  *,*) : ;;
  *) echo "WARNING: WIDTHS='$WIDTHS' has a single arm; did a comma get eaten?" ;;
esac
echo "ITEMS=$ITEMS WIDTHS=$WIDTHS"
H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
CFG=$P/configs/ctsd/pandwm/h20/cd_eval_hpc3.json
RUN=$H/pandwm_runs/percam_floor_items${ITEMS}
[ -f "$CFG" ] || { echo "missing: $CFG"; exit 1; }
echo "=== $(hostname) ==="; nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$P/src:$H/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PANDWM_PANO_BLEND_W=0
mkdir -p "$RUN"
cd "$P"
$H/pandwm_venv/bin/python scripts/hpc3/percam_floor.py \
  -c "$CFG" --items "$ITEMS" --widths "$WIDTHS" -o "$RUN/floor.json"
echo "=== done ==="
