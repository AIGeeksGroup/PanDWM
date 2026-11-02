#!/bin/bash
#SBATCH --job-name=pandwm_n6bridge
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.log
# Band-limited six-real-camera protocol bridge.
#   sbatch --export=ALL,MODE=floor,ITEMS=149 nuscenes6_bridge.sbatch
# Single GPU on purpose: the whole metric family this compares against was
# measured single-GPU, and changing the topology changes the instrument.
set -eu
MODE=${MODE:-floor}; ITEMS=${ITEMS:-149}; FRAMES=${FRAMES:-32}
RH=${RH:-360}; RW=${RW:-640}
H=/path/to/pandwm_root
RUN=$H/pandwm_runs/n6bridge_${MODE}_items${ITEMS}_f${FRAMES}_${RH}x${RW}
echo "=== $(hostname) ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader || true
mkdir -p "$RUN"
export PYTHONPATH=$H/pandwm_code/PanDWM/src:$H/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd $H/pandwm_code/PanDWM
$H/pandwm_venv/bin/python scripts/hpc3/nuscenes6_bridge_run.py \
  --config configs/ctsd/pandwm/h20/cd_eval_hpc3.json \
  --mode "$MODE" --items "$ITEMS" --frames "$FRAMES" \
  --render-height "$RH" --render-width "$RW" \
  --i3d $H/pandwm_code/TATS_fvd/i3d_pretrained_400.pt \
  --out "$RUN/bridge.json"
echo "=== done ==="
