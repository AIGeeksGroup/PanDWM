#!/bin/bash
#SBATCH --job-name=pandwm_sf_parallel
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=7-00:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

CODE_ROOT=/path/to/pandwm_root/pandwm_code/PanDWM_self_forcing_20260915
MARKER=/path/to/pandwm_root/pandwm_runs/self_forcing_stage2_20260915/production/.self_parallel_owner
test -f "$MARKER" || { echo "missing delegation marker: $MARKER" >&2; exit 2; }
echo "parallel Self-Forcing job=$SLURM_JOB_ID host=$(hostname) marker=$(cat "$MARKER")"
cd "$CODE_ROOT"
SELF_PARALLEL_OWNER=1 ARM=self STEPS=8000 SEED=0 RUN_TAG=production \
  bash scripts/hpc3/train_self_forcing_stage2.sbatch
