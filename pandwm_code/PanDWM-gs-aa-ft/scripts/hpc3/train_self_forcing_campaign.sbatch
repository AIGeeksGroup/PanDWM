#!/bin/bash
#SBATCH --job-name=pandwm_sf_full
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
cd "$CODE_ROOT"
echo "campaign_job=$SLURM_JOB_ID host=$(hostname)"
# Verify that 64-frame FID/FVD fits the same single H100 before committing
# several days to the paired training campaign.
ARM=baseline HORIZON=long SEED=0 ITEMS=2 RUN_TAG=preflight_ \
  bash scripts/hpc3/eval_self_forcing_stage2.sbatch
ARM=warmup STEPS=2000 SEED=0 RUN_TAG=production \
  bash scripts/hpc3/train_self_forcing_stage2.sbatch
ARM=control STEPS=8000 SEED=0 RUN_TAG=production \
  bash scripts/hpc3/train_self_forcing_stage2.sbatch
ARM=self STEPS=8000 SEED=0 RUN_TAG=production \
  bash scripts/hpc3/train_self_forcing_stage2.sbatch
echo "campaign complete: shared 2000 + paired 8000/8000"
