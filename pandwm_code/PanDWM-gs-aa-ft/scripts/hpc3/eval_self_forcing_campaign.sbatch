#!/bin/bash
#SBATCH --job-name=pandwm_sf_metrics
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
echo "evaluation_job=$SLURM_JOB_ID host=$(hostname)"
for horizon in short long; do
  for seed in 0 42 1000; do
    for arm in baseline control self; do
      ARM=$arm HORIZON=$horizon SEED=$seed \
        bash scripts/hpc3/eval_self_forcing_stage2.sbatch
    done
  done
done
echo "evaluation complete: three arms x three seeds x two horizons"
