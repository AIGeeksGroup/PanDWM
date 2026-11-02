#!/bin/bash
#SBATCH --job-name=gem_ckpt_audit
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
GEM=$H/external_baselines/GEM
OUT=$H/gem_runs/checkpoint_audit_v2.json
test ! -e "$OUT"
export HF_HOME=$H/external_models/hf_cache
export TORCH_HOME=$H/.cache/torch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$P/scripts/gem/optional_stubs:$GEM
"$H/vista_venv/bin/python" "$P/scripts/gem/audit_checkpoint.py" \
  --config "$GEM/configs/training/stage2.yaml" \
  --checkpoint "$H/external_models/gem/gem.safetensors" \
  --out "$OUT"
