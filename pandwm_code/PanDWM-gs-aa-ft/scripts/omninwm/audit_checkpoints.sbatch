#!/bin/bash
#SBATCH --job-name=omni_audit
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
REPO=$H/external_baselines/OmniNWM
V=$H/omninwm_venv
ASSETS=$H/external_models/omninwm
OUT=$H/omninwm_runs/checkpoint_audit.json

test "$(sed -n '1p' "$REPO/.git/refs/heads/master")" = \
  "8e2333ee0dfb7be7a477e51a97e644bf7ec53039"
test ! -e "$OUT"
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
"$V/bin/python" "$P/scripts/omninwm/audit_checkpoints.py" \
  --config "$REPO/configs/inference/infer.py" \
  --dit "$ASSETS/release/ckpt/dit/model" \
  --vae "$ASSETS/hunyuan_vae.safetensors" \
  --out "$OUT"
