#!/usr/bin/env bash
#SBATCH --job-name=pandwm_gs4dF
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=10:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs4d_hpc3_20261017/logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
# 2026-10-18 formal depth-supervised 4D decoder (report section 38): configs/gs4dd_hpc3_F1.json (make_formal_config.py)
# for STEPS (default 5000) steps into $RUNS/${TAG:-formal_F1}, checkpoints every 500 steps, all kept.
set -uo pipefail
STEPS=${STEPS:-5000}
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
OUT=$RUNS/${TAG:-formal_F1}
echo "GS4DF_START tag=${TAG:-formal_F1} steps=$STEPS node=$(hostname) $(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader
$PY $GSAA/src/dwm_pano/train.py -c $GS4D/configs/gs4dd_hpc3_F1.json -o $OUT --max-steps $STEPS --log-steps 20 \
  --preview-steps 100000000 --evaluation-steps 100000000 --checkpointing-steps ${CKPT_STEPS:-500} --keep-last-ckpts 20
rc=$?
echo "GS4DF_END rc=$rc $(date -Is)"
exit $rc
