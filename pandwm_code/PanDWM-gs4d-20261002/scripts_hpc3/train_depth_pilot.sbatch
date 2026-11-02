#!/usr/bin/env bash
#SBATCH --job-name=pandwm_gs4dd
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs4d_hpc3_20261017/logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
# 2026-10-17 depth pilot (report section 34): ARM=C|D1|D2 trains configs/gs4dd_hpc3_<ARM>.json from the 4D coverage
# decoder step 3000 for STEPS (default 1000) steps into $RUNS/depth_pilot_<ARM>.
set -uo pipefail
ARM=${ARM:?ARM}; STEPS=${STEPS:-1000}
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
OUT=$RUNS/depth_pilot_$ARM
echo "GS4DD_START arm=$ARM steps=$STEPS node=$(hostname) $(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader
$PY $GSAA/src/dwm_pano/train.py -c $GS4D/configs/gs4dd_hpc3_$ARM.json -o $OUT --max-steps $STEPS --log-steps 20 \
  --preview-steps 100000000 --evaluation-steps 100000000 --checkpointing-steps 500 --keep-last-ckpts 5
rc=$?
echo "GS4DD_END arm=$ARM rc=$rc $(date -Is)"
exit $rc
