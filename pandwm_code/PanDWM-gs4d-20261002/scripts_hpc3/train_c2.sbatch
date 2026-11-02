#!/usr/bin/env bash
#SBATCH --job-name=pandwm_gs4dC2
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=14:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs4d_hpc3_20261017/logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
# 2026-10-20 C2 (report section 41): the 4D decoder retrained with Depth Anything 3 pseudo-depth on every batch,
# configs/gs4dd_hpc3_C2.json (make_c2_config.py), for STEPS (default 5000) steps into $RUNS/${TAG:-formal_C2},
# checkpoints every CKPT_STEPS (default 1000), all kept.  Smoke: STEPS=40 CKPT_STEPS=40 TAG=smoke_C2.
set -uo pipefail
STEPS=${STEPS:-5000}
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/da3_env_hpc3.sh
OUT=$RUNS/${TAG:-formal_C2}
echo "GS4DC2_START tag=${TAG:-formal_C2} steps=$STEPS node=$(hostname) $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY $GSAA/src/dwm_pano/train.py -c $GS4D/configs/gs4dd_hpc3_C2.json -o $OUT --max-steps $STEPS --log-steps ${LOG_STEPS:-20} \
  --preview-steps 100000000 --evaluation-steps 100000000 --checkpointing-steps ${CKPT_STEPS:-1000} --keep-last-ckpts 20
rc=$?
echo "GS4DC2_END rc=$rc $(date -Is)"
exit $rc
