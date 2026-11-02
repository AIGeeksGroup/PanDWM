#!/usr/bin/env bash
#SBATCH --job-name=pandwm_gs4dd_eval
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs4d_hpc3_20261017/logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
# 2026-10-17: held-out-frame eval (evaluate_4d.py, 25 val clips, gap 4) of CHECKPOINT into $RUNS/eval/<NAME>.json.
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
mkdir -p $RUNS/eval
echo "GS4DD_EVAL_START name=${NAME:?} ckpt=${CHECKPOINT:?} $(date -Is)"
$PY -m dwm_pano_4d.evaluate_4d -c $GS4D/configs/gs4dd_hpc3_C.json --checkpoint $CHECKPOINT \
  --data-pkl $DR/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl --clips ${CLIPS:-25} \
  --output $RUNS/eval/$NAME.json
echo "GS4DD_EVAL_END rc=$? $(date -Is)"
