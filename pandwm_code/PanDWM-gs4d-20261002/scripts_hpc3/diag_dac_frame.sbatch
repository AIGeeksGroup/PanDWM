#!/usr/bin/env bash
#SBATCH --job-name=pandwm_dacdiag
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs4d_hpc3_20261017/logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
# 2026-10-18 (report section 37): DAC range vs depth label on a few real panos (diag_dac_frame.py).
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
D=/path/to/pandwm_root
$PY -m dwm_pano_4d.diag_dac_frame --timestamps ${TS:?} --rgb-dir $D/pandwm_full_train_700_20260929/rgb \
  --depth-dir $D/pandwm_data/train/depth --dac-root $D/pandwm_code/third_party/depth_any_camera \
  --out $RUNS/label_depth/dac_diag
echo "DACDIAG_END rc=$? $(date -Is)"
