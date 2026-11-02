#!/usr/bin/env bash
#SBATCH --job-name=pandwm_da3diag
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs4d_hpc3_20261017/logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
# 2026-10-19 (report section 40): DA3 range vs depth label on one p2 clip (diag_da3_frame.py).  CLIPS=<i,j,...> TAG=<name>
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
source $GS4D/scripts_hpc3/da3_env_hpc3.sh
$PY -m dwm_pano_4d.diag_da3_frame -c $GS4D/configs/gs4dd_hpc3_C.json --checkpoint $RUNS/init/gs4d_cov_3000.pth \
  --data-pkl $DR/pandwm_gs_aa_experiment/inputs/nuscenes_interp_12Hz_infos_train_p2.pkl --clips ${CLIPS:-0} \
  --da3-src $DA3_SRC --da3-anyview $DA3_ANYVIEW --da3-metric $DA3_METRIC --out $RUNS/label_depth/da3_diag/${TAG:-diag}
rc=$?
echo "DA3DIAG_END rc=$rc $(date -Is)"
exit $rc
