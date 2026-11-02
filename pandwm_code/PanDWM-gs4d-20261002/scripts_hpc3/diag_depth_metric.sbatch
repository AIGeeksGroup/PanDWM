#!/usr/bin/env bash
#SBATCH --job-name=pandwm_gsdiag
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs4d_hpc3_20261017/logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
# 2026-10-17 (report section 36): decoder-vs-label depth error under two clip samplings (diag_depth_metric.py),
# for the cov 3000 decoder and the depth-pilot D1 step 1000.
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
echo "GSDIAG_START node=$(hostname) $(date -Is)"
$PY -m dwm_pano_4d.diag_depth_metric -c $GS4D/configs/gs4dd_hpc3_C.json \
  --checkpoints ${CKPTS:-$RUNS/init/gs4d_cov_3000.pth $RUNS/depth_pilot_D1/checkpoints/1000.pth} \
  --data-pkl ${PKL:-$DR/pandwm_gs_aa_experiment/inputs/nuscenes_interp_12Hz_infos_train_p2.pkl} --clips ${CLIPS:-25} \
  --output $RUNS/label_depth/diag_depth_metric_${TAG:-v1}.json
echo "GSDIAG_END rc=$? $(date -Is)"
