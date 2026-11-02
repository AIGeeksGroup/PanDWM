#!/usr/bin/env bash
#SBATCH --job-name=pandwm_gsr2
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
# 2026-10-17 (report section 34): GS renders v2 of the review clips with export_gs_v2_clips.py.
#   TAG=<out dir name> [CKPT=<decoder>] [START=0 END=149] [MAX_FRAMES=n] [EXTRA="--edge-rel-h 0.1 ..."]
# Clip latents default to the locked model's Dev-149 outputs (v84i4250).
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
CK=${CKPT:-$RUNS/init/gs4d_cov_3000.pth}
CLIPS=${CLIPS_ROOT:-$DR/pandwm_runs/eval149_hpc3_v83_20261012/v84i4250}
OUT=$RUNS/render_v2/${TAG:?TAG}
mkdir -p $OUT
echo "GSR2_START tag=$TAG ckpt=$CK clips=${START:-0}..${END:-149} extra=[${EXTRA:-}] node=$(hostname) $(date -Is)"
$PY -m dwm_pano.export_gs_v2_clips -c $GS4D/configs/g2_render_hpc3.json \
  --data-pkl $DR/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl --checkpoint $CK \
  --clips-root $CLIPS --start ${START:-0} --end ${END:-149} --out-root $OUT \
  ${MAX_FRAMES:+--max-frames $MAX_FRAMES} ${EXTRA:-}
echo "GSR2_END rc=$? $(date -Is)"
