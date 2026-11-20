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
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs4d_hpc3/logs/%x_%j.log
# Render the review clips with export_gs_v2_clips.py.
#   TAG=<out dir name> [CKPT=<decoder>] [START=0 END=149] [MAX_FRAMES=n] [EXTRA="--edge-rel-h 0.1 ..."]
# CLIPS_ROOT selects the clip latents.
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/gaussian/gs4d/scripts_hpc3/common_env_hpc3.sh
CK=${CKPT:-$RUNS/init/gs4d_cov_3000.pth}
CLIPS=${CLIPS_ROOT:-$DR/pandwm_runs/eval149_hpc3_v83/v84i4250}
OUT=$RUNS/render_v2/${TAG:?TAG}
mkdir -p $OUT
echo "GSR2_START tag=$TAG ckpt=$CK clips=${START:-0}..${END:-149} extra=[${EXTRA:-}] node=$(hostname) $(date -Is)"
$PY -m dwm_pano.export_gs_v2_clips -c $GS4D/configs/g2_render_hpc3.json \
  --data-pkl $DR/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl --checkpoint $CK \
  --clips-root $CLIPS --start ${START:-0} --end ${END:-149} --out-root $OUT \
  ${MAX_FRAMES:+--max-frames $MAX_FRAMES} ${EXTRA:-}
echo "GSR2_END rc=$? $(date -Is)"
