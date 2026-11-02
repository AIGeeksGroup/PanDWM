#!/usr/bin/env bash
#SBATCH --job-name=pandwm_gsld
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs4d_hpc3_20261017/logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
# 2026-10-17 (report section 36): label-depth feasibility test (eval_label_depth.py) on the p2 training clips.
#   NAME=<tag> [CHECKPOINT=$RUNS/init/gs4d_cov_3000.pth] [RESIZE=nearest|bilinear] [CLIPS=25] [VIDEO_CLIPS=3]
#   [VIDEO_FRAMES=20] [DAC=1] [DA3=1]  ->  $RUNS/label_depth/<NAME>.json and $RUNS/label_depth/videos_<NAME>/
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
CK=${CHECKPOINT:-$RUNS/init/gs4d_cov_3000.pth}
DACR=/path/to/pandwm_root/pandwm_code/third_party/depth_any_camera   # DAC=1 adds the dac / dac_edge variants (v2)
[ -n "${DA3:-}" ] && source $GS4D/scripts_hpc3/da3_env_hpc3.sh       # DA3=1 adds da3 / da3_edge (v3, section 40)
mkdir -p $RUNS/label_depth
echo "GSLD_START name=${NAME:?} ckpt=$CK resize=${RESIZE:-nearest} clips=${CLIPS:-25} node=$(hostname) $(date -Is)"
$PY -m dwm_pano_4d.eval_label_depth -c $GS4D/configs/gs4dd_hpc3_C.json --checkpoint $CK \
  --data-pkl $DR/pandwm_gs_aa_experiment/inputs/nuscenes_interp_12Hz_infos_train_p2.pkl \
  --clips ${CLIPS:-25} --resize ${RESIZE:-nearest} --video-clips ${VIDEO_CLIPS:-3} \
  --video-frames ${VIDEO_FRAMES:-20} --output $RUNS/label_depth/$NAME.json --video-dir $RUNS/label_depth/videos_$NAME \
  ${DAC:+--dac-root $DACR --dac-config $DACR/checkpoints/dac_swinl_outdoor.json --dac-weights $DACR/checkpoints/dac_swinl_outdoor.pt} \
  ${DA3:+--da3-src $DA3_SRC --da3-anyview $DA3_ANYVIEW --da3-metric $DA3_METRIC} ${DA3_MODE:+--da3-relabel-mode $DA3_MODE}
rc=$?
echo "GSLD_END rc=$rc $(date -Is)"
exit $rc
