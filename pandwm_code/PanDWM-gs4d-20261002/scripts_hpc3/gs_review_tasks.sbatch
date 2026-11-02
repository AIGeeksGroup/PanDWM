#!/usr/bin/env bash
#SBATCH --job-name=pandwm_gsrev
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs4d_hpc3_20261017/logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
# 2026-10-20 (GS_GAP_ANALYSIS_20261020.md) GS-side tasks on HPC3 for the 149 review clips of v84i4250:
#   MODE=refs      dwm_pano.make_difix_refs            -> pandwm_runs/difix_20261020/refs
#   MODE=d1render  dwm_pano.export_shift_pano_latents  -> pandwm_runs/d1_sdedit_20261020/renders
#                  (C2 step 3000 + DA3 scale relabel + 2 bottom rows dropped, as the v4 far views)
#   MODE=refs CLIPS=2,3,4 sbatch gs_review_tasks.sbatch ;  MODE=d1render CLIPS=2,60 SHIFTS=0,1,2 sbatch ...
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/da3_env_hpc3.sh
export MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID} % 20000 ))
CLIPSROOT=$DR/pandwm_runs/eval149_hpc3_v83_20261012/v84i4250
CFG=$GS4D/configs/gs_review_hpc3.json
PKL=$DR/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl
cd $GSAA
echo "GSREV_START mode=${MODE:?} clips=${CLIPS:?} node=$(hostname) $(date -Is)"
case $MODE in
  refs) $PY -m dwm_pano.make_difix_refs -c $CFG --data-pkl $PKL --checkpoint $RUNS/init/gs4d_cov_3000.pth \
          --clips-root $CLIPSROOT --clips "$CLIPS" --out-root $DR/pandwm_runs/difix_20261020/refs ;;
  d1render) $PY -m dwm_pano.export_shift_pano_latents -c $CFG --data-pkl $PKL \
          --checkpoint $RUNS/c2_hpc2/checkpoints/3000.pth --clips-root $CLIPSROOT --clips "$CLIPS" \
          --shifts ${SHIFTS:-0,1,2} --out-root $DR/pandwm_runs/d1_sdedit_20261020/renders --da3-relabel --da3-mode scale \
          --drop-bottom-rows 2 --da3-src $DA3_SRC --da3-anyview $DA3_ANYVIEW --da3-metric $DA3_METRIC ;;
  *) echo "UNKNOWN_MODE $MODE"; exit 1 ;;
esac
rc=$?
echo "GSREV_END rc=$rc $(date -Is)"
exit $rc
