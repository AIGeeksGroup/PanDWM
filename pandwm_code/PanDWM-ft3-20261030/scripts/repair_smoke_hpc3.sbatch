#!/usr/bin/env bash
#SBATCH --job-name=ft3_repsmoke
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs_dev_hpc3/logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58,ACD1-59
# 2026-10-30 GS_NEARFIX_PLAN_20261030.md step 2 smoke: G3b + difix_ref repair on one GS-Dev clip (CLIP, default 25),
# shifts 0/1/2/4, repaired arms 4dfixm / 4dfixf at 2 and 4 m, sample video saved; checks the chain and its timing.
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/da3_env_hpc3.sh
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export PANDWM_PANO_BLEND_W=128
OV=$DR/pandwm_code/PanDWM-ft3-20261030
export PYTHONPATH=$OV/gs4d_src:$OV/gsaa_src:$PYTHONPATH
EM=$DR/external_models
OUT=$DR/pandwm_runs/gs_dev_hpc3/shift/RS_smoke; mkdir -p $OUT
CK=$RUNS/ft3/G3b/2000.pth
cd $GSAA
echo "REPSMOKE_START $(date -Is) node=$(hostname)"
$PY $OV/scripts/difix_example_test.py $EM/difix_ref $EM/Difix3D/src $DR/difix_pkgs $OUT/difix_example_out.png || exit 1
$PY -m dwm_pano.eval_gs_shift -c $GS4D/configs/gs_review_hpc3.json \
  --data-pkl $DR/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl --checkpoint $CK \
  --clips-root $DR/pandwm_runs/eval149_hpc3_v83_20261012/v84i4250 --i3d $DR/pandwm_code/TATS_fvd/i3d_pretrained_400.pt \
  --shifts 0,1,2,4 --modes 4d --da3-relabel --da3-mode scale --drop-bottom-rows 2 --expected-clips 1 \
  --box-reading mmdet3d_v0x --camera-time-poses --da3-src $DA3_SRC --da3-anyview $DA3_ANYVIEW --da3-metric $DA3_METRIC \
  --da3-window clip --da3-fuse --plane-offset-px 1 ${EXTRA:-} \
  --repair difix_ref --repair-model $EM/difix_ref --repair-src $EM/Difix3D/src --repair-pkgs $DR/difix_pkgs \
  --clip-list ${CLIP:-25} --state $OUT/state.pt --output $OUT/report.json \
  --sample-clips ${CLIP:-25} --sample-dir $OUT/samples
rc=$?
echo "REPSMOKE_END rc=$rc $(date -Is)"
exit $rc
