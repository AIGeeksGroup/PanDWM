#!/usr/bin/env bash
#SBATCH --job-name=ft3_devshift
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs_dev_hpc3/logs/%x_%A_%a.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58,ACD1-59
# 2026-10-30 round-3 (GS_FT3_PLAN_20261030.md): dev_shift_hpc3.sbatch with the PanDWM-ft3-20261030 code copy first on
# PYTHONPATH (bounded / no-offset scene options, sharpness metric) and sample videos of 4 clips (real, 0, +2, -2, +4 m).
# Copied from the 2026-10-24 round-2 GS-Dev generated-scene shift eval: eval_gs_shift on the Dev-78 clips of the
# v84i4250 Dev-149 generations, fixed 4D (camera-time poses, v0x boxes), v4 settings, shifts 0/1/2/4, 4d only.
#   ARM=<name> [CKPT=...] [EXTRA="--sky-layer"] sbatch --array=0-5 dev_shift_hpc3.sbatch ; ARM=<name> MERGE=1 sbatch dev_shift_hpc3.sbatch
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/da3_env_hpc3.sh
export MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID} + ${SLURM_ARRAY_TASK_ID:-0}) % 20000 ))
export PANDWM_PANO_BLEND_W=128
OV=$DR/pandwm_code/PanDWM-ft3-20261030
export PYTHONPATH=$OV/gs4d_src:$OV/gsaa_src:$PYTHONPATH
ARM=${ARM:?ARM}
OUT=$DR/pandwm_runs/gs_dev_hpc3/shift/$ARM; mkdir -p $OUT
CK=${CKPT:-$RUNS/c2_hpc2/checkpoints/3000.pth}
LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,43,44,45,46,47,48,52,58,59,63,64,67,68,73,79,84,85,91,95,100,102,105,106,108,110,117,121,122,123,127,132,135,140,146,147,148
TASK=${SLURM_ARRAY_TASK_ID:-0}
SEL=$(python3 -c "import sys; l=sys.argv[1].split(','); print(','.join(l[int(sys.argv[2])::6]))" "$LIST" $TASK)
cd $GSAA
COMMON="-c $GS4D/configs/gs_review_hpc3.json --data-pkl $DR/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl \
  --checkpoint $CK --clips-root $DR/pandwm_runs/eval149_hpc3_v83_20261012/v84i4250 --i3d $DR/pandwm_code/TATS_fvd/i3d_pretrained_400.pt \
  --shifts 0,1,2,4 --modes 4d ${DA3FLAGS---da3-relabel --da3-mode scale} --drop-bottom-rows 2 --expected-clips 78 \
  --box-reading mmdet3d_v0x --camera-time-poses --da3-src $DA3_SRC --da3-anyview $DA3_ANYVIEW --da3-metric $DA3_METRIC ${EXTRA:-}"
echo "DEVSHIFT_START code=$OV arm=$ARM task=$TASK ckpt=$CK extra=${EXTRA:-} $(date -Is)"
if [ -n "${MERGE:-}" ]; then
  $PY -m dwm_pano.eval_gs_shift $COMMON --state $OUT/merged.pt --merge $OUT/state_*.pt --output $OUT/report.json --device cpu
else
  $PY -m dwm_pano.eval_gs_shift $COMMON --clip-list "$SEL" --state $OUT/state_$TASK.pt \
    --sample-clips 0,25,64,122 --sample-dir $OUT/samples
fi
rc=$?
echo "DEVSHIFT_END rc=$rc $(date -Is)"
exit $rc
