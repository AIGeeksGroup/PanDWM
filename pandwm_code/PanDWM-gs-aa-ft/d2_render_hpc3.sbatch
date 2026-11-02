#!/usr/bin/env bash
#SBATCH --job-name=pandwm_d2gs
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/d2_refiner_20261021/logs/%x_%A_%a.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58,ACD1-59
# 2026-10-21 D2 refiner, GS side (PanDWM-d2-refiner-20261021/D2_PLAN_20261021.md Tasks 5 / 7 / 9 / 10).
#   MODE=pairs  [TASKS=14 LIMIT=0 BUDGET_S=9000 OUT=train_pairs]  sbatch --array=0-13 d2_render_hpc3.sbatch
#   MODE=count                                                       sbatch d2_render_hpc3.sbatch
#   MODE=export CLIPS=0:149|<list> [SHIFTS=0,1,-1,2,-2 TAG=c2b_drop2] sbatch --array=0-3 d2_render_hpc3.sbatch
#   MODE=eval   ARGS="..."                                           sbatch d2_render_hpc3.sbatch
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/da3_env_hpc3.sh
export MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID} + ${SLURM_ARRAY_TASK_ID:-0}) % 20000 ))
export PANDWM_PANO_BLEND_W=128                  # the frozen protocol's stitch (eval mode; renders do not stitch halves)
D2R=$DR/pandwm_runs/d2_refiner_20261021
CK=${CKPT:-$RUNS/c2_hpc2/checkpoints/3000.pth}
TRAIN_PKL=$DR/pandwm_recovered_metadata_20260929/nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl
VAL_PKL=$DR/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl
MANIFEST=$DR/pandwm_code/PanDWM_long_horizon_20260928/inputs/split_195_scene_manifest.json
DA3ARGS="--da3-src $DA3_SRC --da3-anyview $DA3_ANYVIEW --da3-metric $DA3_METRIC"
TASK=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:?MODE=pairs|count|export|eval}
cd $GSAA
echo "D2GS_START mode=$MODE task=$TASK node=$(hostname) $(date -Is)"
case $MODE in
  pairs|count)
    EXTRA=""; [[ $MODE == count ]] && EXTRA="--count-only"
    $PY -m dwm_pano.d2_make_train -c $GS4D/configs/gs4dd_hpc3_C.json --data-pkl $TRAIN_PKL --checkpoint $CK \
      --out-root $D2R/${OUT:-train_pairs} --manifest $MANIFEST --task $TASK --tasks ${TASKS:-1} --limit ${LIMIT:-0} \
      --time-budget-s ${BUDGET_S:-9000} --preview-dir $D2R/${OUT:-train_pairs}_previews --previews ${PREVIEWS:-2} \
      $DA3ARGS $EXTRA ;;
  export)
    N=${SLURM_ARRAY_TASK_COUNT:-1}
    CL=$($PY -c "import sys; t=sys.argv[1]; s=(list(range(*map(int,t.split(':')))) if ':' in t else [int(x) for x in t.split(',') if x]); n,k=int(sys.argv[2]),int(sys.argv[3]); print(','.join(map(str,s[k::n])))" "${CLIPS:?CLIPS=a:b or list}" $N $TASK)
    $PY -m dwm_pano.d2_export_shift -c $GS4D/configs/gs_review_hpc3.json --data-pkl $VAL_PKL --checkpoint $CK \
      --clips-root $DR/pandwm_runs/eval149_hpc3_v83_20261012/v84i4250 --clips "$CL" --shifts ${SHIFTS:-0,1,-1,2,-2} \
      --out-root $D2R/review_inputs/${TAG:-c2b_drop2} --video-clips "${VIDEO_CLIPS:-}" \
      --time-budget-s ${BUDGET_S:-9000} $DA3ARGS ;;
  eval)
    $PY -m dwm_pano.d2_eval_refined -c $GS4D/configs/gs_review_hpc3.json --data-pkl $VAL_PKL \
      --clips-root $DR/pandwm_runs/eval149_hpc3_v83_20261012/v84i4250 --i3d $DR/pandwm_code/TATS_fvd/i3d_pretrained_400.pt \
      ${ARGS:?ARGS} ;;
esac
rc=$?
echo "D2GS_END mode=$MODE task=$TASK rc=$rc $(date -Is)"
exit $rc
