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
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/d2_refiner/logs/%x_%A_%a.log
# D2 refiner, GS side.
#   MODE=pairs  [TASKS=14 LIMIT=0 BUDGET_S=9000 OUT=train_pairs]  sbatch --array=0-13 d2_render_hpc3.sbatch
#   MODE=count                                                       sbatch d2_render_hpc3.sbatch
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/gaussian/gs4d/scripts_hpc3/common_env_hpc3.sh
source /path/to/pandwm_root/pandwm_code/gaussian/gs4d/scripts_hpc3/da3_env_hpc3.sh
export MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID} + ${SLURM_ARRAY_TASK_ID:-0}) % 20000 ))
export PANDWM_PANO_BLEND_W=128                  # the frozen protocol's stitch (eval mode; renders do not stitch halves)
D2R=$DR/pandwm_runs/d2_refiner
CK=${CKPT:-$RUNS/c2_hpc2/checkpoints/3000.pth}
TRAIN_PKL=$DR/pandwm_recovered_metadata/nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl
MANIFEST=$DR/pandwm_code/eval/manifests/split_195_scene_manifest.json
DA3ARGS="--da3-src $DA3_SRC --da3-anyview $DA3_ANYVIEW --da3-metric $DA3_METRIC"
TASK=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:?MODE=pairs|count}
cd $GSAA
echo "D2GS_START mode=$MODE task=$TASK node=$(hostname) $(date -Is)"
case $MODE in
  pairs|count)
    EXTRA=""; [[ $MODE == count ]] && EXTRA="--count-only"
    $PY -m dwm_pano.refiner.make_train -c $GS4D/configs/gs4dd_hpc3_C.json --data-pkl $TRAIN_PKL --checkpoint $CK \
      --out-root $D2R/${OUT:-train_pairs} --manifest $MANIFEST --task $TASK --tasks ${TASKS:-1} --limit ${LIMIT:-0} \
      --time-budget-s ${BUDGET_S:-9000} --preview-dir $D2R/${OUT:-train_pairs}_previews --previews ${PREVIEWS:-2} \
      $DA3ARGS $EXTRA ;;
esac
rc=$?
echo "D2GS_END mode=$MODE task=$TASK rc=$rc $(date -Is)"
exit $rc
