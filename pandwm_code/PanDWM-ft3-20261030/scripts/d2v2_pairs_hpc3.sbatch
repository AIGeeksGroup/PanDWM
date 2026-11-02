#!/usr/bin/env bash
#SBATCH --job-name=d2v2_pairs
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/d2v2_20261031/logs/%x_%A_%a.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58,ACD1-59
# 2026-10-31 D2 v2 round-trip training pairs (PanDWM-d2-refiner-20261021/D2V2_PLAN_20261031.md), G3b decoder, with the
# PanDWM-ft3-20261030 code copy first on PYTHONPATH.  Task k of TASKS takes positions k, k + TASKS, ... of the hash order.
#   [TASKS=16 LIMIT=0 BUDGET_S=10200 OUT=train_pairs] sbatch --array=0-15 d2v2_pairs_hpc3.sbatch
#   smoke: TASKS=1 LIMIT=2 OUT=smoke_pairs PREVIEWS=2 sbatch d2v2_pairs_hpc3.sbatch
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/da3_env_hpc3.sh
export MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID} + ${SLURM_ARRAY_TASK_ID:-0}) % 20000 ))
export PANDWM_PANO_BLEND_W=128
OV=$DR/pandwm_code/PanDWM-ft3-20261030
export PYTHONPATH=$OV/gs4d_src:$OV/gsaa_src:$PYTHONPATH
R=$DR/pandwm_runs/d2v2_20261031
CK=${CKPT:-$RUNS/ft3/G3b/2000.pth}
TRAIN_PKL=$DR/pandwm_recovered_metadata_20260929/nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl
MANIFEST=$DR/pandwm_code/PanDWM_long_horizon_20260928/inputs/split_195_scene_manifest.json
TASK=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $R/logs
cd $GSAA
echo "D2V2_PAIRS_START task=$TASK/${TASKS:-1} ckpt=$CK node=$(hostname) $(date -Is)"
$PY -m dwm_pano.d2v2_make_pairs -c $GS4D/configs/gs4dd_hpc3_C.json --data-pkl $TRAIN_PKL --checkpoint $CK \
  --out-root $R/${OUT:-train_pairs} --manifest $MANIFEST --task $TASK --tasks ${TASKS:-1} --limit ${LIMIT:-0} \
  --time-budget-s ${BUDGET_S:-10200} --preview-dir $R/${OUT:-train_pairs}_previews --previews ${PREVIEWS:-1} \
  --da3-src $DA3_SRC --da3-anyview $DA3_ANYVIEW --da3-metric $DA3_METRIC
rc=$?
echo "D2V2_PAIRS_END task=$TASK rc=$rc $(date -Is)"
exit $rc
