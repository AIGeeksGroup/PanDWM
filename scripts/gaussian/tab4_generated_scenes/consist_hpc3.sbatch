#!/usr/bin/env bash
#SBATCH --job-name=consist
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/d2v2_20261031/logs/%x_%A_%a.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58,ACD1-59
# 2026-11-01 multi-view consistency of the lateral-shift views (PanDWM-d2-refiner-20261021/CONSIST_PLAN_20261101.md):
#   [SET=test] [TAG=v1_4000_D] [OUT_DIR=..] [REFINED2=<dir>] [EXTRA=..] sbatch --array=0-3 consist_hpc3.sbatch ; MERGE=1 ...
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/da3_env_hpc3.sh
export MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID} * 7 + ${SLURM_ARRAY_TASK_ID:-0} * 131 + RANDOM) % 20000 ))
export PANDWM_PANO_BLEND_W=128
OV=$DR/pandwm_code/PanDWM-fuse-20261031
export PYTHONPATH=$OV/gs4d_src:$OV/gsaa_src:$PYTHONPATH
R=$DR/pandwm_runs/d2v2_20261031
SET=${SET:-test}; TAG=${TAG:-v1_4000_D}
if [ "$SET" = dev ]; then
  LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,43,44,45,46,47,48,52,58,59,63,64,67,68,73,79,84,85,91,95,100,102,105,106,108,110,117,121,122,123,127,132,135,140,146,147,148
  CLIPS=$DR/pandwm_runs/eval149_hpc3_v83_20261012/v84i4250; FS=10,20
else
  LIST=$(python3 -c "print(','.join(map(str, range(1, 213, 3))))")
  CLIPS=$DR/pandwm_runs/tests_20261018/v84i4250_s42; FS=10,5
fi
[ -n "${CLIP_LIST:-}" ] && LIST=$CLIP_LIST
TASK=${SLURM_ARRAY_TASK_ID:-0}; NT=${SLURM_ARRAY_TASK_COUNT:-1}
SEL=$(python3 -c "import sys; l=sys.argv[1].split(','); print(','.join(l[int(sys.argv[2])::int(sys.argv[3])]))" "$LIST" $TASK $NT)
OUT=${OUT_DIR:-$R/consist_$SET/$TAG}
cd $GSAA
echo "CONSIST_START set=$SET tag=$TAG task=$TASK/$NT merge=${MERGE:-} node=$(hostname) $(date -Is)"
if [ -n "${MERGE:-}" ]; then
  $PY -m dwm_pano.consistency_eval --out $OUT --merge
else
  $PY -m dwm_pano.consistency_eval -c $GS4D/configs/gs_review_hpc3.json \
    --data-pkl $DR/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl --checkpoint $RUNS/ft3/G3b/2000.pth \
    --clips-root $CLIPS --inputs $R/inputs_$SET --refined $R/refined_$SET/$TAG --clips "$SEL" --fps-stride $FS \
    --out $OUT --task $TASK ${REFINED2:+--refined2 $REFINED2} --da3-src $DA3_SRC --da3-anyview $DA3_ANYVIEW \
    --da3-metric $DA3_METRIC ${EXTRA:-}
fi
echo "CONSIST_END rc=$? $(date -Is)"
