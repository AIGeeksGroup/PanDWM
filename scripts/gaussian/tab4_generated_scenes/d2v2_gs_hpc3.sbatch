#!/usr/bin/env bash
#SBATCH --job-name=d2v2_gs
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
# 2026-10-31 D2 v2 GS side (PanDWM-d2-refiner-20261021/D2V2_PLAN_20261031.md section 3), PanDWM-ft3-20261030 code copy.
#   MODE=export [SET=dev] sbatch --array=0-5 d2v2_gs_hpc3.sbatch      export packs (0, +-1, +-2, +-4 m) for GS-Dev 78
#   MODE=eval TAG=<name> [REFINED=<dir>] sbatch --array=0-5 d2v2_gs_hpc3.sbatch ; MODE=eval TAG=<name> MERGE=1 sbatch ...
#   SET=test: Test-S 71 (clips 1, 4, ..., 211 of tests_20261018/v84i4250_s42, fps-stride 10,5)
#   SET=testext: the 12 clips 214, 217, ..., 247 that extend Test-S to the 83 clips of tab:gs_gen
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/da3_env_hpc3.sh
export MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID} + ${SLURM_ARRAY_TASK_ID:-0}) % 20000 ))
export PANDWM_PANO_BLEND_W=128
OV=$DR/pandwm_code/PanDWM-ft3-20261030
export PYTHONPATH=$OV/gs4d_src:$OV/gsaa_src:$PYTHONPATH
R=$DR/pandwm_runs/d2v2_20261031
CK=${CKPT:-$RUNS/ft3/G3b/2000.pth}
VAL_PKL=$DR/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl
I3D=$DR/pandwm_code/TATS_fvd/i3d_pretrained_400.pt
SET=${SET:-dev}
if [ "$SET" = dev ]; then
  LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,43,44,45,46,47,48,52,58,59,63,64,67,68,73,79,84,85,91,95,100,102,105,106,108,110,117,121,122,123,127,132,135,140,146,147,148
  CLIPS=$DR/pandwm_runs/eval149_hpc3_v83_20261012/v84i4250; FS=10,20
elif [ "$SET" = testext ]; then                   # 11-01: the 12 tab:gs_gen ext83 clips (tests_ext36, offset 10 s)
  LIST=214,217,220,223,226,229,232,235,238,241,244,247
  CLIPS=$DR/pandwm_runs/tests_ext_20261024/v84i4250_s42; FS=10,5
else
  LIST=$(python3 -c "print(','.join(map(str, range(1, 213, 3))))")
  CLIPS=$DR/pandwm_runs/tests_20261018/v84i4250_s42; FS=10,5
fi
[ -n "${CLIP_LIST:-}" ] && LIST=$CLIP_LIST                 # smoke: an explicit clip subset
N=$(echo $LIST | tr ',' '\n' | grep -c .)
TASK=${SLURM_ARRAY_TASK_ID:-0}; NT=${SLURM_ARRAY_TASK_COUNT:-1}
SEL=$(python3 -c "import sys; l=sys.argv[1].split(','); print(','.join(l[int(sys.argv[2])::int(sys.argv[3])]))" "$LIST" $TASK $NT)
mkdir -p $R/logs
cd $GSAA
echo "D2V2GS_START mode=$MODE set=$SET task=$TASK/$NT tag=${TAG:-} node=$(hostname) $(date -Is)"
case $MODE in
  export)
    $PY -m dwm_pano.d2v2_export -c $GS4D/configs/gs_review_hpc3.json --data-pkl $VAL_PKL --checkpoint $CK \
      --clips-root $CLIPS --clips "$SEL" --fps-stride $FS --out-root $R/inputs_$SET \
      --da3-src $DA3_SRC --da3-anyview $DA3_ANYVIEW --da3-metric $DA3_METRIC ;;
  shiftcheck)                                          # 10-31 section 6.4: does the refiner keep the shift?
    TAG=${TAG:?TAG}
    $PY -m dwm_pano.d2v2_shiftcheck -c $GS4D/configs/gs_review_hpc3.json --inputs $R/inputs_$SET \
      --refined $R/refined_$SET/$TAG --out $R/shiftcheck_$SET/$TAG ${EXTRA:-} ;;
  eval)
    TAG=${TAG:?TAG}; ST=$R/eval_$SET/$TAG; mkdir -p $ST
    COMMON="-c $GS4D/configs/gs_review_hpc3.json --data-pkl $VAL_PKL --inputs $R/inputs_$SET --fps-stride $FS \
      --i3d $I3D --expected-clips $N ${REFINED:+--refined $REFINED} ${EXTRA:-}"
    if [ -n "${MERGE:-}" ]; then
      $PY -m dwm_pano.d2v2_eval $COMMON --state $ST/merged.pt --merge $ST/state_*.pt --output $ST/report.json --device cpu
    else
      $PY -m dwm_pano.d2v2_eval $COMMON --clips "$SEL" --state $ST/state_$TASK.pt
    fi ;;
esac
rc=$?
echo "D2V2GS_END rc=$rc $(date -Is)"
exit $rc
