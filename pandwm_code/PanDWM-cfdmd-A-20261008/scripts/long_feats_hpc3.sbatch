#!/usr/bin/env bash
# 2026-10-14: features + analysis of long-video diagnostics, one GPU (eval_v7/long_feats.py, long_analyze.py).
#   ARMS="v83h2000 v83h4000 v84i4250 v84i4250raw"  tags in long_videos_hpc3_20261011 (videos <tag>_scene<N>_seed<S>)
#   SEEDS="42 43" FIRST_WINDOW=1                   videos to read (default all 12 scenes)
#   POOL="dev78gt=<eval149 arm dir>"               optional unpaired real pool (the 78 non-test 149-clip eval clips)
#   ANALYZE="A B ..." ANALYZE_ARGS="--pool dev78gt" OUTNAME=long_analyze_x   optional analysis after extraction
#   LV_OUT=<video root> WINDOWS=10 SCENES="0 1 ... 75"   other video sets (e.g. Dev-76-L, long_video_dev76.sbatch)
# Features go to <video root>/feats (existing feats_<tag>.npz are reused).
#SBATCH --job-name=pandwm_lfeat
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
LV=${LV_OUT:-$DR/pandwm_runs/long_videos_hpc3_20261011}
FD=$LV/feats
PY=$DR/pandwm_venv/bin/python
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=$DR/.cache/torch PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
SCENES=${SCENES:-"0 14 21 32 41 49 62 82 10 31 52 72"}; SEEDS=${SEEDS:-"42 43"}; FW=${FIRST_WINDOW:-1}
WIN=${WINDOWS:-12}
cd "$C" || exit 1
echo "LFEAT_START $(date -Is) node=$(hostname) arms=[${ARMS:-}] seeds=[$SEEDS] first_window=$FW pool=${POOL:-none}"
rc=0
if [[ -n ${ARMS:-} ]]; then
  args=(); for t in $ARMS; do args+=(--arm "$t=$LV"); done
  "$PY" long_feats.py "$FD" "${args[@]}" --scenes $SCENES --seeds $SEEDS --first-window "$FW" --windows "$WIN" || rc=1
fi
if [[ -n ${POOL:-} ]]; then
  "$PY" long_feats.py "$FD" --pool "$POOL" --pool-clips "$C/dev149_nontest_split_20261013.json" || rc=1
fi
if [[ -n ${ANALYZE:-} && $rc == 0 ]]; then
  "$PY" long_analyze.py "$FD" "$LV/${OUTNAME:?OUTNAME}.json" --arms $ANALYZE ${ANALYZE_ARGS:-} || rc=1
fi
echo "LFEAT_END $(date -Is) rc=$rc"
exit $rc
