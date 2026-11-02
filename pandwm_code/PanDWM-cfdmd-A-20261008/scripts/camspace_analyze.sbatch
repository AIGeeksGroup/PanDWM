#!/usr/bin/env bash
# 2026-10-20: FID / FVD-1 / FVD-2 of the six-camera band protocol (cam_band_analyze.py = tests_analyze.py with FID over
# 6 cameras x frames), real = the recorded 256x448 views with the same mask (feats_rec448.npz).
#   ARMS="ours=P_s42.npz,P_s1000.npz,P_s2026.npz;unimlvg_P=unimlvg_P_s42.npz;recband=recband.npz" \
#   PAIRS="ours:unimlvg_P,ours:recband" TAG=P sbatch camspace_analyze.sbatch
# ARMS files are relative to $FD (feats_<file>); output $FD/analysis_<TAG>.json.
#SBATCH --job-name=pandwm_caman
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
PY=$DR/pandwm_venv/bin/python
GEOMETRY=${GEOMETRY:-onebev_inverse_own}
FD=${FD:-$DR/pandwm_runs/tests_20261018/feats_val150_cam_$GEOMETRY}
REAL=${REAL:-$FD/feats_rec448.npz}
export PYTHONPATH=$C:$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd PYTHONUNBUFFERED=1
cd "$C" || exit 1
args=()
IFS=';' read -ra A <<< "$ARMS"
for a in "${A[@]}"; do
  name=${a%%=*}; files=${a#*=}; full=""
  IFS=',' read -ra FS <<< "$files"
  for f in "${FS[@]}"; do full="$full${full:+,}$FD/feats_$f"; done
  args+=(--arm "$name=$full")
done
IFS=',' read -ra PS <<< "${PAIRS:-}"
for p in "${PS[@]}"; do args+=(--pair "$p"); done
echo "CAMAN_START $(date -Is) $(hostname) real=$REAL ${args[*]}"
"$PY" cam_band_analyze.py --real "$REAL" "${args[@]}" --boot ${BOOT:-1000} --out "$FD/analysis_${TAG}.json"
rc=$?
echo "CAMAN_END rc=$rc $(date -Is)"
exit $rc
