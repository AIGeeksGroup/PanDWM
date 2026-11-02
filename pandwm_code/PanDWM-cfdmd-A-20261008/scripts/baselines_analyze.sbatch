#!/usr/bin/env bash
# 2026-10-20: Val-150 main-table metrics of the six-camera baselines next to ours, with tests_analyze.py (same real
# features, jackknife intervals and floor as the ours rows of analysis_val150.json).
#   ARMS="unimlvg_P=feats_unimlvg_P_s42.npz;unimlvg_G=..." PAIRS="ours:unimlvg_P,oursG:unimlvg_G" TAG=<name> \
#     sbatch scripts/baselines_analyze.sbatch
# ARMS files are relative to feats_val150_baselines; ours / oursG (3 seeds each) are always included.
#SBATCH --job-name=pandwm_blan
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
PY=$DR/pandwm_venv/bin/python
T=$DR/pandwm_runs/tests_20261018
FD=$T/feats_val150
FB=$T/feats_val150_baselines
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd PYTHONUNBUFFERED=1
cd "$C" || exit 1
args=(--arm "ours=$FD/P_s42.npz,$FD/P_s1000.npz,$FD/P_s2026.npz" --arm "oursG=$FD/G_s42.npz,$FD/G_s1000.npz,$FD/G_s2026.npz")
IFS=';' read -ra A <<< "$ARMS"
for a in "${A[@]}"; do
  name=${a%%=*}; files=${a#*=}; full=""
  IFS=',' read -ra FS <<< "$files"
  for f in "${FS[@]}"; do full="$full${full:+,}$FB/$f"; done
  args+=(--arm "$name=$full")
done
IFS=',' read -ra PS <<< "${PAIRS:-}"
for p in "${PS[@]}"; do args+=(--pair "$p"); done
echo "BLAN_START $(date -Is) $(hostname) ${args[*]}"
"$PY" tests_analyze.py --real "$FD/real.npz" "${args[@]}" --boot ${BOOT:-1000} --out "$FB/analysis_${TAG}.json"
echo "BLAN_END rc=$? $(date -Is)"
