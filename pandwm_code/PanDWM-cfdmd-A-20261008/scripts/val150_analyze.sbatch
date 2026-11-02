#!/usr/bin/env bash
# 2026-10-19: 150-scene val protocol (Dev-149 + scene-0018) for the main table: merge the per-part features and run
# tests_analyze.py on ours (setting P, 3 seeds) and ours-G (setting G, 3 seeds).
#SBATCH --job-name=pandwm_v150an
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
F149=$T/feats_dev149
F18=$T/feats_scene0018
FD=$T/feats_val150
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd PYTHONUNBUFFERED=1
cd "$C" || exit 1
mkdir -p "$FD"
echo "V150_START $(date -Is) node=$(hostname)"
fail=0
"$PY" merge_feats.py "$FD/real.npz" "$F149/feats_real_dev149_manifest_20261018.npz" "$F18/feats_real_scene0018_manifest_20261019.npz" || fail=1
"$PY" merge_feats.py "$FD/P_s42.npz" "$F149/feats_v84i4250_dev149.npz" "$F18/feats_val_v84i4250_s42.npz" || fail=1
for s in 1000 2026; do
  "$PY" merge_feats.py "$FD/P_s$s.npz" "$F149/feats_val_v84i4250_s$s.npz" "$F18/feats_val_v84i4250_s$s.npz" || fail=1
done
for s in 42 1000 2026; do
  "$PY" merge_feats.py "$FD/G_s$s.npz" "$F149/feats_val_v84i4250G_s$s.npz" "$F18/feats_val_v84i4250G_s$s.npz" || fail=1
done
[[ $fail == 0 ]] || { echo "MERGE_FAILED"; exit 1; }
"$PY" tests_analyze.py --real "$FD/real.npz" --arm "ours=$FD/P_s42.npz,$FD/P_s1000.npz,$FD/P_s2026.npz" \
  --arm "oursG=$FD/G_s42.npz,$FD/G_s1000.npz,$FD/G_s2026.npz" --boot 0 --out "$FD/analysis_val150.json" || fail=1
echo "V150_END fail=$fail $(date -Is)"
exit $fail
