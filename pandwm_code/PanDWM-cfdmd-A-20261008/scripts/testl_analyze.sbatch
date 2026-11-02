#!/usr/bin/env bash
# 2026-10-18: Test-L analysis (PAPER_COMPARISON_PLAN_20261012.md section 3.8; report section 39): long_feats.py
# features of every listed arm (71 test scenes, windows 1-12 from the scene start), then long_analyze2.py with the
# Test-L buckets (windows 1-2 are the Dev-149 clip and are reported for reference only) and the split-half reference.
#   ARMS="v84i4250" SEEDS="42" OUTNAME=testl_v84_s42 sbatch scripts/testl_analyze.sbatch
# Feature files are per (arm, seed set): $FD/feats_<arm>.npz under FD=$LV/feats_s<seeds joined by _>.
#SBATCH --job-name=pandwm_tlan
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
LV=${LV_OUT:-$DR/pandwm_runs/long_videos_testl_20261018}
LIST=$C/review149_test71_long12_20261016.json
PY=$DR/pandwm_venv/bin/python
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=$DR/.cache/torch PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
ARMS=${ARMS:?ARMS}; OUTNAME=${OUTNAME:?OUTNAME}; SEEDS=${SEEDS:-42}
FD=$LV/feats_s${SEEDS// /_}
SC=$(python3 -c "import json; print(' '.join(str(r['dev76_index']) for r in json.load(open('$LIST'))['scenes']))")
cd "$C" || exit 1
echo "TLAN_START $(date -Is) node=$(hostname) arms=[$ARMS] seeds=[$SEEDS] scenes=$(echo $SC | wc -w)"
ok=""
for t in $ARMS; do
  if "$PY" long_feats.py "$FD" --arm "$t=$LV" --scenes $SC --seeds $SEEDS --windows 12; then ok="$ok $t"; else echo "FEATS_FAILED $t"; fi
done
[[ -n $ok ]] || { echo "NO_ARMS"; exit 1; }
"$PY" long_analyze2.py "$FD" "$LV/$OUTNAME.json" --arms $ok --seeds $SEEDS --pool-split --jack-group 1 --boot 300 \
  --buckets "w1-2=1-2" "w3-5=3-5" "w6-8=6-8" "w9-11=9-11" "w12=12-12" "early w3-5=3-5" "late w9-11=9-11"
OMP_NUM_THREADS=4 "$PY" long_color.py "$LV" "$LV/${OUTNAME}_color.json" --arms $ok --scenes $SC --seeds $SEEDS \
  --windows 12 || echo "COLOR_FAILED"
echo "TLAN_END $(date -Is)"
