#!/usr/bin/env bash
# Long-video analysis: long_feats.py features of every listed arm (windows 1-12 from the scene start), then
# long_analyze.py with window buckets (windows 1-2 are the standard evaluation clip and are reported for
# reference only) and the split-half reference.
#   ARMS="v84i4250" SEEDS="42" OUTNAME=testl_v84_s42 sbatch jobs/testl_analyze.sbatch
# LIST=<scene list> replaces the 71 test scenes (e.g. eval/manifests/testl_ext12_long12.json); PERWINDOW=<name> also
# writes per-window metrics to $LV/<name>.json.
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
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/eval
LV=${LV_OUT:-$DR/pandwm_runs/long_videos_testl}
LIST=${LIST:-$C/manifests/review149_test71_long12.json}
PY=$DR/pandwm_venv/bin/python
export PYTHONPATH=$DR/pandwm_code/src:$DR/pandwm_code/TATS_fvd
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
"$PY" long_analyze.py "$FD" "$LV/$OUTNAME.json" --arms $ok --seeds $SEEDS --pool-split --jack-group 1 --boot 300 \
  --buckets "w1-2=1-2" "w3-5=3-5" "w6-8=6-8" "w9-11=9-11" "w12=12-12" "early w3-5=3-5" "late w9-11=9-11"
OMP_NUM_THREADS=4 "$PY" long_color.py "$LV" "$LV/${OUTNAME}_color.json" --arms $ok --scenes $SC --seeds $SEEDS \
  --windows 12 || echo "COLOR_FAILED"
echo "TLAN_END $(date -Is)"
if [[ -n ${PERWINDOW:-} ]]; then
  B=""; for w in $(seq 1 12); do B="$B w$w=$w-$w"; done
  # shellcheck disable=SC2086
  "$PY" long_analyze.py "$FD" "$LV/$PERWINDOW.json" --arms $ok --seeds $SEEDS --buckets $B && echo "PERWINDOW_DONE"
fi
