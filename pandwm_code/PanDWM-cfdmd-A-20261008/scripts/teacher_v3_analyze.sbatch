#!/usr/bin/env bash
# 2026-10-15: teacher fine-tune v3 evaluation analysis (report section 30): features of every listed arm
# (76 Dev-76-L scenes, seed 42, windows FIRST..LAST), long_analyze2.py with the split-half real reference, colour.
#   ARMS="t0_gt2 tc_gt2 tr_gt2" FIRST=1 LAST=2 BUCKETS="w1=1-1 w2=2-2 all_w1-2=1-2" OUTNAME=tv3_e2 sbatch ...
#   ARMS="t0_d7 tc_d7 tr_d7 tr_d7n10" FIRST=7 LAST=7 BUCKETS="w7=7-7" OUTNAME=tv3_e1 sbatch ...
#SBATCH --job-name=pandwm_tv3an
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
LV=${ROOT:-$DR/pandwm_runs/teacher_v3_20261015/eval}
FD=$LV/feats
PY=$DR/pandwm_venv/bin/python
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=$DR/.cache/torch PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
ARMS=${ARMS:?ARMS}; OUTNAME=${OUTNAME:?OUTNAME}; FIRST=${FIRST:-1}; LAST=${LAST:-2}; BUCKETS=${BUCKETS:?BUCKETS}
SC=$(python3 -c "print(' '.join(str(i) for i in range(76)))")
cd "$C" || exit 1
echo "TV3AN_START $(date -Is) node=$(hostname) arms=[$ARMS] windows=$FIRST-$LAST"
ok=""
for t in $ARMS; do
  if "$PY" long_feats.py "$FD" --arm "$t=$LV" --scenes $SC --seeds 42 --first-window "$FIRST" --windows "$LAST"; then
    ok="$ok $t"; else echo "FEATS_FAILED $t"; fi
done
[[ -n $ok ]] || { echo "NO_ARMS"; exit 1; }
"$PY" long_analyze2.py "$FD" "$LV/$OUTNAME.json" --arms $ok --seeds 42 --pool-split --jack-group 4 --boot 300 \
  --buckets $BUCKETS
NW=$((LAST - FIRST + 1))
OMP_NUM_THREADS=4 "$PY" long_color.py "$LV" "$LV/${OUTNAME}_color.json" --arms ${COLOR_ARMS:-$ok} --scenes $SC --seeds 42 \
  --windows "$NW" || echo "COLOR_FAILED"
echo "TV3AN_END $(date -Is)"
