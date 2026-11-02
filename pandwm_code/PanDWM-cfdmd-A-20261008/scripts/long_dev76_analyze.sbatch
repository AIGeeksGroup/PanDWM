#!/usr/bin/env bash
# 2026-10-14: Dev-76-L analysis (report section 27.9): features of every listed arm (76 scenes, seed 42, 10 windows),
# then eval_v7/long_analyze2.py with Test-L-like buckets and the split-half real reference.
#   ARMS="v84i4250 v83h4000" OUTNAME=dev76_analyze_4250 sbatch --dependency=afterany:<rollouts> scripts/long_dev76_analyze.sbatch
#SBATCH --job-name=pandwm_l76an
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
LV=$DR/pandwm_runs/long_videos_dev76_20261014
FD=$LV/feats
PY=$DR/pandwm_venv/bin/python
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=$DR/.cache/torch PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
ARMS=${ARMS:?ARMS}; OUTNAME=${OUTNAME:?OUTNAME}
SC=$(python3 -c "print(' '.join(str(i) for i in range(76)))")
cd "$C" || exit 1
echo "L76AN_START $(date -Is) node=$(hostname) arms=[$ARMS]"
ok=""
for t in $ARMS; do
  if "$PY" long_feats.py "$FD" --arm "$t=$LV" --scenes $SC --seeds 42 --windows 10; then ok="$ok $t"; else echo "FEATS_FAILED $t"; fi
done
[[ -n $ok ]] || { echo "NO_ARMS"; exit 1; }
"$PY" long_analyze2.py "$FD" "$LV/$OUTNAME.json" --arms $ok --seeds 42 --pool-split --jack-group 4 --boot 300 \
  --buckets "w1-2=1-2" "w3-5=3-5" "w6-8=6-8" "w9-10=9-10" "early w1-5=1-5" "late w6-10=6-10"
OMP_NUM_THREADS=4 "$PY" long_color.py "$LV" "$LV/${OUTNAME}_color.json" --arms $ok --scenes $SC --seeds 42 --windows 10 \
  || echo "COLOR_FAILED"
echo "L76AN_END $(date -Is)"
