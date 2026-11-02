#!/usr/bin/env bash
# 2026-10-11 19:10: phase 2b of the split HPC3 evaluation: merge the 8 FID / FVD states into metrics/report.json and
# compute look stats and frame-to-frame change (realism_arms.py, b0 first as in realism3.py) for the complete arms.
# Exits 0 only when the merge succeeded (v7 run D waits on this job with afterok).
#   sbatch --dependency=afterany:<metrics array> scripts/eval149_hpc3_fin.sbatch
#SBATCH --job-name=pandwm_e149f
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=01:30:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
CFG=$C/cd_eval_frozen149_hpc3.json
OUT=${EVAL_OUT:-$DR/pandwm_runs/eval149_hpc3_20261011}
PY=$DR/pandwm_venv/bin/python
TAGS=${TAGS:-"v82g2000 v81f1600 v7d1000 v82g1750 b0"}
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch PYTHONUNBUFFERED=1
MARGS=(); RTAGS=()
for tag in $TAGS; do
  if [ "$(ls "$OUT/$tag"/clip*/summary.json 2>/dev/null | wc -l)" = 149 ]; then
    MARGS+=(--arm "$tag=$OUT/$tag")
    [ "$tag" = b0 ] && RTAGS=(b0 "${RTAGS[@]}") || RTAGS+=("$tag")
  else
    echo "ARM_INCOMPLETE $tag $(ls "$OUT/$tag"/clip*/summary.json 2>/dev/null | wc -l)/149"
  fi
done
echo "E149F_START $(date -Is) node=$(hostname) arms: ${RTAGS[*]}"
[ ${#RTAGS[@]} -gt 0 ] || { echo "NO_COMPLETE_ARM"; exit 1; }
cd "$C"
OMP_NUM_THREADS=14 "$PY" realism_arms.py "$OUT" "$OUT/realism_arms.json" "${RTAGS[@]}" > "$OUT/realism.log" 2>&1 &
RPID=$!
n=$(ls "$OUT"/metrics/state_*.pt 2>/dev/null | wc -l)
echo "metric states: $n/8"
rc=1
if [ "$n" = 8 ]; then
  OMP_NUM_THREADS=2 "$PY" compute_eval149_metrics.py --base-config "$CFG" "${MARGS[@]}" --state "$OUT/metrics/unused.pt" \
    --output "$OUT/metrics/report.json" --merge "$OUT"/metrics/state_*.pt > "$OUT/metrics/merge.log" 2>&1
  if grep -q METRICS_DONE "$OUT/metrics/merge.log"; then cat "$OUT/metrics/report.json"; rc=0; else echo "MERGE_FAILED"; tail -20 "$OUT/metrics/merge.log"; fi
fi
wait $RPID
tail -40 "$OUT/realism.log"
echo "EVAL149_HPC3_DONE $(date -Is) rc=$rc"
exit $rc
