#!/usr/bin/env bash
# 2026-10-11 19:20: phase 2 of the split HPC3 evaluation in ONE job (the QOS allows 50 submitted jobs per user and the
# 40-task generation array plus other projects' jobs left room for one more): FID / FVD over all 149 clips for every
# complete arm in a single compute_eval149_metrics.py run (writes metrics/report.json itself, no merge), and look stats
# / frame-to-frame change (realism_arms.py, b0 first as in realism3.py) on the CPU at the same time.  Exits 0 only when
# the report was written (v7 run D waits on this job with afterok).
#   sbatch --dependency=afterany:<generation array> scripts/eval149_hpc3_fin1.sbatch
#SBATCH --job-name=pandwm_e149f
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=03:00:00
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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch PYTHONUNBUFFERED=1
MARGS=(); RTAGS=()
for tag in $TAGS; do
  n=$(ls "$OUT/$tag"/clip*/summary.json 2>/dev/null | wc -l)
  if [ "$n" = 149 ]; then
    MARGS+=(--arm "$tag=$OUT/$tag")
    [ "$tag" = b0 ] && RTAGS=(b0 "${RTAGS[@]}") || RTAGS+=("$tag")
  else
    echo "ARM_INCOMPLETE $tag $n/149"
  fi
done
echo "E149F_START $(date -Is) node=$(hostname) arms: ${RTAGS[*]}"
[ ${#RTAGS[@]} -gt 0 ] || { echo "NO_COMPLETE_ARM"; exit 1; }
mkdir -p "$OUT/metrics"
cd "$C"
OMP_NUM_THREADS=8 "$PY" realism_arms.py "$OUT" "$OUT/realism_arms.json" "${RTAGS[@]}" > "$OUT/realism.log" 2>&1 &
RPID=$!
rc=1
for attempt in 1 2; do
  OMP_NUM_THREADS=4 "$PY" compute_eval149_metrics.py --base-config "$CFG" "${MARGS[@]}" \
    --state "$OUT/metrics/state_0_149.pt" --output "$OUT/metrics/report.json" --clip-start 0 --clip-end 149 \
    >> "$OUT/metrics/metrics.log" 2>&1
  if grep -q METRICS_DONE "$OUT/metrics/metrics.log"; then cat "$OUT/metrics/report.json"; rc=0; break; fi
  echo "$(date +%T) metrics attempt $attempt incomplete: $(tail -2 "$OUT/metrics/metrics.log" | tr '\n' ' ')"
done
wait $RPID
tail -40 "$OUT/realism.log"
echo "EVAL149_HPC3_DONE $(date -Is) rc=$rc"
exit $rc
