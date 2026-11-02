#!/usr/bin/env bash
# 2026-10-11 19:10: phase 2a of the split HPC3 evaluation (see eval149_hpc3_gen.sbatch): FID / FVD for clip range
# i of the 8 ranges, all COMPLETE arms (149 clips) at once, with the frozen protocol's metric code
# (compute_eval149_metrics.py); eval149_hpc3_fin.sbatch merges the 8 states.  Every task derives the same arm set from
# the clip counts, which are final once the generation array has ended.
#   sbatch --array=0-7 --dependency=afterany:<gen array> scripts/eval149_hpc3_met.sbatch
#SBATCH --job-name=pandwm_e149m
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=01:00:00
#SBATCH --output=logs_eval149/%x_%A_%a.log
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
export PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
RANGES=(0 19 38 57 76 95 114 132 149)
r=${SLURM_ARRAY_TASK_ID:?array task}; s=${RANGES[$r]}; e=${RANGES[$((r + 1))]}
MARGS=()
for tag in $TAGS; do
  [ "$(ls "$OUT/$tag"/clip*/summary.json 2>/dev/null | wc -l)" = 149 ] && MARGS+=(--arm "$tag=$OUT/$tag")
done
echo "E149M_START $(date -Is) node=$(hostname) range=[$s,$e) arms: ${MARGS[*]}"
[ ${#MARGS[@]} -gt 0 ] || { echo "NO_COMPLETE_ARM"; exit 1; }
mkdir -p "$OUT/metrics"
cd "$C"
for attempt in 1 2; do
  "$PY" compute_eval149_metrics.py --base-config "$CFG" "${MARGS[@]}" --state "$OUT/metrics/state_${s}_$e.pt" \
    --output "$OUT/metrics/report_$s.json" --clip-start $s --clip-end $e 2>&1 | tee -a "$OUT/metrics/met_$s.log" | grep -vE "^\[metrics\] clip"
  grep -q SHARD_DONE "$OUT/metrics/met_$s.log" && { echo "E149M_DONE [$s,$e)"; exit 0; }
  echo "$(date +%T) metrics attempt $attempt incomplete"
done
exit 1
