#!/usr/bin/env bash
# 2026-10-13: FID / FVD of several branch-H milestones on the Dev-78 selection set (the Dev-149 clips whose scene is not
# a Test-S / Test-L scene; dev149_nontest_split_20261013.json), with compute_eval149_metrics_subset.py.  Writes only to
# $OUT/metrics_dev78/; never touches metrics/state_0_149.pt or metrics/report.json of the main evaluation.
#   TAGS="v83h3000 v83h3250 v83h3500 v83h3750" LABEL=a sbatch scripts/eval149_hpc3_dev78.sbatch
#   CLIP_LIST=<json> overrides the clip list (smoke tests).
#SBATCH --job-name=pandwm_e149d78
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
OUT=${EVAL_OUT:-$DR/pandwm_runs/eval149_hpc3_v83_20261012}
PY=$DR/pandwm_venv/bin/python
TAGS=${TAGS:?set TAGS}
LABEL=${LABEL:?set LABEL}
LIST=${CLIP_LIST:-$C/dev149_nontest_split_20261013.json}
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch PYTHONUNBUFFERED=1
MARGS=()
for tag in $TAGS; do
  n=$(ls "$OUT/$tag"/clip*/summary.json 2>/dev/null | wc -l)
  [ "$n" = 149 ] || { echo "ARM_INCOMPLETE $tag $n/149"; exit 1; }
  MARGS+=(--arm "$tag=$OUT/$tag")
done
D=$OUT/metrics_dev78
mkdir -p "$D"
echo "E149D78_START $(date -Is) node=$(hostname) arms: $TAGS list: $LIST label: $LABEL"
cd "$C"
rc=1
for attempt in 1 2; do
  OMP_NUM_THREADS=4 "$PY" compute_eval149_metrics_subset.py --base-config "$CFG" "${MARGS[@]}" --clip-list "$LIST" \
    --state "$D/state_$LABEL.pt" --output "$D/report_$LABEL.json" >> "$D/metrics_$LABEL.log" 2>&1
  if grep -q METRICS_DONE "$D/metrics_$LABEL.log"; then cat "$D/report_$LABEL.json"; rc=0; break; fi
  echo "$(date +%T) attempt $attempt incomplete: $(tail -2 "$D/metrics_$LABEL.log" | tr '\n' ' ')"
done
echo "E149D78_DONE $(date -Is) rc=$rc"
exit $rc
