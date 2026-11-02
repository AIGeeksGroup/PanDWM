#!/usr/bin/env bash
# 2026-10-24 copy of tests_feats.sbatch with ACD1-59 excluded (Test-S over 249 clips).
# 2026-10-18: clip_feats.py features of clip runs (and the recorded frames of the manifest), then optionally
# tests_analyze.py.  Features are written once per run directory: $FD/feats_<name>.npz; existing files are kept.
#   ROOTS="v84i4250_s42=$OUT/v84i4250_s42 ..." sbatch scripts/tests_feats.sbatch               (Test-S, + real)
#   ROOTS=... ANALYZE="--arm ours=FD/feats_a.npz,FD/feats_b.npz --pair ours:b0" NAME=main sbatch ...
#   MANIFEST=dev149_manifest_20261018.json STRIDE=20 FD=<dir> ROOTS=... (Dev sets)
#SBATCH --job-name=pandwm_tfeat
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58,ACD1-59
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
CFG=$C/cd_eval_frozen149_hpc3.json
PY=$DR/pandwm_venv/bin/python
OUT=${OUT:-$DR/pandwm_runs/tests_20261018}
FD=${FD:-$OUT/feats}
MANIFEST=${MANIFEST:-tests213_manifest_20261018.json}; STRIDE=${STRIDE:-5}
REALNAME=${REALNAME:-real_$(basename "$MANIFEST" .json)}
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$C" || exit 1
mkdir -p "$FD"
echo "TFEAT_START $(date -Is) node=$(hostname) manifest=$MANIFEST stride=$STRIDE fd=$FD"
fail=0
"$PY" ${PRELAUNCH:+$PRELAUNCH} clip_feats.py --base-config "$CFG" --manifest "$MANIFEST" --dataset-stride "$STRIDE" \
  --out "$FD/feats_${REALNAME}.npz" --real || fail=1
for spec in ${ROOTS:-}; do
  name=${spec%%=*}; root=${spec#*=}
  "$PY" ${PRELAUNCH:+$PRELAUNCH} clip_feats.py --base-config "$CFG" --manifest "$MANIFEST" --dataset-stride "$STRIDE" \
    --out "$FD/feats_${name}.npz" --arm-root "$root" || { echo "FEATS_FAIL $name"; fail=1; }
done
if [[ -n ${ANALYZE:-} && $fail == 0 ]]; then
  # shellcheck disable=SC2086
  "$PY" tests_analyze.py --real "$FD/feats_${REALNAME}.npz" ${ANALYZE//FD\//$FD/} --boot "${BOOT:-1000}" \
    --out "$FD/analysis_${NAME:-run}.json" || fail=1
fi
echo "TFEAT_END fail=$fail $(date -Is)"
exit $fail
