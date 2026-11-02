#!/usr/bin/env bash
# 2026-10-21 ablation: final analysis on Dev-78 (pre-registered in ABLATION_20261021.md section 3).
#   FID / FVD-1 / FVD-2: tests_analyze.py (PanDWM_eval149_20261011, unchanged) on the clip_feats.py features of every
#   arm x seed {42, 1000}, pairs CONTROL:arm (control minus arm), delete-one-scene jackknife, 1000 scene bootstraps for
#   the sign probability; flicker: flicker_stats.py (flicker_20261020, unchanged) on every gt_vs_generated.mp4, then
#   abl_flicker.py (dev rule, fixed reference = recorded half of eval149_hpc3_v83_20261012/v84i4250); table abl_table.py.
#   NAME=abl3 ARMS="A_ctrl B_noadv D_notemp b0" CONTROL=A_ctrl sbatch scripts/abl_analyze.sbatch
# Feature files: $EV/feats/feats_<arm>_s<seed>.npz; mp4 roots: $EV/abl_<arm>_s<seed> (b0 seed 42: the eval149 run whose
# features were reused, eval149_hpc3_v83_20261012/b0).
#SBATCH --job-name=pandwm_abl_an
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
ABL=$DR/pandwm_code/PanDWM-ablation-20261021
FLK=$DR/pandwm_code/flicker_20261020
EV=$DR/pandwm_runs/abl_20261021_eval
REF=$DR/pandwm_runs/eval149_hpc3_v83_20261012
PY=$DR/pandwm_venv/bin/python
NAME=${NAME:?NAME}; ARMS=${ARMS:?ARMS}; CONTROL=${CONTROL:-A_ctrl}; SEEDS=${SEEDS:-42 1000}
FD=$EV/feats; AD=$EV/analysis; FL=$EV/flicker
mkdir -p "$AD" "$FL"
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 MALLOC_ARENA_MAX=2
echo "ABL_ANALYZE_START $(date -Is) node=$(hostname) name=$NAME arms=[$ARMS] control=$CONTROL seeds=[$SEEDS]"
md5sum "$C/tests_analyze.py" "$FLK/flicker_stats.py" "$ABL/scripts/abl_flicker.py" "$ABL/scripts/abl_table.py"
fail=0
ARGS=(); FARGS=()
for arm in $ARMS; do
  files=(); stats=()
  for s in $SEEDS; do
    f=$FD/feats_${arm}_s${s}.npz
    [[ -s $f ]] || { echo "MISSING $f"; fail=1; }
    files+=("$f")
    root=$EV/abl_${arm}_s${s}
    [[ $arm == b0 && $s == 42 ]] && root=$REF/b0
    st=$FL/stats_${arm}_s${s}.json
    stats+=("$st")
    if [[ ! -s $st ]]; then
      "$PY" "$ABL/scripts/dev78_clips.py" "$C/dev78_manifest_20261018.json" "$FL/dev78_clips.json"
      "$PY" "$FLK/flicker_stats.py" --root "$root" --clips "$FL/dev78_clips.json" --out "$st" || { echo "FLICKER_FAIL $arm s$s"; fail=1; }
    fi
  done
  ARGS+=(--arm "$arm=$(IFS=,; echo "${files[*]}")")
  FARGS+=(--arm "$arm=$(IFS=,; echo "${stats[*]}")")
  [[ $arm != "$CONTROL" ]] && { ARGS+=(--pair "$CONTROL:$arm"); FARGS+=(--pair "$CONTROL:$arm"); }
done
RS=$FL/stats_ref_v84i4250_s42_eval149.json
if [[ ! -s $RS ]]; then
  "$PY" "$ABL/scripts/dev78_clips.py" "$C/dev78_manifest_20261018.json" "$FL/dev78_clips.json"
  "$PY" "$FLK/flicker_stats.py" --root "$REF/v84i4250" --clips "$FL/dev78_clips.json" --out "$RS" || fail=1
fi
[[ $fail == 0 ]] || { echo "ABL_ANALYZE FAIL (inputs)"; exit 1; }
cd "$C" || exit 1
"$PY" tests_analyze.py --real "$FD/feats_real_dev78_manifest_20261018.npz" "${ARGS[@]}" --boot 1000 \
  --out "$AD/analysis_${NAME}.json" || fail=1
"$PY" "$ABL/scripts/abl_flicker.py" --manifest "$C/dev78_manifest_20261018.json" --ref "$RS" "${FARGS[@]}" \
  --out "$AD/flicker_${NAME}.json" || fail=1
if [[ $fail == 0 ]]; then
  "$PY" "$ABL/scripts/abl_table.py" "$AD/analysis_${NAME}.json" "$AD/flicker_${NAME}.json" "$CONTROL" \
    > "$AD/TABLE_${NAME}.txt" && cat "$AD/TABLE_${NAME}.txt" || fail=1
fi
echo "ABL_ANALYZE_END name=$NAME fail=$fail $(date -Is)"
exit $fail
