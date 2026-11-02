#!/usr/bin/env bash
# 2026-10-14: long-video diagnostics with eval_v7/long_analyze2.py (memory-light) and a DISJOINT real pool: the Dev-78
# real frames minus the 12 clips that show the evaluated scenes' first windows (eval-149 clips 9 58 64 73 84 85 95 106
# 110 121 135 146; PanDWM_eval149_20261011/clip_tokens_20261014.json).  The pool numbers of report 27.3 used all 78 clips.
#   STAGE=main      the 4 existing arms, 24 videos and day-only (scenes 72 / 82 are night)
#   STAGE=gt7       features of the GT-restart arm + A1 (after 716693)
#   STAGE=controls  features of teacher / no-CFG teacher / history noise / moment_align + A2, A3 + colour (after all)
#SBATCH --job-name=pandwm_ldiag3
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
LV=$DR/pandwm_runs/long_videos_hpc3_20261011
FD=$LV/feats
PY=$DR/pandwm_venv/bin/python
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=$DR/.cache/torch PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
ALL="0 14 21 32 41 49 62 82 10 31 52 72"
DAY="0 14 21 32 41 49 62 10 31 52"
POOL=(--pool dev78gt --pool-exclude-clips 9 58 64 73 84 85 95 106 110 121 135 146)
STAGE=${STAGE:?STAGE}
cd "$C" || exit 1
echo "LDIAG3_START $(date -Is) node=$(hostname) stage=$STAGE"
have() { [[ -f $FD/feats_$1.npz ]]; }
if [[ $STAGE == main ]]; then
  "$PY" long_analyze2.py "$FD" "$LV/long_analyze2_4250.json" --arms v83h2000 v83h4000 v84i4250 v84i4250raw \
    "${POOL[@]}" --boot 300
  "$PY" long_analyze2.py "$FD" "$LV/long_analyze2_4250_day.json" --arms v83h2000 v83h4000 v84i4250 v84i4250raw \
    --scenes $DAY "${POOL[@]}" --boot 300
fi
if [[ $STAGE == gt7 ]]; then
  "$PY" long_feats.py "$FD" --arm "v84i4250gt7=$LV" --scenes $ALL --seeds 42 43 --first-window 7 || echo "FEATS_FAILED gt7"
  have v84i4250gt7 && "$PY" long_analyze2.py "$FD" "$LV/long_analyze2_gt7.json" --arms v84i4250 v84i4250gt7 \
    "${POOL[@]}" --boot 300
fi
if [[ $STAGE == controls ]]; then
  for t in teacher50 teacher50nocfg v84i4250fn05 v84i4250fn02 v84i4250ma05; do
    "$PY" long_feats.py "$FD" --arm "$t=$LV" --scenes $ALL --seeds 42 || echo "FEATS_FAILED $t"
  done
  arms=""; for t in teacher50 teacher50nocfg; do have $t && arms="$arms $t"; done
  if [[ -n $arms ]]; then
    echo "== A2 teacher (all 12 scenes, then day-only)"
    "$PY" long_analyze2.py "$FD" "$LV/long_analyze2_teacher.json" --arms v84i4250 $arms v83h2000 --seeds 42 \
      "${POOL[@]}" --boot 300
    "$PY" long_analyze2.py "$FD" "$LV/long_analyze2_teacher_day.json" --arms v84i4250 $arms v83h2000 --seeds 42 \
      --scenes $DAY "${POOL[@]}" --boot 300
  fi
  arms=""; for t in v84i4250fn05 v84i4250fn02 v84i4250ma05; do have $t && arms="$arms $t"; done
  if [[ -n $arms ]]; then
    echo "== A3 inference knobs"
    "$PY" long_analyze2.py "$FD" "$LV/long_analyze2_knobs.json" --arms v84i4250 $arms --seeds 42 "${POOL[@]}" --boot 300
  fi
  echo "== colour"
  carms="v84i4250"; for t in teacher50 teacher50nocfg v84i4250fn05 v84i4250fn02 v84i4250ma05; do have $t && carms="$carms $t"; done
  OMP_NUM_THREADS=4 "$PY" long_color.py "$LV" "$LV/long_color_controls.json" --arms $carms --scenes $ALL --seeds 42
fi
echo "LDIAG3_END $(date -Is) stage=$STAGE"
