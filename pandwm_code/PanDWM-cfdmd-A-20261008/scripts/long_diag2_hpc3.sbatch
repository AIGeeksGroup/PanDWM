#!/usr/bin/env bash
# 2026-10-14: analysis of the report-section-27 control rollouts, one GPU, after the generation jobs
# (sbatch --dependency=afterany:<jobs> scripts/long_diag2_hpc3.sbatch).  Features per arm (an arm whose videos are
# incomplete is skipped, not fatal), then:
#   A1 GT restart      v84i4250 vs v84i4250gt7 on windows 7-12, 24 videos
#   A2 teacher         teacher50 (CFG 3), teacher50nocfg vs v84i4250 / v83h2000, seed 42, 12 videos
#   A3 inference knobs v84i4250 vs history noise 0.05 / 0.02 and moment_align 0.5, seed 42
#   colour / brightness per window (long_color.py, with the day / night split), seed 42
#SBATCH --job-name=pandwm_ldiag2
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
cd "$C" || exit 1
echo "LDIAG2_START $(date -Is) node=$(hostname)"
"$PY" long_feats.py "$FD" --arm "v84i4250gt7=$LV" --scenes $ALL --seeds 42 43 --first-window 7 || echo "FEATS_FAILED v84i4250gt7"
for t in teacher50 teacher50nocfg v84i4250fn05 v84i4250fn02 v84i4250ma05; do
  "$PY" long_feats.py "$FD" --arm "$t=$LV" --scenes $ALL --seeds 42 || echo "FEATS_FAILED $t"
done
have() { [[ -f $FD/feats_$1.npz ]]; }
if have v84i4250gt7; then
  echo "== A1 GT restart"
  "$PY" long_analyze.py "$FD" "$LV/long_analyze_gt7.json" --arms v84i4250 v84i4250gt7 --pool dev78gt --boot 300
fi
arms=""; for t in teacher50 teacher50nocfg; do have $t && arms="$arms $t"; done
if [[ -n $arms ]]; then
  echo "== A2 teacher"
  "$PY" long_analyze.py "$FD" "$LV/long_analyze_teacher.json" --arms v84i4250 $arms v83h2000 --seeds 42 --pool dev78gt --boot 300
fi
arms=""; for t in v84i4250fn05 v84i4250fn02 v84i4250ma05; do have $t && arms="$arms $t"; done
if [[ -n $arms ]]; then
  echo "== A3 inference knobs"
  "$PY" long_analyze.py "$FD" "$LV/long_analyze_knobs.json" --arms v84i4250 $arms --seeds 42 --pool dev78gt --boot 300
fi
echo "== colour"
carms="v84i4250"; for t in teacher50 teacher50nocfg v84i4250fn05 v84i4250fn02 v84i4250ma05; do have $t && carms="$carms $t"; done
OMP_NUM_THREADS=4 "$PY" long_color.py "$LV" "$LV/long_color_diag2.json" --arms $carms --scenes $ALL --seeds 42
echo "LDIAG2_END $(date -Is)"
