#!/usr/bin/env bash
# 2026-10-15: teacher fine-tune v3 evaluation on Dev-76-L scenes (eval_v7/run_long_rollout_v3.py; report section 30).
#   TAG=tr_gt2 CKPT=<teacher ckpt> WINDOWS=2 TW=scene_description sbatch --array=0-3 scripts/teacher_eval_v3.sbatch
#   TAG=tr_d7 CKPT=... START=7 WINDOWS=7 HFROM=<root>/v84i4250 TW=... CHUNK=38 sbatch --array=0-1 ...
# 50-step teacher, CFG 3, seed 42; array task i runs dev76 indices [i*CHUNK, (i+1)*CHUNK).  Output
# $DR/pandwm_runs/teacher_v3_20261015/eval/<TAG>_scene<idx>_seed42.  SIGMA > 0 noises the history (both engines get
# reference timestep 1000 SIGMA).
#SBATCH --job-name=pandwm_tv3ev
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs_eval149/%x_%A_%a.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
T=$DR/pandwm_code/PanDWM_teacher_v3_20261015
CFG=$C/cd_eval_frozen149_hpc3.json
LIST=$C/dev76_long10_20261014.json
OUTR=${OUT_ROOT:-$DR/pandwm_runs/teacher_v3_20261015/eval}
PY=$DR/pandwm_venv/bin/python
TG=${TAG:?TAG}; CK=${CKPT:?CKPT}; CHUNK=${CHUNK:-19}; WIN=${WINDOWS:-2}; START=${START:-1}
TW=${TW:-fixed}; SIG=${SIGMA:-0.0}; HFROM=${HFROM:-}; GS=${TEACHER_CFG:-3.0}; ENGINE=${ENGINE:-teacher}
export PYTHONPATH=$T/src:$C:$C/vendor:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=12 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR PANDWM_HISTORY_GUIDANCE || true
[[ -f "$CK" ]] || { echo "CKPT_MISSING $CK"; exit 1; }
mkdir -p "$OUTR"
cd "$C" || exit 1
TASK=${SLURM_ARRAY_TASK_ID:-0}
IDX=$(python3 -c "
import json
n = len(json.load(open('$LIST'))['scenes'])
print(' '.join(str(i) for i in range($TASK * $CHUNK, min(n, ($TASK + 1) * $CHUNK))))")
[[ -n $IDX ]] || { echo "NO_SCENES task $TASK"; exit 0; }
EXTRA=()
[[ -n $HFROM ]] && EXTRA+=(--history-from "$HFROM")
echo "TV3EV_START $(date -Is) node=$(hostname) tag=$TG task=$TASK indices=[$IDX] start=$START windows=$WIN tw=$TW sigma=$SIG hfrom=${HFROM:-gt}"
export MASTER_PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$PY" lowmem_launch.py "$DR/pandwm_code/PanDWM-cfdmd-A-20261008/eval_v7/run_long_rollout_v3.py" \
  --base-config "$CFG" --checkpoint "$CK" --output-root "$OUTR" --tag "$TG" --scene-list "$LIST" --indices $IDX \
  --seed 42 --windows "$WIN" --start-window "$START" --engine "$ENGINE" --teacher-guidance-scale "$GS" \
  --history-sigma "$SIG" --time-weather "$TW" --time-budget-s 13800 "${EXTRA[@]}"
rc=$?
n=0; for i in $IDX; do [[ -f $OUTR/${TG}_scene${i}_seed42/gt_vs_generated.mp4 ]] && n=$((n + 1)); done
echo "TV3EV_END $TG task=$TASK rc=$rc videos_done=$n $(date -Is)"
exit $rc
