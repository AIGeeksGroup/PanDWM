#!/usr/bin/env bash
# 2026-10-14: Dev-76-L, the larger long-video dev set (eval_v7/V82_EVAL_HPC3_20261011.md section 27; the user agreed
# 2026-10-14 to switch the long-video evaluation to it).  The 76 non-test nuScenes val scenes with a valid 163-frame
# clip (3 real + 10 windows x 16 = 16.3 s), PanDWM_eval149_20261011/dev76_long10_20261014.json (dataset order; holds the
# 12 scenes of the old 24-video set).  Same rollout protocol as long_video_hpc3_v3.sbatch (student, clean history,
# stochastic 4-step sampler, blend 128), but one process per array task (run_long_rollout_multi.py, model loaded once;
# outputs equal the single-scene runner's bit for bit, verified by scripts/verify_multi_rollout.sbatch).
# Array task i runs dev76 indices [i*CHUNK, (i+1)*CHUNK); output <LV76>/<TAG>_scene<dev76_index>_seed<SEED>, read
# by long_feats.py with --scenes 0 .. 75 --windows 10.  A time budget stops at a window boundary; resubmit to resume.
#   TAG=v84i4250 CKPT=<..._ema.pth> sbatch --array=0-7%4 scripts/long_video_dev76.sbatch
#SBATCH --job-name=pandwm_lv76
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs_eval149/%x_%A_%a.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
CFG=$C/cd_eval_frozen149_hpc3.json
LIST=$C/dev76_long10_20261014.json
LV=${LV_OUT:-$DR/pandwm_runs/long_videos_dev76_20261014}
PY=$DR/pandwm_venv/bin/python
T=${TAG:?TAG}; CK=${CKPT:?CKPT}; SEED=${SEED:-42}; CHUNK=${CHUNK:-10}; WIN=${WINDOWS:-10}
ENGINE=${ENGINE:-student}; HM=${HISTORY_MODE:-clean}; SIG=${NOISE_SIGMA:-0.0}
# 2026-10-17: STEPS = student step list (default = the old hard-coded one); the paper uses STEPS="1000 750 500 250"
STEPS=${STEPS:-1000 625 250 42}
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=12 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR PANDWM_HISTORY_GUIDANCE || true
EXTRA=()
[[ $ENGINE == teacher ]] && EXTRA=(--teacher-guidance-scale "${TEACHER_CFG:-3.0}")
[[ -f "$CK" ]] || { echo "CKPT_MISSING $CK"; exit 1; }
# the one-process runner must have passed its equality check against the single-scene runner
VLOG=$(ls -t /path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008/logs_eval149/pandwm_vmulti_*.log 2>/dev/null | head -1)
grep -q "VERIFY_MULTI PASS" "${VLOG:-/dev/null}" || { echo "MULTI_NOT_VERIFIED ${VLOG:-none}"; exit 1; }
mkdir -p "$LV"
cd "$C" || exit 1
TASK=${SLURM_ARRAY_TASK_ID:-0}
IDX=$(python3 -c "
import json
n = len(json.load(open('$LIST'))['scenes'])
print(' '.join(str(i) for i in range($TASK * $CHUNK, min(n, ($TASK + 1) * $CHUNK))))")
[[ -n $IDX ]] || { echo "NO_SCENES task $TASK"; exit 0; }
echo "LV76_START $(date -Is) node=$(hostname) tag=$T task=$TASK indices=[$IDX] seed=$SEED windows=$WIN steps=[$STEPS]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export MASTER_PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$PY" lowmem_launch.py run_long_rollout_multi.py --base-config "$CFG" --checkpoint "$CK" --output-root "$LV" \
  --tag "$T" --scene-list "$LIST" --indices $IDX --seed "$SEED" --windows "$WIN" --history-mode "$HM" \
  --noise-sigma "$SIG" --engine "$ENGINE" --student-steps $STEPS "${EXTRA[@]}" --time-budget-s 10200
rc=$?
n=0; for i in $IDX; do [[ -f $LV/${T}_scene${i}_seed${SEED}/gt_vs_generated.mp4 ]] && n=$((n + 1)); done
echo "LV76_END $T task=$TASK rc=$rc videos_done=$n $(date -Is)"
exit $rc
