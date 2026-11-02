#!/usr/bin/env bash
# 2026-10-16: long videos for the user's 149-scene review (user: "149个全做", current best v84i4250 EMA temporarily
# locked).  Copy of scripts/long_video_dev76.sbatch (same protocol: student, clean history, stochastic 4-step sampler,
# blend 128, run_long_rollout_multi.py) with three changes:
#   - LIST is required (scenes[i] = {dev76_index: <Dev-149 clip index>, scene_token}); LV_OUT is required;
#   - array task i runs list POSITIONS [i*CHUNK, (i+1)*CHUNK) and passes their dev76_index keys as --indices, so the
#     outputs are named <TAG>_scene<Dev-149 clip index>_seed<SEED>;
#   - WINDOWS_TRY="9 8 7 ..." tries each window count in turn until every scene of the task has a video (scenes whose
#     recording is too short for more windows get the longest clip that fits).
# 2026-10-18: STEPS = student step list, default = the paper/training list [1000, 750, 500, 250]
# (LOCK_20261017_PAPER.json, user decision 2026-10-17); the 10-16 review videos used the old hard-coded
# [1000, 625, 250, 42] (pass STEPS="1000 625 250 42" to reproduce them).
#   LIST=<json> LV_OUT=<dir> TAG=v84i4250 CKPT=<..._ema.pth> WINDOWS=12 CHUNK=9 sbatch --array=0-7 scripts/long_video_review149.sbatch
#SBATCH --job-name=pandwm_lv149
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
LIST=${LIST:?LIST}
LV=${LV_OUT:?LV_OUT}
PY=$DR/pandwm_venv/bin/python
T=${TAG:?TAG}; CK=${CKPT:?CKPT}; SEED=${SEED:-42}; CHUNK=${CHUNK:-10}; WIN=${WINDOWS:-10}
ENGINE=${ENGINE:-student}; HM=${HISTORY_MODE:-clean}; SIG=${NOISE_SIGMA:-0.0}
STEPS=${STEPS:-1000 750 500 250}
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
rows = json.load(open('$LIST'))['scenes']
print(' '.join(str(r['dev76_index']) for r in rows[$TASK * $CHUNK:($TASK + 1) * $CHUNK]))")
[[ -n $IDX ]] || { echo "NO_SCENES task $TASK"; exit 0; }
echo "LV149_START $(date -Is) node=$(hostname) tag=$T task=$TASK indices=[$IDX] seed=$SEED windows=${WINDOWS_TRY:-$WIN} steps=[$STEPS]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export MASTER_PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
rc=1
for W in ${WINDOWS_TRY:-$WIN}; do
  "$PY" lowmem_launch.py run_long_rollout_multi.py --base-config "$CFG" --checkpoint "$CK" --output-root "$LV" \
    --tag "$T" --scene-list "$LIST" --indices $IDX --seed "$SEED" --windows "$W" --history-mode "$HM" \
    --noise-sigma "$SIG" --engine "$ENGINE" --student-steps $STEPS "${EXTRA[@]}" --time-budget-s 10200
  rc=$?
  n=0; for i in $IDX; do [[ -f $LV/${T}_scene${i}_seed${SEED}/gt_vs_generated.mp4 ]] && n=$((n + 1)); done
  echo "LV149_TRY windows=$W rc=$rc videos_done=$n"
  [[ $n == $(echo $IDX | wc -w) ]] && { rc=0; break; }
done
echo "LV149_END $T task=$TASK rc=$rc videos_done=$n $(date -Is)"
exit $rc
