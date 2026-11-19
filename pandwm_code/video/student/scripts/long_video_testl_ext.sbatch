#!/usr/bin/env bash
# copy of long_video_testl.sbatch with LIST overridable (Test-L extension to 83 scenes).
# Test-L, the long-video test set: the 71 test
# scenes of eval/review149_test71_long12.json, 3 recorded frames + 12 windows x 16 (19.5 s)
# from the scene start; windows 1-2 are the Dev-149 clip and are not scored.  Student, clean history,
# stochastic 4-step sampler, blend 128, steps [1000, 750, 500, 250].
# Array task i runs list indices [i*CHUNK, (i+1)*CHUNK) (the list's dev76_index field holds the Dev-149 clip index);
# output <LVT>/<TAG>_scene<index>_seed<SEED>.  A time budget stops at a window boundary; resubmit to resume.
#   TAG=v84i4250 CKPT=<..._ema.pth> SEED=42 sbatch --array=0-7%4 scripts/long_video_testl.sbatch
#   TAG=teacher50 CKPT=<stage1> ENGINE=teacher CHUNK=2 sbatch --array=0-35%4 scripts/long_video_testl.sbatch
#SBATCH --job-name=pandwm_lvtl
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs_eval149/%x_%A_%a.log
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/eval
CFG=$C/cd_eval_frozen149_hpc3.json
LIST=${LIST:-$C/review149_test71_long12.json}
LV=${LV_OUT:-$DR/pandwm_runs/long_videos_testl}
PY=$DR/pandwm_venv/bin/python
T=${TAG:?TAG}; CK=${CKPT:?CKPT}; SEED=${SEED:-42}; CHUNK=${CHUNK:-9}; WIN=${WINDOWS:-12}
ENGINE=${ENGINE:-student}; HM=${HISTORY_MODE:-clean}; SIG=${NOISE_SIGMA:-0.0}
STEPS=${STEPS:-1000 750 500 250}
export PYTHONPATH=$DR/pandwm_code/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=12 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR PANDWM_HISTORY_GUIDANCE || true
EXTRA=()
[[ $ENGINE == teacher ]] && EXTRA=(--teacher-guidance-scale "${TEACHER_CFG:-3.0}")
[[ -f "$CK" ]] || { echo "CKPT_MISSING $CK"; exit 1; }
mkdir -p "$LV"
cd "$C" || exit 1
TASK=${SLURM_ARRAY_TASK_ID:-0}
# list POSITIONS [i*CHUNK, (i+1)*CHUNK) -> their dev76_index keys (Dev-149 clip indices)
IDX=$(python3 -c "
import json
rows = json.load(open('$LIST'))['scenes']
print(' '.join(str(r['dev76_index']) for r in rows[$TASK * $CHUNK:($TASK + 1) * $CHUNK]))")
[[ -n $IDX ]] || { echo "NO_SCENES task $TASK"; exit 0; }
echo "LVTL_START $(date -Is) node=$(hostname) tag=$T task=$TASK indices=[$IDX] seed=$SEED windows=$WIN steps=[$STEPS]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export MASTER_PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$PY" lowmem_launch.py run_long_rollout_multi.py --base-config "$CFG" --checkpoint "$CK" --output-root "$LV" \
  --tag "$T" --scene-list "$LIST" --indices $IDX --seed "$SEED" --windows "$WIN" --history-mode "$HM" \
  --noise-sigma "$SIG" --engine "$ENGINE" --student-steps $STEPS "${EXTRA[@]}" --time-budget-s 10200
rc=$?
n=0; for i in $IDX; do [[ -f $LV/${T}_scene${i}_seed${SEED}/gt_vs_generated.mp4 ]] && n=$((n + 1)); done
echo "LVTL_END $T task=$TASK rc=$rc videos_done=$n $(date -Is)"
exit $rc
