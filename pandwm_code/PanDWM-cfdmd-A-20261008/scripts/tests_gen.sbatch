#!/usr/bin/env bash
# 2026-10-18: generate a manifest of 35-frame clips (Test-S, Dev-78, Dev-149) with run_clips_videos.py, one model per
# job, array task i = manifest positions [i*CHUNK, (i+1)*CHUNK).  Finished clips are skipped, so a rerun resumes.
# Output: $OUT/<TAG>_s<SEED>/clipNNN/{latents.pt,gt_vs_generated.mp4,summary.json} (NNN = test_index).
#   TAG=v84i4250 CKPT=<..._ema.pth> SEED=42 sbatch --array=0-7%4 scripts/tests_gen.sbatch          (Test-S, student)
#   TAG=teacher50 CKPT=<stage1> ENGINE=teacher CHUNK=4 sbatch --array=0-53%4 --time=03:00:00 scripts/tests_gen.sbatch
#   TAG=stage0_50 CKPT=<stage0> ENGINE=teacher BIDIR=1 ...                                         (bidirectional)
#   TAG=v84i4250G HISTORY=none MANIFEST=dev78_manifest_20261018.json STRIDE=20 ...                  (setting G)
# Settings fixed by LOCK_20261017_PAPER.json: stochastic 4-step sampler, steps [1000, 750, 500, 250], blend 128.
#SBATCH --job-name=pandwm_tgen
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=logs_eval149/%x_%A_%a.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
CFG=$C/cd_eval_frozen149_hpc3.json
PY=$DR/pandwm_venv/bin/python
OUT=${OUT:-$DR/pandwm_runs/tests_20261018}
T=${TAG:?TAG}; CK=${CKPT:?CKPT}; SEED=${SEED:-42}; CHUNK=${CHUNK:-27}
MANIFEST=${MANIFEST:-tests213_manifest_20261018.json}; STRIDE=${STRIDE:-5}
ENGINE=${ENGINE:-student}; HISTORY=${HISTORY:-recorded}; BIDIR=${BIDIR:-0}; TSTEPS=${TEACHER_STEPS:-50}
BUDGET=${BUDGET:-6600}   # seconds; set it to about (time limit - 10 min) when --time is changed
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR PANDWM_HISTORY_GUIDANCE || true
[[ -f "$CK" ]] || { echo "CKPT_MISSING $CK"; exit 1; }
cd "$C" || exit 1
[[ -f "$MANIFEST" ]] || { echo "MANIFEST_MISSING $MANIFEST"; exit 1; }
# the runner must have passed its equality check against run_eval149_videos.py
VLOG=$(ls -t $DR/pandwm_code/PanDWM-cfdmd-A-20261008/logs_eval149/pandwm_vclips_*.log 2>/dev/null | head -1)
grep -q "VERIFY_CLIPS_EQUAL PASS" "${VLOG:-/dev/null}" || { echo "CLIPS_RUNNER_NOT_VERIFIED ${VLOG:-none}"; exit 1; }
TASK=${SLURM_ARRAY_TASK_ID:-0}
N=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['count'])")
S=$((TASK * CHUNK)); E=$(( (TASK + 1) * CHUNK < N ? (TASK + 1) * CHUNK : N ))
(( S < N )) || { echo "NO_CLIPS task $TASK"; exit 0; }
D=$OUT/${T}_s${SEED}
EXTRA=(--engine "$ENGINE" --history "$HISTORY")
[[ $ENGINE == teacher ]] && EXTRA+=(--teacher-steps "$TSTEPS" --teacher-guidance-scale "${TEACHER_CFG:-3.0}")
[[ $BIDIR == 1 ]] && EXTRA+=(--bidirectional)
echo "TGEN_START $(date -Is) node=$(hostname) tag=$T seed=$SEED task=$TASK positions=[$S,$E) manifest=$MANIFEST stride=$STRIDE ${EXTRA[*]} budget=${BUDGET}s ckpt=$CK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$D"
"$PY" ${PRELAUNCH:+$PRELAUNCH} lowmem_launch.py run_clips_videos.py --base-config "$CFG" --checkpoint "$CK" --out-root "$D" \
  --manifest "$MANIFEST" --dataset-stride "$STRIDE" --positions $S $E --seed "$SEED" "${EXTRA[@]}" \
  --time-budget-s "$BUDGET"
rc=$?
n=$(python3 -c "
import json, os
rows = json.load(open('$MANIFEST'))['clips'][$S:$E]
print(sum(os.path.isfile('$D/clip%03d/summary.json' % r['test_index']) for r in rows))")
echo "TGEN_END $T s$SEED task=$TASK rc=$rc clips_done=$n/$((E - S)) $(date -Is)"
[[ $n == $((E - S)) ]]
