#!/usr/bin/env bash
# 2026-10-12: branch H = v8.3 (pipeline_v8_3, configs/cfdmd_H_gan_v8_3_hpc3_8gpu.json: v8.2 with the reworked motion match:
# seam step included, blur-proof low/high-frequency split, smooth log-ratio push, rollout window too), seeded from
# seeds/F_1600 like v8.2.  Copy of train_v8_2.sbatch with the H names; its original header:
# 2026-10-11: branch G = v8.2 (pipeline_v8_2, configs/cfdmd_G_gan_v8_2_hpc3_8gpu.json: v8.1 + first-window motion match,
# see V8_2_CHANGES_20261011.md; seeded from branch F).  It only resumes: the branch directory must first be seeded with a
# hard-linked v7 run-D checkpoint (scripts/seed_branch.py); it never starts from B0.  Same campaign logic as
# train_v7.sbatch otherwise (pause file, latest committed checkpoint, same data offset), no OOM fallback config.
#SBATCH --job-name=pandwm_cfdmd_H
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -euo pipefail
source /path/to/pandwm_root/pandwm_venv/bin/activate
CF=/path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008
DEMO=/path/to/pandwm_root/pandwm_code/PanDWM_long_horizon_20260928
export CAMPAIGN_DIR=/path/to/pandwm_root/pandwm_runs/cfdmd_20261010_H
export PAUSE_FILE=$CAMPAIGN_DIR/PAUSE_REQUESTED
export PYTHONPATH=$CF/src:$DEMO/vendor:$DEMO/src:/path/to/pandwm_root/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/path/to/pandwm_root/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR || true
trap 'touch "$PAUSE_FILE"; echo "PAUSE_SIGNAL_RECEIVED"' USR1 TERM
if [[ -e "$PAUSE_FILE" ]]; then
    echo "Campaign is paused; remove PAUSE_REQUESTED only when resume is requested."
    exit 0
fi
OUT=$CAMPAIGN_DIR/gan
CONFIG=$CF/configs/cfdmd_H_gan_v8_3_hpc3_8gpu.json
STEP=$(python3 - "$OUT" <<'PY'
import sys
from dwm_pano_cfdmd.checkpoint_io import latest_step
step = latest_step(sys.argv[1], 8)
print("none" if step is None else step)
PY
)
if [[ "$STEP" == none ]]; then
    echo "BRANCH_NOT_SEEDED: no committed checkpoint in $OUT (run scripts/seed_branch.py first)"
    exit 1
fi
if (( STEP >= 2000 )); then echo "ARM_ALREADY_COMPLETE gan step=$STEP"; exit 0; fi
echo "PLAN_H_START $(date -Is) node=$(hostname) job=$SLURM_JOB_ID resume=$STEP"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$CF/scripts/test_rank_optimizer_resume.py"
cp "$CONFIG" "$OUT/log/config_used_job_${SLURM_JOB_ID}.json"
echo "ARM_START gan resume=$STEP config=$(basename "$CONFIG") $(date -Is)"
set +e
python -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$CF/scripts/train_resumable_v7.py" -c "$CONFIG" -o "$OUT" \
    --log-steps 10 --preview-steps 100000000 --evaluation-steps 100000000 \
    --checkpointing-steps 100000000 --max-steps 2000 --resume-from "$STEP" --resume-data-offset "$STEP" &
TRAIN_PID=$!
wait "$TRAIN_PID"
RESULT=$?
# A trap can interrupt wait while the child is still checkpointing.
while kill -0 "$TRAIN_PID" 2>/dev/null; do wait "$TRAIN_PID"; RESULT=$?; done
set -e
if [[ -e "$PAUSE_FILE" ]]; then
    echo "CAMPAIGN_PAUSED gan checkpoint safely requested; rc=$RESULT $(date -Is)"
    exit 0
fi
if (( RESULT == 0 )); then echo "ARM_COMPLETE gan $(date -Is)"; echo "PLAN_H_COMPLETE $(date -Is)"; exit 0; fi
echo "ARM_FAILED gan rc=$RESULT"; exit "$RESULT"
