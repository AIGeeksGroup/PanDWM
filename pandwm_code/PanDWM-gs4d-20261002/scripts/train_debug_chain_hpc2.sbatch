#!/usr/bin/env bash
#SBATCH --job-name=gs4d_dbg
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
# One 30-minute segment of a 4D training run on the debug partition.  Resumes
# from the newest complete checkpoint (decoder + optimizer, both loadable),
# continuing the data order with --resume-data-offset.  Resubmit until
# checkpoints/$STEPS.pth exists.
set -euo pipefail
source $HOME/pandwm_code/PanDWM-gs4d-20261002/scripts/common_env.sh
OUT=$RUNS/${RUN_TAG:-train_debug_v1}; STEPS=${STEPS:-1000}
mkdir -p $OUT
RESUME=$(python - "$OUT" <<'PY'
import os, sys, torch
out = sys.argv[1]
ck = os.path.join(out, "checkpoints")
steps = sorted((int(f[:-4]) for f in os.listdir(ck) if f.endswith(".pth")), reverse=True) \
    if os.path.isdir(ck) else []
for s in steps:
    try:
        torch.load(os.path.join(out, "checkpoints", f"{s}.pth"), map_location="cpu")
        torch.load(os.path.join(out, "optimizer", f"{s}.pth"), map_location="cpu")
        print(s); break
    except Exception:
        for d in ("checkpoints", "optimizer"):
            p = os.path.join(out, d, f"{s}.pth")
            if os.path.exists(p): os.rename(p, p + ".broken")
PY
)
EXTRA=()
[[ -n "$RESUME" ]] && EXTRA=(--resume-from $RESUME --resume-data-offset $RESUME)
# Each segment trains a fixed chunk and exits normally (100 steps, about 22 min at
# 13 s/step).  A TERM from `timeout` left dataloader workers hanging past the
# slurm limit once, and the node was drained ("Kill task failed").
CHUNK=${CHUNK:-100}   # multiple of the 50-step checkpoint cadence
SEG_END=$(( ${RESUME:-0} + CHUNK )); (( SEG_END > STEPS )) && SEG_END=$STEPS
echo "segment start $(date -Is) resume=${RESUME:-none} until=$SEG_END"
timeout -k 30 ${SEG_TIMEOUT:-1650} python $GSAA/src/dwm_pano/train.py -c $GS4D/configs/${CONFIG:-gs4d_hpc2.json} -o $OUT \
  --max-steps $SEG_END --log-steps 10 --preview-steps 100000000 \
  --evaluation-steps 100000000 --checkpointing-steps ${CKPT_STEPS:-50} \
  --keep-last-ckpts 3 "${EXTRA[@]}" || echo "segment ended rc=$?"
echo "segment end $(date -Is)"
