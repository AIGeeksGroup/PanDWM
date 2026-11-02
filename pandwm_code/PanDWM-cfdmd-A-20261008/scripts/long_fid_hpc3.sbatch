#!/usr/bin/env bash
# 2026-10-14: early / late window FID of the 24-video long dev set (eval_v7/long_fid.py), one GPU.  A diagnostic for
# report section 26, not a gate.   ARMS="v83h2000 v83h4000 v84i4250 v84i4250raw" OUTNAME=long_fid_4250 sbatch ...
#SBATCH --job-name=pandwm_lfid
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
LV=$DR/pandwm_runs/long_videos_hpc3_20261011
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=$DR/.cache/torch PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
ARMS=${ARMS:?set ARMS}; OUTNAME=${OUTNAME:?set OUTNAME}
cd "$C" || exit 1
echo "LFID_START $(date -Is) node=$(hostname) arms: $ARMS"
"$DR/pandwm_venv/bin/python" long_fid.py "$LV" "$LV/$OUTNAME.json" --scenes 0 14 21 32 41 49 62 82 10 31 52 72 \
    --seeds 42 43 --arms $ARMS
echo "LFID_END $(date -Is) rc=$?"
