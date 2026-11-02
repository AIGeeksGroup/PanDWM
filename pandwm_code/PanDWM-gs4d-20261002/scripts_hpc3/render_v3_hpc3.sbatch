#!/usr/bin/env bash
#SBATCH --job-name=pandwm_gsr3h
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=01:30:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs4d_hpc3_20261017/logs/%x_%A_%a.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
# 2026-10-20: HPC3 twin of PanDWM-gs-aa-ft/render_v3_hpc2_array.sbatch (dwm_pano.export_gs_v3_clips) for the 149 review
# clips of v84i4250 (HPC3 copies in eval149_hpc3_v83_20261012/v84i4250), config configs/gs_review_hpc3.json.
#   TAG=<tag> [CKPT=...] [CHUNK=20] EXTRA="..." sbatch --array=0-7 render_v3_hpc3.sbatch
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/da3_env_hpc3.sh
export MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID} % 20000 ))
CHUNK=${CHUNK:-20}
S=$(( ${SLURM_ARRAY_TASK_ID:-0} * CHUNK )); E=$(( S + CHUNK )); if (( E > 149 )); then E=149; fi
CK=${CKPT:-$RUNS/c2_hpc2/checkpoints/3000.pth}
OUT=$DR/pandwm_runs/review_render_v3_20261020/${TAG:?TAG}
mkdir -p $OUT
cd $GSAA
echo "GSR3H_START tag=$TAG ckpt=$CK clips=$S..$E extra=[${EXTRA:-}] node=$(hostname) $(date -Is)"
$PY -m dwm_pano.export_gs_v3_clips -c $GS4D/configs/gs_review_hpc3.json \
  --data-pkl $DR/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl --checkpoint $CK \
  --clips-root $DR/pandwm_runs/eval149_hpc3_v83_20261012/v84i4250 --start $S --end $E --out-root $OUT \
  --time-budget-s ${BUDGET_S:-4800} --da3-src $DA3_SRC --da3-anyview $DA3_ANYVIEW --da3-metric $DA3_METRIC ${EXTRA:-}
rc=$?
echo "GSR3H_END rc=$rc clips=$S..$E $(date -Is)"
exit $rc
