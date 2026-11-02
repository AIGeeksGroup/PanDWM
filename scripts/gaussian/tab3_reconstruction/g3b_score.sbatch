#!/usr/bin/env bash
#SBATCH --partition=<partition>
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs_recon_g3b_20261101/logs/%x_%A_%a.log
# 2026-11-01 G3b tab:gs_recon scores: gs_recon_ext249_20261024/scripts/ext_score.sbatch with this run's job file, the
# unchanged scorer (snapshot score.py, md5 3bc20a25...) on CPU.   JOBFILE=<file> sbatch --array=0-11 g3b_score.sbatch
set -uo pipefail
source $HOME/pandwm_env.sh
X=$HOME/pandwm_runs/gs_recon_ext249_20261024
mapfile -t ALL < <(grep -v '^#' ${JOBFILE:?JOBFILE} | grep -v '^\s*$')
read -r M RS T MK RR SMP OUTF <<< "${ALL[${SLURM_ARRAY_TASK_ID:-0}]}"
[ "$SMP" = "-" ] && SMP=""
cd $X/snapshot/PanDWM-gs-aa-ft
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1
md5sum src/dwm_pano/recon/score.py
echo "SCORE_START $M $RS targets=$T masks=$MK renders=$RR/$M out=$OUTF node=$(hostname) $(date -Is)"
PYTHONPATH=src python -m dwm_pano.recon.score --targets $T --masks $MK --renders $RR/$M --res $RS --out $OUTF \
  ${SMP:+--samples $SMP}
rc=$?
echo "SCORE_END $M $RS rc=$rc $(date -Is)"
exit $rc
