#!/usr/bin/env bash
#SBATCH --partition=<partition>
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs_recon_g3b/logs/%x_%A_%a.log
# Score rendered reconstruction targets on CPU with recon/score.py; one job-file line per array task.
#   JOBFILE=<file> sbatch --array=0-11 g3b_score.sbatch
set -uo pipefail
source $HOME/pandwm_env.sh
X=$HOME/pandwm_runs/gs_recon_ext249
mapfile -t ALL < <(grep -v '^#' ${JOBFILE:?JOBFILE} | grep -v '^\s*$')
read -r M RS T MK RR SMP OUTF <<< "${ALL[${SLURM_ARRAY_TASK_ID:-0}]}"
[ "$SMP" = "-" ] && SMP=""
cd $X/snapshot/gaussian/gs3d
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1
md5sum src/dwm_pano/recon/score.py
echo "SCORE_START $M $RS targets=$T masks=$MK renders=$RR/$M out=$OUTF node=$(hostname) $(date -Is)"
PYTHONPATH=src python -m dwm_pano.recon.score --targets $T --masks $MK --renders $RR/$M --res $RS --out $OUTF \
  ${SMP:+--samples $SMP}
rc=$?
echo "SCORE_END $M $RS rc=$rc $(date -Is)"
exit $rc
