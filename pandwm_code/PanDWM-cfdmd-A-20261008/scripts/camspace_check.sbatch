#!/usr/bin/env bash
# 2026-10-20: validation of the 256x448 six-camera band protocol (cam_band_check.py): pixel check recorded canvas ->
# bridge vs recorded 256x448 views, mask coverage over Val-150, preview of one clip.
#SBATCH --job-name=pandwm_camchk
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=01:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
PY=$DR/pandwm_venv/bin/python
B=$DR/pandwm_runs/baselines_20261019
R=$DR/pandwm_runs
OUT=${OUT:-$R/camspace_20261020/check}
export PYTHONPATH=$C:$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=$DR/.cache/torch OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$C" || exit 1
echo "CAMCHK_START $(date -Is) $(hostname)"
"$PY" cam_band_check.py --base-config cd_eval_frozen149_hpc3.json --rec448 $B/rec448_val150 --unimlvg $B/unimlvg_P_s42 \
  --ours "$R/eval149_hpc3_v83_20261012/v84i4250+$R/tests_20261018/val_v84i4250_s42" --out "$OUT" ${CLIPS:+--clips $CLIPS} \
  ${PREVIEW:+--preview $PREVIEW} ${GEOMETRY:+--geometry $GEOMETRY} ${SKIP_COV:+--skip-coverage}
rc=$?
echo "CAMCHK_END rc=$rc $(date -Is)"
exit $rc
