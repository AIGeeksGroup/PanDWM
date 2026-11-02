#!/usr/bin/env bash
# 2026-10-19: panorama canvases + six-camera grids of selected Dev-149 clips (pano_canvas_dump.py), six-camera figure.
# ARM/OUT/POSITIONS/FRAMES can be overridden.
#SBATCH --job-name=pandwm_canvas
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
PY=$DR/pandwm_venv/bin/python
ARM=${ARM:-$DR/pandwm_runs/eval149_hpc3_v83_20261012/v84i4250}
OUT=${OUT:-$DR/pandwm_runs/front_band_20261019/v84i4250_s42_canvas}
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=$DR/.cache/torch OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$C" || exit 1
echo "CANVAS_START $(date -Is) node=$(hostname) arm=$ARM out=$OUT"
"$PY" pano_canvas_dump.py --base-config cd_eval_frozen149_hpc3.json --manifest dev149_manifest_20261018.json \
  --dataset-stride 20 --arm-root "$ARM" --out "$OUT" --positions "$POSITIONS" --frames "${FRAMES:-25}" --blend ${BLEND:-128}
rc=$?
echo "CANVAS_END rc=$rc $(date -Is)"
exit $rc
