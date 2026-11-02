#!/usr/bin/env bash
# 2026-10-19: CAM_FRONT band dumps of the Dev-149 v84i4250 run for the qualitative comparison with the September
# front-view baselines (front_band_dump.py). ARM/OUT can be overridden for other seeds.
#SBATCH --job-name=pandwm_fband
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
OUT=${OUT:-$DR/pandwm_runs/front_band_20261019/v84i4250_s42}
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=$DR/.cache/torch OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$C" || exit 1
echo "FBAND_START $(date -Is) node=$(hostname) arm=$ARM out=$OUT"
"$PY" front_band_dump.py --base-config cd_eval_frozen149_hpc3.json --manifest dev149_manifest_20261018.json \
  --dataset-stride 20 --arm-root "$ARM" --out "$OUT" ${POSITIONS:+--positions $POSITIONS} --blend ${BLEND:-128} \
  ${CAMS:+--cams $CAMS} ${RENDER:+--render $RENDER} ${FRAMES:+--frames $FRAMES}
rc=$?
echo "FBAND_END rc=$rc $(date -Is)"
exit $rc
