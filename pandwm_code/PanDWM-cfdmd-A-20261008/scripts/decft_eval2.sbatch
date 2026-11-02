#!/usr/bin/env bash
# 2026-10-12 19:20: copy of scripts/decft_eval.sbatch for further decoder variants (weight interpolations of the
# original and the run3a decoder).  REDEC_FROM=<an earlier eval>/v83h2250_redec is symlinked in and decode149.py runs
# with --skip-redec (that arm depends only on the latents), so only v83h2250_dft and vaert_dft are decoded.
#   DECODER_CKPT=<mix.pt> DECEV_OUT=<dir> REDEC_FROM=<dir> sbatch scripts/decft_eval2.sbatch
#SBATCH --job-name=pandwm_decev
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/decoder_ft_20261012/logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
W=$DR/pandwm_runs/decoder_ft_20261012
PY=$DR/pandwm_venv/bin/python
CFG=$C/cd_eval_frozen149_hpc3.json
CK=${DECODER_CKPT:-$W/run2/decoder_6000.pt}
SRC=$DR/pandwm_runs/eval149_hpc3_v83_20261012/v83h2250
E=${DECEV_OUT:-$W/eval}
export PYTHONPATH=$C:$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
echo "DECEV_START $(date -Is) node=$(hostname) decoder=$CK redec_from=${REDEC_FROM:-}"
[ -f "$CK" ] || { echo "DECODER_MISSING $CK"; exit 1; }
mkdir -p "$E/metrics"
[ -e "$E/b0" ] || ln -s "$DR/pandwm_runs/eval149_hpc3_20261011/b0" "$E/b0"
[ -e "$E/vaert" ] || ln -s "$DR/pandwm_runs/eval149_hpc3_20261011/vaert" "$E/vaert"
REDEC_FROM=${REDEC_FROM:?set REDEC_FROM to an earlier v83h2250_redec}
[ -e "$E/v83h2250_redec" ] || ln -s "$REDEC_FROM" "$E/v83h2250_redec"
cd "$C"
"$PY" "$W/code/decode149.py" --base-config "$CFG" --arm-root "$SRC" --tag v83h2250 --decoder "$CK" --out "$E" --skip-redec \
  || { echo "DECODE_FAILED"; exit 1; }
"$PY" realism_arms.py "$E" "$E/realism_arms.json" v83h2250_redec v83h2250_dft vaert_dft vaert > "$E/realism.log" 2>&1 &
RP=$!
nice "$PY" flicker_diag.py "$E" "$E/flicker_diag.json" --pair v83h2250_redec v83h2250_dft \
  b0 vaert vaert_dft v83h2250_redec v83h2250_dft > "$E/flicker_diag.log" 2>&1 &
FP=$!
PANDWM_DECODER_CKPT=$CK "$PY" "$W/code/decoder_override_launch.py" compute_eval149_metrics.py --base-config "$CFG" \
  --arm "v83h2250_dft=$E/v83h2250_dft" --state "$E/metrics/state_dft.pt" --output "$E/metrics/report.json" \
  --clip-start 0 --clip-end 149 > "$E/metrics/metrics.log" 2>&1
wait $RP; wait $FP
grep -q DECODER_OVERRIDE "$E/metrics/metrics.log" || echo "WARNING: no DECODER_OVERRIDE line in metrics.log"
if grep -q METRICS_DONE "$E/metrics/metrics.log"; then cat "$E/metrics/report.json"; else echo "METRICS_FAILED"; fi
tail -3 "$E/realism.log"; tail -6 "$E/flicker_diag.log"
echo "DECEV_DONE $(date -Is)"
