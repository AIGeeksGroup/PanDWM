#!/usr/bin/env bash
# 2026-10-12: evaluate a fine-tuned SD3.5 VAE decoder, 1 GPU (after scripts/decft_train.sbatch).
#   1) eval_v7/decode149.py: v8.3 step-2250 latents decoded with the original decoder (v83h2250_redec, like-for-like
#      baseline) and the fine-tuned one (v83h2250_dft), plus the GT round trip with the fine-tuned one (vaert_dft);
#   2) on the CPU meanwhile: realism_arms.py and flicker_diag.py (b0 and the original vaert symlinked in);
#   3) FID/FVD of v83h2250_dft: compute_eval149_metrics.py through eval_v7/decoder_override_launch.py.  The baseline
#      is the existing v83h2250 report (same script, original decoder).
#   DECODER_CKPT=<decoder_N.pt> sbatch --dependency=afterok:<train job> scripts/decft_eval.sbatch
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
echo "DECEV_START $(date -Is) node=$(hostname) decoder=$CK"
[ -f "$CK" ] || { echo "DECODER_MISSING $CK"; exit 1; }
mkdir -p "$E/metrics"
[ -e "$E/b0" ] || ln -s "$DR/pandwm_runs/eval149_hpc3_20261011/b0" "$E/b0"
[ -e "$E/vaert" ] || ln -s "$DR/pandwm_runs/eval149_hpc3_20261011/vaert" "$E/vaert"
cd "$C"
"$PY" "$W/code/decode149.py" --base-config "$CFG" --arm-root "$SRC" --tag v83h2250 --decoder "$CK" --out "$E" \
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
