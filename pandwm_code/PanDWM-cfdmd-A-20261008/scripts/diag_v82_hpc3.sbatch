#!/usr/bin/env bash
# 2026-10-11: v8.2 diagnostics on the HPC3 149-clip run (eval149_hpc3_20261011):
#   1) VAE round-trip floor: GT frames encoded and decoded by the SD3.5 VAE, written as arm "vaert" (vae_roundtrip149.py);
#   2) realism_arms.py on b0 / vaert / v81f1600 / v82g2000 (GT from b0, as realism3.py);
#   3) flicker_diag.py: low/high-frequency split of the within-window and window-boundary change, per-clip
#      distributions, motion tertiles, sharpness-vs-change link for v8.1 1600 -> v8.2 2000.
#   sbatch scripts/diag_v82_hpc3.sbatch
#SBATCH --job-name=pandwm_diag82
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=02:30:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
CFG=$C/cd_eval_frozen149_hpc3.json
OUT=$DR/pandwm_runs/eval149_hpc3_20261011
PY=$DR/pandwm_venv/bin/python
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch OMP_NUM_THREADS=12 PYTHONUNBUFFERED=1
cd "$C"
echo "DIAG82_START $(date -Is) node=$(hostname)"
"$PY" vae_roundtrip149.py --base-config "$CFG" --out-root "$OUT/vaert" 2>&1 | grep -vE "^\[vaert\] clip [0-9]*[1-9] " | tail -20
n=$(ls "$OUT"/vaert/clip*/summary.json 2>/dev/null | wc -l)
echo "VAERT_CLIPS $n"
[ "$n" = 149 ] || { echo "VAERT_INCOMPLETE"; exit 1; }
"$PY" realism_arms.py "$OUT" "$OUT/realism_vaert.json" b0 vaert v81f1600 v82g2000 2>&1 | tail -30
"$PY" flicker_diag.py "$OUT" "$OUT/flicker_diag.json" --pair v81f1600 v82g2000 b0 vaert v7d1000 v81f1600 v82g2000 2>&1 | tail -12
echo "DIAG82_END $(date -Is)"
