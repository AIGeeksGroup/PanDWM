#!/usr/bin/env bash
# 2026-10-20: six-camera band features at 256x448 (cam_band_feats.py) of Val-150 arms, one dataset pass for all ARMS.
#   ARMS="rec448=views:$B/rec448_val150 recband=recband unimlvg_P_s42=views:$B/unimlvg_P_s42" sbatch camspace_feats.sbatch
#   ARMS="P_s42=latents:<root>+<root2> ..." REFS="P_s42=$FV/P_s42.npz" sbatch camspace_feats.sbatch
# Output: $FD/feats_<name>.npz per geometry ({geometry} in FD; existing files are kept).
# TOPROWS=52 masks native rows 0..51 for every arm (MagicDrive-V2 padding); use a separate FD then.
#SBATCH --job-name=pandwm_camfeat
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
PY=$DR/pandwm_venv/bin/python
# (not ${FD:-...{geometry}}: with FD set, bash closes the expansion at the first brace and appends a stray one)
[[ -n ${FD:-} ]] || FD="$DR/pandwm_runs/tests_20261018/feats_val150_cam_{geometry}"
GEOMETRY=${GEOMETRY:-bridge,onebev_inverse,onebev_inverse_own}
export PYTHONPATH=$C:$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=$DR/.cache/torch OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$C" || exit 1
args=()
for a in $ARMS; do args+=(--arm "$a"); done
for r in ${REFS:-}; do args+=(--source-ref "$r"); done
echo "CAMFEAT_START $(date -Is) $(hostname) fd=$FD ${args[*]}"
"$PY" cam_band_feats.py --base-config cd_eval_frozen149_hpc3.json --out-dir "$FD" "${args[@]}" \
  ${POSITIONS:+--positions $POSITIONS} ${GEOMETRY:+--geometry $GEOMETRY} ${TOPROWS:+--mask-top-native-rows $TOPROWS}
rc=$?
echo "CAMFEAT_END rc=$rc $(date -Is)"
exit $rc
