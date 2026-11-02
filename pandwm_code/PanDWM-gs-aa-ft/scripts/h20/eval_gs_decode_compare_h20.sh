#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="$PANDWM_ROOT/envs/pandwm"
CFG="${CFG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_nuscenes6_stitch_g2.json}"
RUN="${RUN:-$PANDWM_ROOT/runs/gs_nuscenes6_stitch_g2}"
STEP="${STEP:-2500}"
HOLDOUT_PKL="${HOLDOUT_PKL:-$PANDWM_ROOT/data/pickle/nuscenes_interp_12Hz_infos_holdout_p2.pkl}"
STAGE2_CONFIG="${STAGE2_CONFIG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_h20.json}"
STAGE2_RUN="${STAGE2_RUN:-$PANDWM_ROOT/runs/cd_baseline}"
STAGE2_STEP="${STAGE2_STEP:-10000}"
EMA="${EMA:-1}"
CLIP_LIMIT="${CLIP_LIMIT:-0}"
GAUSSIAN_RGB_BLEND="${GAUSSIAN_RGB_BLEND:-0.0}"
OUT="${OUT:-$RUN/eval_decode_compare/$STEP}"

for path in \
    "$CFG" \
    "$RUN/checkpoints/$STEP.pth" \
    "$HOLDOUT_PKL" \
    "$STAGE2_CONFIG"; do
    [[ -f "$path" ]] || {
        printf 'Required file is missing: %s\n' "$path" >&2
        exit 1
    }
done
if [[ "$EMA" != "0" && "$EMA" != "1" ]]; then
    printf 'EMA must be 0 or 1, got: %s\n' "$EMA" >&2
    exit 1
fi
if (( CLIP_LIMIT < 0 )); then
    printf 'CLIP_LIMIT must be non-negative, got: %s\n' "$CLIP_LIMIT" >&2
    exit 1
fi
python3 - "$GAUSSIAN_RGB_BLEND" <<'PY'
import sys
value = float(sys.argv[1])
if not 0.0 <= value <= 1.0:
    raise SystemExit("GAUSSIAN_RGB_BLEND must be in [0, 1]")
PY

stage2_suffix=""
if [[ "$EMA" == "1" ]]; then
    stage2_suffix="_ema"
fi
[[ -f "$STAGE2_RUN/checkpoints/${STAGE2_STEP}${stage2_suffix}.pth" ]] || {
    printf 'Stage-2 checkpoint is missing: %s\n' \
        "$STAGE2_RUN/checkpoints/${STAGE2_STEP}${stage2_suffix}.pth" >&2
    exit 1
}

# The Stage-2 stack, G2, rasterizer and LPIPS share one card.  The selector
# refuses to launch below this headroom and never terminates co-tenant jobs.
export PANDWM_MIN_FREE_MB="${PANDWM_MIN_FREE_MB:-70000}"
pandwm_pick_cuda_device

mkdir -p "$PANDWM_ROOT/logs" "$PANDWM_ROOT/torchinductor_cache"
eval "$($CONDA_BIN shell.bash hook)"
conda activate "$ENV_PREFIX"

export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29519}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$PANDWM_ROOT/torchinductor_cache"
export PANDWM_G1_DECODE_BATCH="${PANDWM_G1_DECODE_BATCH:-2}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$PANDWM_ROOT/logs/eval_gs_decode_compare_step${STEP}_${TIMESTAMP}.log"
printf 'PanDWM matched no-GS / GS / GT evaluation\n' | tee "$LOG"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
printf 'GS config/run/step: %s / %s / %s\n' \
    "$CFG" "$RUN" "$STEP" | tee -a "$LOG"
printf 'Stage-2 config/run/step/EMA: %s / %s / %s / %s\n' \
    "$STAGE2_CONFIG" "$STAGE2_RUN" "$STAGE2_STEP" "$EMA" | tee -a "$LOG"
printf 'Clip limit: %s (0 means complete holdout)\n' "$CLIP_LIMIT" | tee -a "$LOG"
printf 'Gaussian RGB blend diagnostic: %s\n' "$GAUSSIAN_RGB_BLEND" | tee -a "$LOG"
printf 'Output: %s\nLog: %s\n' "$OUT" "$LOG" | tee -a "$LOG"

cmd=(
    python "$PANDWM_ROOT/code/PanDWM/src/dwm_pano/evaluate_gs_decode_compare.py"
    -c "$CFG"
    -r "$RUN"
    -s "$STEP"
    --holdout-pkl "$HOLDOUT_PKL"
    --stage2-config "$STAGE2_CONFIG"
    --stage2-run "$STAGE2_RUN"
    --stage2-step "$STAGE2_STEP"
    --clip-limit "$CLIP_LIMIT"
    --gaussian-rgb-blend "$GAUSSIAN_RGB_BLEND"
    --out "$OUT"
)
if [[ "$EMA" == "1" ]]; then
    cmd+=(--stage2-ema)
fi
"${cmd[@]}" 2>&1 | tee -a "$LOG"
