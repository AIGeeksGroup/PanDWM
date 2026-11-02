#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

DUMP_DIR="${DUMP_DIR:-$PANDWM_ROOT/eval_dumps/full_cd10000ema_googlefvd_seed0}"
RUN="${RUN:-$PANDWM_ROOT/runs/cd_baseline}"
STEP="${STEP:-10000}"
EMA="${EMA:-1}"
GENERATOR_SEED="${GENERATOR_SEED:-0}"
MASTER_PORT="${MASTER_PORT:-29536}"
RESULT="$DUMP_DIR/google_fvd_pano.json"

if [[ -e "$DUMP_DIR" ]]; then
    printf 'Refusing to append to existing dump directory: %s\n' \
        "$DUMP_DIR" >&2
    exit 1
fi
mkdir -p "$DUMP_DIR" "$PANDWM_ROOT/logs"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$PANDWM_ROOT/logs/eval_stage2_full_google_fvd_${TIMESTAMP}.log"

printf 'Stage-2 full dump followed by Google-reference FVD\n' | tee "$LOG"
printf 'Dump: %s\nResult: %s\n' "$DUMP_DIR" "$RESULT" | tee -a "$LOG"

env -u CUDA_VISIBLE_DEVICES \
    PANDWM_FID_DUMP_DIR="$DUMP_DIR" \
    PANDWM_MIN_FREE_MB="${PANDWM_MIN_FREE_MB:-90000}" \
    RUN="$RUN" STEP="$STEP" EMA="$EMA" \
    GENERATOR_SEED="$GENERATOR_SEED" MASTER_PORT="$MASTER_PORT" \
    bash "$SCRIPT_DIR/eval_4step_h20.sh" 2>&1 | tee -a "$LOG"

export TFHUB_CACHE_DIR="${TFHUB_CACHE_DIR:-$PANDWM_ROOT/ckpts/tfhub_cache}"
export PYTHONPATH="$PANDWM_ROOT/code/PanDWM/src"
"$PANDWM_ROOT/envs/admeval/bin/python" \
    -m dwm_pano.metrics.google_fvd_pano \
    --dump-dir "$DUMP_DIR" \
    --sequence-length 16 \
    --batch-size 16 \
    --expect-clips 149 \
    --out "$RESULT" 2>&1 | tee -a "$LOG"

printf 'Completed full Stage-2 metric chain: %s\n' "$RESULT" | tee -a "$LOG"
