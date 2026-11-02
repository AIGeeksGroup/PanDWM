#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="$PANDWM_ROOT/envs/pandwm"
GS_CFG="${GS_CFG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_g2_ft_appearance_stage2mix_500.json}"
STAGE2_CFG="${STAGE2_CFG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_h20.json}"
STAGE2_RUN="${STAGE2_RUN:-$PANDWM_ROOT/runs/cd_baseline}"
STAGE2_STEP="${STAGE2_STEP:-10000}"
STAGE2_EMA="${STAGE2_EMA:-1}"
STEPS="${STEPS:-500}"
START_OFFSET="${START_OFFSET:-2500}"
OUT="${OUT:-$PANDWM_ROOT/cache/stage2_train_g2_seed0_offset2500_500}"
export PANDWM_MIN_FREE_MB="${PANDWM_MIN_FREE_MB:-70000}"

stage2_suffix=""
[[ "$STAGE2_EMA" == "1" ]] && stage2_suffix="_ema"
for path in "$GS_CFG" "$STAGE2_CFG" \
    "$STAGE2_RUN/checkpoints/${STAGE2_STEP}${stage2_suffix}.pth"; do
    [[ -f "$path" ]] || { printf 'Missing required file: %s\n' "$path" >&2; exit 1; }
done
(( STEPS > 0 )) || { printf 'STEPS must be positive\n' >&2; exit 1; }
(( START_OFFSET >= 0 )) || { printf 'START_OFFSET must be non-negative\n' >&2; exit 1; }
free_kb="$(df --output=avail "$PANDWM_ROOT" | tail -1 | tr -d ' ')"
(( free_kb >= 10 * 1024 * 1024 )) || {
    printf 'Refusing cache generation with less than 10 GiB free.\n' >&2
    exit 1
}

pandwm_pick_cuda_device
mkdir -p "$PANDWM_ROOT/logs" "$PANDWM_ROOT/torchinductor_cache" "$OUT"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$ENV_PREFIX"

export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29529}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$PANDWM_ROOT/torchinductor_cache"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

timestamp="$(date +%Y%m%d_%H%M%S)"
log="$PANDWM_ROOT/logs/cache_stage2_train_latents_${timestamp}.log"
printf 'Stage-2 selected-train-latent cache\nGPU: %s\nOut: %s\nLog: %s\n' \
    "$CUDA_VISIBLE_DEVICES" "$OUT" "$log" | tee "$log"
cmd=(python3 -m dwm_pano.cache_stage2_train_latents
    --gs-config "$GS_CFG"
    --stage2-config "$STAGE2_CFG"
    --stage2-run "$STAGE2_RUN"
    --stage2-step "$STAGE2_STEP"
    --steps "$STEPS"
    --start-offset "$START_OFFSET"
    --out "$OUT")
[[ "$STAGE2_EMA" == "1" ]] && cmd+=(--stage2-ema)
"${cmd[@]}" 2>&1 | tee -a "$log"
