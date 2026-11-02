#!/usr/bin/env bash
# Publication-oriented 3+16 speed benchmark.  Unlike normal evaluation this
# must not share compute with another job: co-residency changes latency even
# when enough memory remains.  The launcher therefore requires both a large
# free-memory margin and near-zero GPU utilization.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="$PANDWM_ROOT/envs/pandwm"
GS_CFG="${GS_CFG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_prod_rd0p01_rl0p01.json}"
GS_RUN="${GS_RUN:-$PANDWM_ROOT/runs/gs_prod_rd0p01_rl0p01}"
GS_STEP="${GS_STEP:-15000}"
STAGE2_CFG="${STAGE2_CFG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_h20.json}"
STAGE2_RUN="${STAGE2_RUN:-$PANDWM_ROOT/runs/cd_baseline}"
STAGE2_STEP="${STAGE2_STEP:-10000}"
STAGE2_EMA="${STAGE2_EMA:-1}"
DISABLE_50STEP_CFG="${DISABLE_50STEP_CFG:-0}"
CLIP_INDEX="${CLIP_INDEX:-0}"
WARMUP="${WARMUP:-1}"
REPEATS="${REPEATS:-3}"
OUT="${OUT:-$PANDWM_ROOT/benchmarks/speed_prod15k_stage2ema.json}"
MIN_FREE_MB="${MIN_FREE_MB:-80000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-5}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MASTER_PORT="${MASTER_PORT:-29562}"

for boolean_pair in \
    "STAGE2_EMA:$STAGE2_EMA" \
    "DISABLE_50STEP_CFG:$DISABLE_50STEP_CFG"
do
    name="${boolean_pair%%:*}"
    value="${boolean_pair#*:}"
    if [[ "$value" != "0" && "$value" != "1" ]]; then
        printf '%s must be 0 or 1, got: %s\n' "$name" "$value" >&2
        exit 1
    fi
done
for integer_pair in \
    "WARMUP:$WARMUP" "REPEATS:$REPEATS" \
    "MIN_FREE_MB:$MIN_FREE_MB" "MAX_GPU_UTIL:$MAX_GPU_UTIL" \
    "WAIT_TIMEOUT_SECONDS:$WAIT_TIMEOUT_SECONDS" "POLL_SECONDS:$POLL_SECONDS"
do
    name="${integer_pair%%:*}"
    value="${integer_pair#*:}"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        printf '%s must be a non-negative integer, got: %s\n' \
            "$name" "$value" >&2
        exit 1
    fi
done
if [[ "$REPEATS" -lt 1 || "$POLL_SECONDS" -lt 1 ]]; then
    printf 'REPEATS and POLL_SECONDS must be at least 1\n' >&2
    exit 1
fi

stage2_suffix=""
[[ "$STAGE2_EMA" == "1" ]] && stage2_suffix="_ema"
for required in \
    "$GS_CFG" "$GS_RUN/checkpoints/$GS_STEP.pth" \
    "$STAGE2_CFG" "$STAGE2_RUN/checkpoints/${STAGE2_STEP}${stage2_suffix}.pth"
do
    if [[ ! -f "$required" ]]; then
        printf 'Required benchmark input is missing: %s\n' "$required" >&2
        exit 1
    fi
done
if [[ ! -x "$CONDA_BIN" ]]; then
    printf 'Conda executable not found: %s\n' "$CONDA_BIN" >&2
    exit 1
fi
if [[ ! -f "$ENV_PREFIX/conda-meta/history" ]]; then
    printf 'Conda environment is missing: %s\n' "$ENV_PREFIX" >&2
    exit 1
fi

pick_idle_gpu() {
    local table selection
    table="$(nvidia-smi \
        --query-gpu=index,memory.free,utilization.gpu \
        --format=csv,noheader,nounits)"
    selection="$(printf '%s\n' "$table" | awk -F ',' \
        -v min_free="$MIN_FREE_MB" -v max_util="$MAX_GPU_UTIL" '
        {
            idx=$1; free=$2; util=$3
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", idx)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", free)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", util)
            if (free + 0 >= min_free + 0 && util + 0 <= max_util + 0 &&
                (!found || free + 0 > best_free + 0)) {
                found=1; best_idx=idx; best_free=free; best_util=util
            }
        }
        END { if (found) print best_idx ":" best_free ":" best_util }
    ')"
    if [[ -n "$selection" ]]; then
        export CUDA_VISIBLE_DEVICES="${selection%%:*}"
        printf 'Selected physical GPU %s: free=%s MiB, util=%s%%\n' \
            "$CUDA_VISIBLE_DEVICES" \
            "$(cut -d: -f2 <<<"$selection")" \
            "$(cut -d: -f3 <<<"$selection")"
        return 0
    fi
    printf 'No benchmark-safe GPU (need free >= %s MiB and util <= %s%%).\n' \
        "$MIN_FREE_MB" "$MAX_GPU_UTIL" >&2
    printf '%s\n' "$table" >&2
    return 1
}

if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    printf 'Refusing preset CUDA_VISIBLE_DEVICES for a publication benchmark; unset it so the utilization guard can select and verify a card.\n' >&2
    exit 1
fi

wait_start="$(date +%s)"
last_report="$wait_start"
while ! pick_idle_gpu; do
    now="$(date +%s)"
    elapsed=$((now - wait_start))
    if [[ "$WAIT_TIMEOUT_SECONDS" -eq 0 || "$elapsed" -ge "$WAIT_TIMEOUT_SECONDS" ]]; then
        printf 'Timed out waiting for a benchmark-safe GPU after %s seconds.\n' \
            "$elapsed" >&2
        exit 1
    fi
    if (( now - last_report >= 300 )); then
        printf 'Still waiting for a benchmark-safe GPU (%s seconds elapsed).\n' \
            "$elapsed" >&2
        last_report="$now"
    fi
    sleep "$POLL_SECONDS"
done

mkdir -p "$(dirname -- "$OUT")" "$PANDWM_ROOT/logs" \
    "$PANDWM_ROOT/torchinductor_cache"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$ENV_PREFIX"

export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$PANDWM_ROOT/torchinductor_cache"

timestamp="$(date +%Y%m%d_%H%M%S)"
log="$PANDWM_ROOT/logs/benchmark_speed_${timestamp}.log"
printf 'PanDWM segmented speed benchmark\n' | tee "$log"
printf 'Physical CUDA device: %s\n' "$CUDA_VISIBLE_DEVICES" | tee -a "$log"
printf 'Protocol: batch=1, clip=%s, warmup=%s, repeats=%s, disable_50step_cfg=%s\n' \
    "$CLIP_INDEX" "$WARMUP" "$REPEATS" "$DISABLE_50STEP_CFG" | tee -a "$log"
printf 'Output: %s\n' "$OUT" | tee -a "$log"

cmd=(
    python -m dwm_pano.benchmark_speed
    --gs-config "$GS_CFG" --gs-run "$GS_RUN" --gs-step "$GS_STEP"
    --stage2-config "$STAGE2_CFG" --stage2-run "$STAGE2_RUN"
    --stage2-step "$STAGE2_STEP"
    --clip-index "$CLIP_INDEX" --warmup "$WARMUP" --repeats "$REPEATS"
    --out "$OUT"
)
[[ "$STAGE2_EMA" == "1" ]] && cmd+=(--stage2-ema)
[[ "$DISABLE_50STEP_CFG" == "1" ]] && cmd+=(--disable-50step-cfg)
"${cmd[@]}" 2>&1 | tee -a "$log"
