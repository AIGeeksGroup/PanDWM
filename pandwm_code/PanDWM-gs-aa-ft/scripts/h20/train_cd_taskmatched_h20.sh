#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="$PANDWM_ROOT/envs/pandwm"
BASE_CFG="${BASE_CFG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_h20.json}"
TEACHER="${TEACHER:-$PANDWM_ROOT/ckpts/stage1_causal_20000.pth}"
RUN="${RUN:-$PANDWM_ROOT/runs/cd_taskmatched_stage1_20000}"
GPU_COUNT="${GPU_COUNT:-4}"
MIN_FREE_MB="${MIN_FREE_MB:-70000}"
MAX_STEPS="${MAX_STEPS:-500}"
LOG_STEPS="${LOG_STEPS:-10}"
PREVIEW_STEPS="${PREVIEW_STEPS:-250}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-500}"
EVAL_ITEMS="${EVAL_ITEMS:-5}"
MASTER_PORT="${MASTER_PORT:-29540}"

require_positive() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || {
        printf '%s must be a positive integer: %s\n' "$1" "$2" >&2
        exit 1
    }
}
for pair in \
    "GPU_COUNT:$GPU_COUNT" "MIN_FREE_MB:$MIN_FREE_MB" \
    "MAX_STEPS:$MAX_STEPS" "LOG_STEPS:$LOG_STEPS" \
    "PREVIEW_STEPS:$PREVIEW_STEPS" \
    "CHECKPOINT_STEPS:$CHECKPOINT_STEPS" "EVAL_ITEMS:$EVAL_ITEMS"; do
    require_positive "${pair%%:*}" "${pair#*:}"
done
if [[ "$GPU_COUNT" != "4" ]]; then
    printf 'The audited mesh currently requires exactly 4 GPUs.\n' >&2
    exit 1
fi

for file in \
    "$BASE_CFG" "$TEACHER" \
    "$PANDWM_ROOT/data/pickle/nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl" \
    "$PANDWM_ROOT/data/pickle/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl"; do
    [[ -f "$file" ]] || { printf 'Required file is missing: %s\n' "$file" >&2; exit 1; }
done
for directory in \
    "$PANDWM_ROOT/data/train/rgb" "$PANDWM_ROOT/data/val/rgb" \
    "$PANDWM_ROOT/data/bbox" "$PANDWM_ROOT/data/hdmap"; do
    [[ -d "$directory" ]] || { printf 'Required directory is missing: %s\n' "$directory" >&2; exit 1; }
done

pick_four_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || {
            printf 'CUDA_VISIBLE_DEVICES must contain exactly 4 GPUs: %s\n' \
                "$CUDA_VISIBLE_DEVICES" >&2
            return 1
        }
        return 0
    fi
    local selected
    selected="$(nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits | sort -t, -k2,2nr | \
        awk -F, -v limit="$MIN_FREE_MB" -v count="$GPU_COUNT" '
            {
                idx=$1; free=$2
                gsub(/ /,"",idx); gsub(/ /,"",free)
                if (free+0 >= limit+0 && n<count) { a[n++]=idx }
            }
            END {
                if (n==count) {
                    for (i=0;i<n;i++) printf "%s%s", a[i], (i+1<n ? "," : "")
                }
            }')"
    if [[ -z "$selected" ]]; then
        printf 'Refusing to start: fewer than %s GPUs have >= %s MiB free.\n' \
            "$GPU_COUNT" "$MIN_FREE_MB" >&2
        return 1
    fi
    export CUDA_VISIBLE_DEVICES="$selected"
}
pick_four_gpus

available_kb="$(df --output=avail -k "$PANDWM_ROOT" | tail -1 | tr -d ' ')"
required_gb=30
if (( MAX_STEPS >= CHECKPOINT_STEPS )); then
    required_gb=120
fi
if (( available_kb < required_gb * 1024 * 1024 )); then
    printf 'Refusing to start: checkpoint plan needs >= %s GiB free disk.\n' \
        "$required_gb" >&2
    exit 1
fi

if [[ -e "$RUN" ]]; then
    printf 'Refusing to overwrite existing run: %s\n' "$RUN" >&2
    exit 1
fi
mkdir -p "$RUN/log" "$PANDWM_ROOT/logs" "$PANDWM_ROOT/torchinductor_cache"
CFG="$RUN/log/taskmatched_config.json"
python3 "$SCRIPT_DIR/make_cd_taskmatched_config.py" \
    --base "$BASE_CFG" --output "$CFG" --teacher "$TEACHER" \
    --mesh-size "$GPU_COUNT" --eval-items "$EVAL_ITEMS"

eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$ENV_PREFIX"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$PANDWM_ROOT/torchinductor_cache"
# eval_4step_h20.sh sets this to reclaim allocator fragmentation on a shared
# GPU; this training launcher never did, and a 4-rank FSDP backward pass hit
# "Failed to CUDA calloc async 24 bytes" on 2026-08-29 despite 70GB+ free.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN/log/train_${TIMESTAMP}.log"
printf 'Task-matched causal consistency distillation\n' | tee "$LOG"
printf 'GPUs: %s\nConfig: %s\nRun: %s\n' \
    "$CUDA_VISIBLE_DEVICES" "$CFG" "$RUN" | tee -a "$LOG"

torchrun --standalone --nproc_per_node="$GPU_COUNT" \
    --master_port="$MASTER_PORT" \
    "$PANDWM_ROOT/code/PanDWM/src/dwm_pano/train.py" \
    -c "$CFG" -o "$RUN" \
    --max-steps "$MAX_STEPS" \
    --log-steps "$LOG_STEPS" \
    --preview-steps "$PREVIEW_STEPS" \
    --checkpointing-steps "$CHECKPOINT_STEPS" \
    --evaluation-steps "$CHECKPOINT_STEPS" \
    --eval-on-save 2>&1 | tee -a "$LOG"
