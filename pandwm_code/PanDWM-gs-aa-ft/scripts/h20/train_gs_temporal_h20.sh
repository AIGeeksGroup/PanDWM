#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="$PANDWM_ROOT/envs/pandwm"
CFG="${CFG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_temporal_delta_xrender.json}"
RUN="${RUN:-$PANDWM_ROOT/runs/gs_temporal_delta_xrender_v1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29517}"
RESUME_FROM="${RESUME_FROM:-}"
RESUME_DATA_OFFSET="${RESUME_DATA_OFFSET:-}"
MAX_STEPS="${MAX_STEPS:-2500}"
LOG_STEPS="${LOG_STEPS:-10}"
PREVIEW_STEPS="${PREVIEW_STEPS:-250}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-500}"
export PANDWM_MIN_FREE_MB="${PANDWM_MIN_FREE_MB:-50000}"

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer, got: %s\n' \
            "$name" "$value" >&2
        exit 1
    fi
}

require_file() {
    local label="$1"
    local path="$2"
    if [[ ! -f "$path" ]]; then
        printf '%s is missing: %s\n' "$label" "$path" >&2
        exit 1
    fi
}

require_directory() {
    local label="$1"
    local path="$2"
    if [[ ! -d "$path" ]]; then
        printf '%s is missing: %s\n' "$label" "$path" >&2
        exit 1
    fi
}

for pair in \
    "MAX_STEPS:$MAX_STEPS" \
    "LOG_STEPS:$LOG_STEPS" \
    "PREVIEW_STEPS:$PREVIEW_STEPS" \
    "CHECKPOINT_STEPS:$CHECKPOINT_STEPS"; do
    require_positive_integer "${pair%%:*}" "${pair#*:}"
done
if [[ -n "$RESUME_FROM" ]]; then
    require_positive_integer RESUME_FROM "$RESUME_FROM"
    if (( MAX_STEPS <= RESUME_FROM )); then
        printf 'MAX_STEPS must be greater than RESUME_FROM.\n' >&2
        exit 1
    fi
fi
if [[ -n "$RESUME_DATA_OFFSET" ]]; then
    require_positive_integer RESUME_DATA_OFFSET "$RESUME_DATA_OFFSET"
    if [[ -z "$RESUME_FROM" ]]; then
        printf 'RESUME_DATA_OFFSET requires RESUME_FROM.\n' >&2
        exit 1
    fi
fi
if [[ ! -x "$CONDA_BIN" ]]; then
    printf 'Conda executable not found: %s\n' "$CONDA_BIN" >&2
    exit 1
fi

require_file "Conda environment history" "$ENV_PREFIX/conda-meta/history"
require_file "Temporal GS config" "$CFG"
require_file "Production GS checkpoint" \
    "$PANDWM_ROOT/runs/gs_prod_rd0p01_rl0p01/checkpoints/15000.pth"
require_file "P2 train pkl" \
    "$PANDWM_ROOT/data/pickle/nuscenes_interp_12Hz_infos_train_p2.pkl"
require_file "Validation pkl" \
    "$PANDWM_ROOT/data/pickle/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl"
require_file "nuScenes sample metadata" \
    "$PANDWM_ROOT/data/nuscenes_meta/v1.0-trainval/sample.json"
require_file "SD3.5 VAE config" \
    "$PANDWM_ROOT/ckpts/stable-diffusion-3.5-medium/vae/config.json"
require_directory "Train RGB directory" "$PANDWM_ROOT/data/train/rgb"
require_directory "Train segmentation directory" "$PANDWM_ROOT/data/train/seg"
require_directory "Train depth directory" "$PANDWM_ROOT/data/train/depth"
require_directory "Validation RGB directory" "$PANDWM_ROOT/data/val/rgb"
require_directory "Validation segmentation directory" "$PANDWM_ROOT/data/val/seg"

pandwm_pick_cuda_device

mkdir -p "$RUN" "$PANDWM_ROOT/logs" \
    "$PANDWM_ROOT/torchinductor_cache"

eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$ENV_PREFIX"

python3 - "$CFG" <<'PY'
import json
import os
import sys


with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = json.load(handle)

pipeline = config["pipeline"]
training = pipeline["training_config"]
decoder = pipeline["decoder"]
initial_checkpoint = pipeline.get("initial_decoder_checkpoint_path")
required = {
    "adjacent pair sampling":
        training.get("frame_sampling") == "adjacent_pairs",
    "even frames_per_step": training.get("frames_per_step", 0) % 2 == 0,
    "positive cross-frame RGB loss":
        training.get("weight_cross_rgb_l1", 0.0) > 0.0,
    "causal temporal delta adapter":
        decoder.get("temporal_delta_enabled") is True,
    "metric range prior remains disabled":
        decoder.get("use_range_prior") is False,
    "production warm start exists":
        isinstance(initial_checkpoint, str)
        and os.path.isfile(initial_checkpoint),
}
failed = [name for name, valid in required.items() if not valid]
if failed:
    raise RuntimeError(
        "Temporal GS launcher config guard failed: " + "; ".join(failed))
print("Temporal GS config guard passed.")
PY

export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR
export MASTER_PORT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$PANDWM_ROOT/torchinductor_cache"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$PANDWM_ROOT/logs/gs_temporal_${TIMESTAMP}.log"

printf 'PanDWM causal temporal-delta + cross-render GS training\n' | tee "$LOG"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
printf 'Config: %s\n' "$CFG" | tee -a "$LOG"
printf 'Run: %s\n' "$RUN" | tee -a "$LOG"
printf 'Log/preview/checkpoint cadence: %s/%s/%s\n' \
    "$LOG_STEPS" "$PREVIEW_STEPS" "$CHECKPOINT_STEPS" | tee -a "$LOG"
printf 'Resume from/data offset: %s/%s\n' \
    "${RESUME_FROM:-none}" "${RESUME_DATA_OFFSET:-none}" | tee -a "$LOG"
printf 'Max steps: %s\n' "$MAX_STEPS" | tee -a "$LOG"
printf 'Log: %s\n' "$LOG" | tee -a "$LOG"

train_args=(
    -c "$CFG"
    -o "$RUN"
    --log-steps "$LOG_STEPS"
    --preview-steps "$PREVIEW_STEPS"
    --checkpointing-steps "$CHECKPOINT_STEPS"
    --evaluation-steps 0
    --keep-last-ckpts 0
    --max-steps "$MAX_STEPS"
)
if [[ -n "$RESUME_FROM" ]]; then
    train_args+=(--resume-from "$RESUME_FROM")
fi
if [[ -n "$RESUME_DATA_OFFSET" ]]; then
    train_args+=(--resume-data-offset "$RESUME_DATA_OFFSET")
fi

python3 -m dwm_pano.train "${train_args[@]}" 2>&1 | tee -a "$LOG"
