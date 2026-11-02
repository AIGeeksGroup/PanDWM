#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="$PANDWM_ROOT/envs/pandwm"
CFG="${CFG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_smoke_h20.json}"
RUN="${RUN:-$PANDWM_ROOT/runs/gs_smoke}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29505}"
LOG_STEPS="${LOG_STEPS:-20}"
PREVIEW_STEPS="${PREVIEW_STEPS:-200}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
RESUME_FROM="${RESUME_FROM:-}"
export PANDWM_MIN_FREE_MB="${PANDWM_MIN_FREE_MB:-30000}"

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

require_positive_integer LOG_STEPS "$LOG_STEPS"
require_positive_integer PREVIEW_STEPS "$PREVIEW_STEPS"
require_positive_integer CHECKPOINTING_STEPS "$CHECKPOINTING_STEPS"
if [[ -n "$RESUME_FROM" ]]; then
    require_positive_integer RESUME_FROM "$RESUME_FROM"
fi

if [[ ! -x "$CONDA_BIN" ]]; then
    printf 'Conda executable not found: %s\n' "$CONDA_BIN" >&2
    exit 1
fi
require_file "Conda environment history" "$ENV_PREFIX/conda-meta/history"
require_file "GS smoke config" "$CFG"
require_file "Filtered train pkl" \
    "$PANDWM_ROOT/data/pickle/nuscenes_interp_12Hz_infos_train_smoke.pkl"
require_file "Validation pkl" \
    "$PANDWM_ROOT/data/pickle/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl"
require_file "nuScenes sample metadata" \
    "$PANDWM_ROOT/data/nuscenes_meta/v1.0-trainval/sample.json"
require_file "SD3.5 VAE config" \
    "$PANDWM_ROOT/ckpts/stable-diffusion-3.5-medium/vae/config.json"
require_directory "Train RGB directory" "$PANDWM_ROOT/data/train/rgb"
require_directory "Train segmentation directory" "$PANDWM_ROOT/data/train/seg"
require_directory "Validation RGB directory" "$PANDWM_ROOT/data/val/rgb"
require_directory "Validation segmentation directory" "$PANDWM_ROOT/data/val/seg"

pandwm_pick_cuda_device

mkdir -p "$RUN" "$PANDWM_ROOT/logs" \
    "$PANDWM_ROOT/torchinductor_cache"

eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$ENV_PREFIX"

# train.py selects DDP whenever LOCAL_RANK is present. A one-rank NCCL group
# exercises the same DistributedSampler and DDP decoder path as larger runs.
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
LOG="$PANDWM_ROOT/logs/gs_smoke_${TIMESTAMP}.log"

printf 'PanDWM h20 GS decoder smoke training\n' | tee "$LOG"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
printf 'Config: %s\n' "$CFG" | tee -a "$LOG"
printf 'Run: %s\n' "$RUN" | tee -a "$LOG"
printf 'Log cadence: %s\n' "$LOG_STEPS" | tee -a "$LOG"
printf 'Preview cadence: %s\n' "$PREVIEW_STEPS" | tee -a "$LOG"
printf 'Checkpoint cadence: %s\n' "$CHECKPOINTING_STEPS" | tee -a "$LOG"
printf 'Resume from: %s\n' "${RESUME_FROM:-none}" | tee -a "$LOG"
printf 'Log: %s\n' "$LOG" | tee -a "$LOG"

# The pipeline currently loads range h5 files before checking any depth loss
# weight. This guarded shim supplies zeros only for the exact depth-free G0
# configuration. Those zeros reach only zero-weight depth/range diagnostics;
# use_range_prior=false keeps them out of the decoder input.
python - \
    "$CFG" "$RUN" "$LOG_STEPS" "$PREVIEW_STEPS" \
    "$CHECKPOINTING_STEPS" "$RESUME_FROM" <<'PY' 2>&1 | tee -a "$LOG"
import json
import runpy
import sys

import torch

from dwm_pano.pipelines import ctsd_pano_gs_decoder as gs_decoder


config_path, run_path, log_steps, preview_steps, checkpoint_steps, resume = \
    sys.argv[1:]
with open(config_path, "r", encoding="utf-8") as handle:
    config = json.load(handle)

pipeline_config = config["pipeline"]
training_config = pipeline_config["training_config"]
decoder_config = pipeline_config["decoder"]
refiner_config = pipeline_config.get("latent_refinement")

required = {
    "range_dir is null": training_config.get("range_dir") is None,
    "weight_render_depth is zero":
        training_config.get("weight_render_depth") == 0.0,
    "weight_range_l1 is zero":
        training_config.get("weight_range_l1") == 0.0,
    "use_range_prior is false":
        decoder_config.get("use_range_prior") is False,
    "latent refinement is disabled":
        refiner_config is None or refiner_config.get("enabled") is False,
}
failed = [name for name, valid in required.items() if not valid]
if failed:
    raise RuntimeError(
        "Depth-free GS launcher config guard failed: " + "; ".join(failed)
    )


def load_zero_frame_ranges(self, batch, frame_indices):
    batch_size = batch["vae_images"].shape[0]
    frame_count = int(frame_indices.numel())
    return torch.zeros(
        (
            batch_size * frame_count,
            2,
            self.training_config["half_height"],
            self.training_config["half_width"],
        ),
        dtype=torch.float32,
        device=self.device,
    )


gs_decoder.PanoGaussianDecoderPipeline._load_frame_ranges = \
    load_zero_frame_ranges
print(
    "Depth-free G0 shim active: no range h5 file will be opened; "
    "depth and range diagnostics use zero targets and have zero loss weight.",
    flush=True,
)

sys.argv = [
    "train.py",
    "-c",
    config_path,
    "-o",
    run_path,
    "--log-steps",
    log_steps,
    "--preview-steps",
    preview_steps,
    "--checkpointing-steps",
    checkpoint_steps,
    "--evaluation-steps",
    "0",
    "--keep-last-ckpts",
    "0",
]
if resume:
    sys.argv.extend(["--resume-from", resume])

# train.py has no main() function, so run it as __main__ after installing the
# narrow pipeline shim above. External logging and save-time eval stay off.
runpy.run_module("dwm_pano.train", run_name="__main__")
PY
