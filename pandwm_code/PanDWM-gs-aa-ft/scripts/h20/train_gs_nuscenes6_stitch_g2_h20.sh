#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="$PANDWM_ROOT/envs/pandwm"
CFG="${CFG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_nuscenes6_stitch_g2.json}"
RUN="${RUN:-$PANDWM_ROOT/runs/gs_nuscenes6_stitch_g2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29519}"
RESUME_FROM="${RESUME_FROM:-}"
RESUME_DATA_OFFSET="${RESUME_DATA_OFFSET:-}"
INITIAL_DATA_OFFSET="${INITIAL_DATA_OFFSET:-}"
MAX_STEPS="${MAX_STEPS:-2500}"
LOG_STEPS="${LOG_STEPS:-10}"
PREVIEW_STEPS="${PREVIEW_STEPS:-250}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-500}"
export PANDWM_MIN_FREE_MB="${PANDWM_MIN_FREE_MB:-80000}"

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
if [[ -n "$INITIAL_DATA_OFFSET" ]]; then
    require_positive_integer INITIAL_DATA_OFFSET "$INITIAL_DATA_OFFSET"
    if [[ -n "$RESUME_FROM" || -n "$RESUME_DATA_OFFSET" ]]; then
        printf 'INITIAL_DATA_OFFSET cannot be combined with resume options.\n' >&2
        exit 1
    fi
fi
if [[ ! -x "$CONDA_BIN" ]]; then
    printf 'Conda executable not found: %s\n' "$CONDA_BIN" >&2
    exit 1
fi

require_file "Conda environment history" "$ENV_PREFIX/conda-meta/history"
require_file "nuScenes6 G2 config" "$CFG"
require_file "G1 checkpoint" \
    "$PANDWM_ROOT/runs/gs_temporal_delta_xrender_v1/checkpoints/2500.pth"
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

python3 - "$CFG" "$MAX_STEPS" "${INITIAL_DATA_OFFSET:-0}" <<'PY'
import json
import os
import sys


with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = json.load(handle)
max_steps = int(sys.argv[2])
initial_data_offset = int(sys.argv[3])

pipeline = config["pipeline"]
training = pipeline["training_config"]
decoder = pipeline["decoder"]
initial_checkpoint = pipeline.get("initial_decoder_checkpoint_path")
adaptation = pipeline.get("latent_domain_adaptation")
perceptual_height, perceptual_width = training["perceptual_resolution"]
canvas_aspect = (
    2 * training["half_width"] - 494
) / training["half_height"]
required = {
    "real-six-camera OneBEV stitch supervision":
        training.get("supervision_geometry")
        == "nuscenes6_onebev_stitch",
    "legacy OneBEV target mapping":
        training.get("stitch_projection_model") == "onebev_legacy",
    "full camera origins from step zero":
        training.get("camera_origin_warmup_steps") == 0,
    "random independent-frame training":
        training.get("frame_sampling") == "random",
    "cross RGB disabled for geometry isolation":
        training.get("weight_cross_rgb_l1", 0.0) == 0.0,
    "cross alpha coverage disabled for geometry isolation":
        training.get("weight_cross_alpha_coverage", 0.0) == 0.0,
    "cross SSIM disabled for geometry isolation":
        training.get("cross_ssim_weight", 0.0) == 0.0,
    "preview temporal context disabled for geometry isolation":
        training.get("preview_use_temporal_context") is False,
    "evaluation temporal context disabled for geometry isolation":
        training.get("evaluation_use_temporal_context") is False,
    "demo temporal context disabled for geometry isolation":
        training.get("demo_use_temporal_context") is False,
    "stitched LPIPS preserves panorama aspect": abs(
        perceptual_width / perceptual_height / canvas_aspect - 1.0
    ) <= 0.05,
    "source-camera-z depth":
        training.get("range_is_source_camera_z") is True,
    "calibrated Gaussian origins":
        decoder.get("use_camera_origins") is True,
    "Plucker ray embedding": decoder.get("ray_embedding_mode") == "plucker",
    "metric range prior remains disabled":
        decoder.get("use_range_prior") is False,
    "homogeneous one-rig batches":
        config["training_dataloader"].get("batch_size") == 1,
    "G1 warm start exists":
        isinstance(initial_checkpoint, str)
        and os.path.isfile(initial_checkpoint),
}
failed = [name for name, valid in required.items() if not valid]
if adaptation is not None and adaptation.get("enabled", False):
    cache_dir = adaptation.get("cache_dir", "")
    manifest_path = os.path.join(cache_dir, "manifest.json")
    cache_ready = False
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        cache_ready = manifest.get("completed_steps", 0) >= max_steps \
            and manifest.get("start_offset", 0) == initial_data_offset
    if not cache_ready:
        failed.append(
            "generated-latent cache has at least MAX_STEPS complete items")
if failed:
    raise RuntimeError(
        "nuScenes6 G2 launcher config guard failed: " + "; ".join(failed))
print("nuScenes6 G2 config guard passed.")
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
LOG="$PANDWM_ROOT/logs/gs_nuscenes6_g2_${TIMESTAMP}.log"

printf 'PanDWM real-six-camera OneBEV-stitch GS G2 training\n' | tee "$LOG"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
printf 'Config: %s\n' "$CFG" | tee -a "$LOG"
printf 'Run: %s\n' "$RUN" | tee -a "$LOG"
printf 'Log/preview/checkpoint cadence: %s/%s/%s\n' \
    "$LOG_STEPS" "$PREVIEW_STEPS" "$CHECKPOINT_STEPS" | tee -a "$LOG"
printf 'Resume from/data offset: %s/%s\n' \
    "${RESUME_FROM:-none}" "${RESUME_DATA_OFFSET:-none}" | tee -a "$LOG"
printf 'Initial data offset: %s\n' \
    "${INITIAL_DATA_OFFSET:-0}" | tee -a "$LOG"
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
if [[ -n "$INITIAL_DATA_OFFSET" ]]; then
    train_args+=(--initial-data-offset "$INITIAL_DATA_OFFSET")
fi

python3 -m dwm_pano.train "${train_args[@]}" 2>&1 | tee -a "$LOG"
