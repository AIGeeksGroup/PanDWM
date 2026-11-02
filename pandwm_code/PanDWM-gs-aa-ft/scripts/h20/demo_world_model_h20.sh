#!/usr/bin/env bash
# End-to-end world model demo on h20: Stage-2 generates the future, the
# Gaussian decoder lifts it to 3D, and the result is rendered from viewpoints
# the model never saw.
#
# Activating the conda env matters here beyond PYTHONPATH: gsplat JIT-compiles
# its CUDA extension on first use and needs `ninja`, which lives in the env's
# bin. Calling `envs/pandwm/bin/python` directly leaves ninja off PATH and the
# run dies at rasterization -- after having spent three minutes generating
# latents.
#
# Env overrides: GS_CFG GS_RUN GS_STEP STAGE2_CFG STAGE2_RUN STAGE2_STEP
#                STAGE2_EMA CLIP_INDEX FRAME_COUNT SHIFTS RENDER_TARGET
#                CALIB_PKL LATENT_SOURCE OUT SAVE_PANOS SAVE_GT_VIEWS
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
LATENT_SOURCE="${LATENT_SOURCE:-stage2}"
CLIP_INDEX="${CLIP_INDEX:-0}"
FRAME_COUNT="${FRAME_COUNT:-}"
SHIFTS="${SHIFTS:-0.0 1.0 2.0}"
RENDER_TARGET="${RENDER_TARGET:-virtual}"
CALIB_PKL="${CALIB_PKL:-$PANDWM_ROOT/data/pickle/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl}"
# The default output directory encodes everything that changes what the
# renders MEAN. Without the latent source and the step numbers, an A/B of
# stage2-vs-gt (or of two checkpoints) writes identical filenames into one
# directory: the second run overwrites the generated frames of the first and
# leaves a manifest claiming a single provenance for a mixed set.
OUT="${OUT:-$PANDWM_ROOT/demos/clip${CLIP_INDEX}_${RENDER_TARGET}_${LATENT_SOURCE}_gs${GS_STEP}_s2${STAGE2_STEP}ema${STAGE2_EMA}}"
SAVE_PANOS="${SAVE_PANOS:-1}"
SAVE_GT_VIEWS="${SAVE_GT_VIEWS:-0}"
MASTER_PORT="${MASTER_PORT:-29541}"
# Matches eval_gs_g1_h20.sh, which says of the identical co-resident Stage-2
# + GS stack: "never start below this floor". This script adds gsplat
# rasterization on top, so it cannot need less.
export PANDWM_MIN_FREE_MB="${PANDWM_MIN_FREE_MB:-70000}"

for flag_pair in \
    "STAGE2_EMA:$STAGE2_EMA" \
    "SAVE_PANOS:$SAVE_PANOS" \
    "SAVE_GT_VIEWS:$SAVE_GT_VIEWS"
do
    flag_name="${flag_pair%%:*}"
    flag_value="${flag_pair#*:}"
    if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
        printf '%s must be 0 or 1, got: %s\n' "$flag_name" "$flag_value" >&2
        exit 1
    fi
done

for label_path in \
    "GS config:$GS_CFG" "Stage-2 config:$STAGE2_CFG"
do
    label="${label_path%%:*}"
    path="${label_path#*:}"
    if [[ ! -f "$path" ]]; then
        printf '%s is missing: %s\n' "$label" "$path" >&2
        exit 1
    fi
done
if [[ ! -f "$GS_RUN/checkpoints/${GS_STEP}.pth" ]]; then
    printf 'GS checkpoint is missing: %s\n' \
        "$GS_RUN/checkpoints/${GS_STEP}.pth" >&2
    exit 1
fi
# Guarded here rather than left to Python: demo_world_model.py initializes
# NCCL and claims a GPU before it checks this path. Note the default target
# is a symlink that eval_4step_h20.sh creates on demand from
# ckpts/cd_10000_ema.pth, so on a fresh box it may simply not exist yet.
if [[ "$LATENT_SOURCE" == "stage2" ]]; then
    stage2_suffix=""
    [[ "$STAGE2_EMA" == "1" ]] && stage2_suffix="_ema"
    STAGE2_CKPT="$STAGE2_RUN/checkpoints/${STAGE2_STEP}${stage2_suffix}.pth"
    if [[ ! -f "$STAGE2_CKPT" ]]; then
        printf 'Stage-2 checkpoint is missing: %s\n' "$STAGE2_CKPT" >&2
        printf 'It is created by scripts/h20/eval_4step_h20.sh as a symlink to ckpts/.\n' >&2
        exit 1
    fi
fi
if [[ "$RENDER_TARGET" == "nuscenes6" && ! -f "$CALIB_PKL" ]]; then
    printf 'Calibration pkl is missing: %s\n' "$CALIB_PKL" >&2
    exit 1
fi

pandwm_pick_cuda_device
mkdir -p "$OUT" "$PANDWM_ROOT/logs"

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

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$PANDWM_ROOT/logs/demo_world_model_${TIMESTAMP}.log"

cmd=(
    python3 -m dwm_pano.demo_world_model
    --gs-config "$GS_CFG" --gs-run "$GS_RUN" --gs-step "$GS_STEP"
    --latent-source "$LATENT_SOURCE"
    --clip-index "$CLIP_INDEX"
    --render-target "$RENDER_TARGET"
    --out-dir "$OUT"
)
if [[ "$LATENT_SOURCE" == "stage2" ]]; then
    cmd+=(
        --stage2-config "$STAGE2_CFG" --stage2-run "$STAGE2_RUN"
        --stage2-step "$STAGE2_STEP"
    )
    [[ "$STAGE2_EMA" == "1" ]] && cmd+=(--stage2-ema)
fi
[[ -n "$FRAME_COUNT" ]] && cmd+=(--frame-count "$FRAME_COUNT")
[[ "$SAVE_PANOS" == "1" ]] && cmd+=(--save-panos)
[[ "$SAVE_GT_VIEWS" == "1" ]] && cmd+=(--save-gt-views)
if [[ "$RENDER_TARGET" == "virtual" ]]; then
    # `set -f` for the split only: SHIFTS is meant to be whitespace-separated
    # numbers, but unquoted expansion also globs, so a stray `*` would turn
    # into a list of filenames from the caller's working directory.
    set -f
    # shellcheck disable=SC2206
    shift_array=($SHIFTS)
    set +f
    if [[ "${#shift_array[@]}" -eq 0 ]]; then
        printf 'SHIFTS must list at least one offset in meters\n' >&2
        exit 1
    fi
    cmd+=(--lateral-shifts-m "${shift_array[@]}")
else
    cmd+=(--calibration-pkl "$CALIB_PKL")
fi

printf 'PanDWM h20 end-to-end world model demo\n' | tee "$LOG"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
printf 'GS: %s @ %s\n' "$GS_RUN" "$GS_STEP" | tee -a "$LOG"
printf 'Latent source: %s\n' "$LATENT_SOURCE" | tee -a "$LOG"
printf 'Render target: %s\n' "$RENDER_TARGET" | tee -a "$LOG"
printf 'Out: %s\n' "$OUT" | tee -a "$LOG"

"${cmd[@]}" 2>&1 | tee -a "$LOG"
