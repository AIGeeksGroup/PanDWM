#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="$PANDWM_ROOT/envs/pandwm"
CFG="${CFG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_p2_h20.json}"
RUN="${RUN:-$PANDWM_ROOT/runs/gs_p2}"
STEP="${STEP:-15000}"
HOLDOUT_PKL="${HOLDOUT_PKL:-$PANDWM_ROOT/data/pickle/nuscenes_interp_12Hz_infos_holdout_p2.pkl}"
VAL_LIMIT="${VAL_LIMIT:-}"
STAGE2_CONFIG="${STAGE2_CONFIG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_h20.json}"
STAGE2_RUN="${STAGE2_RUN:-$PANDWM_ROOT/runs/cd_baseline}"
STAGE2_STEP="${STAGE2_STEP:-10000}"
EMA="${EMA:-1}"
ARM="${ARM:-gt}"
SMOKE="${SMOKE:-0}"
OUT="${OUT:-$RUN/eval_gs_g1/$STEP/$ARM}"

if [[ -z "$VAL_LIMIT" ]]; then
    if [[ "$SMOKE" == "1" ]]; then
        VAL_LIMIT=1
    else
        VAL_LIMIT=32
    fi
fi

if [[ "$EMA" != "0" && "$EMA" != "1" ]]; then
    printf 'EMA must be 0 or 1, got: %s\n' "$EMA" >&2
    exit 1
fi
if [[ "$SMOKE" != "0" && "$SMOKE" != "1" ]]; then
    printf 'SMOKE must be 0 or 1, got: %s\n' "$SMOKE" >&2
    exit 1
fi
case "$ARM" in
    gt|stage2|stage2_reencode|gt_reencode) ;;
    *)
        printf 'ARM must be gt, stage2, stage2_reencode, or gt_reencode; got: %s\n' \
            "$ARM" >&2
        exit 1
        ;;
esac
if [[ "$ARM" == "stage2" || "$ARM" == "stage2_reencode" ]]; then
    [[ -f "$STAGE2_CONFIG" ]] || {
        printf 'Stage-2 config is missing: %s\n' "$STAGE2_CONFIG" >&2
        exit 1
    }
    stage2_suffix=""
    if [[ "$EMA" == "1" ]]; then
        stage2_suffix="_ema"
    fi
    stage2_checkpoint="$STAGE2_RUN/checkpoints/${STAGE2_STEP}${stage2_suffix}.pth"
    [[ -f "$stage2_checkpoint" ]] || {
        printf 'Stage-2 checkpoint is missing: %s\n' "$stage2_checkpoint" >&2
        exit 1
    }
fi

if [[ ! -x "$CONDA_BIN" ]]; then
    printf 'Conda executable not found: %s\n' "$CONDA_BIN" >&2
    exit 1
fi
if [[ ! -f "$ENV_PREFIX/conda-meta/history" ]]; then
    printf 'Conda environment is missing: %s\n' "$ENV_PREFIX" >&2
    exit 1
fi
if [[ ! -f "$CFG" ]]; then
    printf 'GS evaluation config is missing: %s\n' "$CFG" >&2
    exit 1
fi
if [[ ! -f "$RUN/checkpoints/$STEP.pth" ]]; then
    printf 'GS decoder checkpoint is missing: %s\n' "$RUN/checkpoints/$STEP.pth" >&2
    exit 1
fi
if [[ ! -f "$HOLDOUT_PKL" ]]; then
    printf 'Holdout pkl is missing: %s\n' "$HOLDOUT_PKL" >&2
    exit 1
fi

# G1 co-residency needs the freer H20 cards. Never start below this floor.
export PANDWM_MIN_FREE_MB="${PANDWM_MIN_FREE_MB:-70000}"
pandwm_pick_cuda_device

mkdir -p "$PANDWM_ROOT/logs" "$PANDWM_ROOT/torchinductor_cache"
eval "$($CONDA_BIN shell.bash hook)"
conda activate "$ENV_PREFIX"

export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29506}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$PANDWM_ROOT/torchinductor_cache"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$PANDWM_ROOT/logs/eval_gs_g1_${ARM}_step${STEP}_${TIMESTAMP}.log"
printf 'PanDWM h20 G1 GS domain-gap evaluation\n' | tee "$LOG"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
printf 'Minimum free GPU memory: %s MiB\n' "$PANDWM_MIN_FREE_MB" | tee -a "$LOG"
printf 'Arm: %s\n' "$ARM" | tee -a "$LOG"
printf 'GS config: %s\n' "$CFG" | tee -a "$LOG"
printf 'GS run/step: %s / %s\n' "$RUN" "$STEP" | tee -a "$LOG"
printf 'Stage-2 config/run/step: %s / %s / %s\n' \
    "$STAGE2_CONFIG" "$STAGE2_RUN" "$STAGE2_STEP" | tee -a "$LOG"
printf 'EMA: %s\n' "$EMA" | tee -a "$LOG"
printf 'Log: %s\n' "$LOG" | tee -a "$LOG"

cmd=(
    python "$PANDWM_ROOT/code/PanDWM/src/dwm_pano/evaluate_gs.py"
    -c "$CFG"
    -r "$RUN"
    -s "$STEP"
    --holdout-pkl "$HOLDOUT_PKL"
    --val-limit "$VAL_LIMIT"
    --latent-source "$ARM"
    --stage2-config "$STAGE2_CONFIG"
    --stage2-run "$STAGE2_RUN"
    --stage2-step "$STAGE2_STEP"
    --out "$OUT"
)
if [[ "$EMA" == "1" ]]; then
    cmd+=(--stage2-ema)
fi
if [[ "$SMOKE" == "1" ]]; then
    cmd+=(--smoke)
fi
"${cmd[@]}" 2>&1 | tee -a "$LOG"
