#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

CAMPAIGN_DIR="${CAMPAIGN_DIR:-$PANDWM_ROOT/runs/g2_mix_followup_campaign_20260825}"
BASE_METRICS="$PANDWM_ROOT/runs/gs_nuscenes6_stitch_g2/eval_decode_compare/2500_holdout5/metrics.json"
MIX500_RUN="$PANDWM_ROOT/runs/gs_g2_ft_appearance_stage2mix500_offset2500"
mkdir -p "$CAMPAIGN_DIR" "$PANDWM_ROOT/logs"
STATUS_LOG="$CAMPAIGN_DIR/campaign.log"

wait_for_gpu() {
    local label="$1"
    local minimum="$2"
    while true; do
        unset CUDA_VISIBLE_DEVICES || true
        export PANDWM_MIN_FREE_MB="$minimum"
        if pandwm_pick_cuda_device 2>>"$STATUS_LOG"; then
            printf '%s selected GPU %s for %s\n' \
                "$(date --iso-8601=seconds)" "$CUDA_VISIBLE_DEVICES" "$label" \
                | tee -a "$STATUS_LOG"
            return 0
        fi
        printf '%s waiting for GPU >= %s MiB: %s\n' \
            "$(date --iso-8601=seconds)" "$minimum" "$label" \
            | tee -a "$STATUS_LOG"
        sleep 60
    done
}

summarize() {
    local args=(python3 -m dwm_pano.summarize_g2_adaptation_results
        --baseline "$BASE_METRICS"
        --out "$CAMPAIGN_DIR/summary")
    local spec metrics
    for spec in \
        "mix500=$MIX500_RUN/eval_decode_compare/500_holdout5/metrics.json" \
        "continue50=$PANDWM_ROOT/runs/gs_g2_ft_stage2mix50_control_250_offset2500/eval_decode_compare/250_holdout5/metrics.json" \
        "mix75=$PANDWM_ROOT/runs/gs_g2_ft_stage2mix75_250_offset2500/eval_decode_compare/250_holdout5/metrics.json" \
        "lpips20=$PANDWM_ROOT/runs/gs_g2_ft_stage2mix50_lpips0p2_250_offset2500/eval_decode_compare/250_holdout5/metrics.json"; do
        metrics="${spec#*=}"
        if [[ -f "$metrics" ]]; then
            args+=(--arm "$spec")
        fi
    done
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda activate "$PANDWM_ROOT/envs/pandwm"
    "${args[@]}" 2>&1 | tee -a "$STATUS_LOG"
}

train_arm() {
    local name="$1"
    local config="$2"
    local run="$3"
    if [[ ! -f "$run/checkpoints/250.pth" ]]; then
        wait_for_gpu "train-$name" 60000
        CFG="$config" RUN="$run" MAX_STEPS=250 CHECKPOINT_STEPS=250 \
            PREVIEW_STEPS=250 LOG_STEPS=10 INITIAL_DATA_OFFSET=2500 \
            bash "$SCRIPT_DIR/train_gs_nuscenes6_stitch_g2_h20.sh"
        unset CUDA_VISIBLE_DEVICES
    fi
}

eval_arm() {
    local name="$1"
    local config="$2"
    local run="$3"
    local out="$run/eval_decode_compare/250_holdout5"
    if [[ ! -f "$out/metrics.json" ]]; then
        wait_for_gpu "eval-$name" 70000
        CFG="$config" RUN="$run" STEP=250 CLIP_LIMIT=0 OUT="$out" \
            bash "$SCRIPT_DIR/eval_gs_decode_compare_h20.sh"
        unset CUDA_VISIBLE_DEVICES
    fi
    summarize
}

printf '%s follow-up campaign start\n' "$(date --iso-8601=seconds)" \
    | tee -a "$STATUS_LOG"
[[ -f "$BASE_METRICS" ]] || {
    printf 'Missing baseline: %s\n' "$BASE_METRICS"
    exit 1
}
[[ -f "$MIX500_RUN/checkpoints/500.pth" ]] || {
    printf 'Missing best mix checkpoint: %s\n' \
        "$MIX500_RUN/checkpoints/500.pth"
    exit 1
}

CONTROL_CFG="$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_g2_ft_stage2mix50_control_250.json"
MIX75_CFG="$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_g2_ft_stage2mix75_250.json"
LPIPS20_CFG="$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_g2_ft_stage2mix50_lpips0p2_250.json"

train_arm continue50 "$CONTROL_CFG" \
    "$PANDWM_ROOT/runs/gs_g2_ft_stage2mix50_control_250_offset2500"
eval_arm continue50 "$CONTROL_CFG" \
    "$PANDWM_ROOT/runs/gs_g2_ft_stage2mix50_control_250_offset2500"

train_arm mix75 "$MIX75_CFG" \
    "$PANDWM_ROOT/runs/gs_g2_ft_stage2mix75_250_offset2500"
eval_arm mix75 "$MIX75_CFG" \
    "$PANDWM_ROOT/runs/gs_g2_ft_stage2mix75_250_offset2500"

train_arm lpips20 "$LPIPS20_CFG" \
    "$PANDWM_ROOT/runs/gs_g2_ft_stage2mix50_lpips0p2_250_offset2500"
eval_arm lpips20 "$LPIPS20_CFG" \
    "$PANDWM_ROOT/runs/gs_g2_ft_stage2mix50_lpips0p2_250_offset2500"

printf '%s follow-up campaign complete\n' "$(date --iso-8601=seconds)" \
    | tee -a "$STATUS_LOG"
