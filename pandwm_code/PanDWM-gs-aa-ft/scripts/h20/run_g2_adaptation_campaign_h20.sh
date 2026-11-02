#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

CAMPAIGN_DIR="${CAMPAIGN_DIR:-$PANDWM_ROOT/runs/g2_adaptation_campaign_20260825_offset2500}"
BASE_METRICS="$PANDWM_ROOT/runs/gs_nuscenes6_stitch_g2/eval_decode_compare/2500_holdout5/metrics.json"
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
    local name run
    for name in control appearance appearance_stage2mix; do
        run="$PANDWM_ROOT/runs/gs_g2_ft_${name}500_offset2500"
        args+=(--arm "$name=$run/eval_decode_compare/500_holdout5/metrics.json")
    done
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda activate "$PANDWM_ROOT/envs/pandwm"
    "${args[@]}" 2>&1 | tee -a "$STATUS_LOG"
}

train_arm() {
    local name="$1"
    local config="$2"
    local run="$3"
    if [[ ! -f "$run/checkpoints/500.pth" ]]; then
        wait_for_gpu "train-$name" 60000
        CFG="$config" RUN="$run" MAX_STEPS=500 CHECKPOINT_STEPS=500 \
            PREVIEW_STEPS=500 LOG_STEPS=10 INITIAL_DATA_OFFSET=2500 \
            bash "$SCRIPT_DIR/train_gs_nuscenes6_stitch_g2_h20.sh"
        unset CUDA_VISIBLE_DEVICES
    fi
}

eval_arm() {
    local name="$1"
    local config="$2"
    local run="$3"
    local out="$run/eval_decode_compare/500_holdout5"
    if [[ ! -f "$out/metrics.json" ]]; then
        wait_for_gpu "eval-$name" 70000
        CFG="$config" RUN="$run" STEP=500 CLIP_LIMIT=0 OUT="$out" \
            bash "$SCRIPT_DIR/eval_gs_decode_compare_h20.sh"
        unset CUDA_VISIBLE_DEVICES
    fi
    summarize
}

printf '%s campaign start\n' "$(date --iso-8601=seconds)" | tee -a "$STATUS_LOG"
[[ -f "$BASE_METRICS" ]] || { printf 'Missing baseline: %s\n' "$BASE_METRICS"; exit 1; }

CONTROL_CFG="$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_g2_ft_control_500.json"
APPEARANCE_CFG="$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_g2_ft_appearance_500.json"
MIX_CFG="$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_g2_ft_appearance_stage2mix_500.json"

train_arm control "$CONTROL_CFG" "$PANDWM_ROOT/runs/gs_g2_ft_control500_offset2500"
eval_arm control "$CONTROL_CFG" "$PANDWM_ROOT/runs/gs_g2_ft_control500_offset2500"

train_arm appearance "$APPEARANCE_CFG" "$PANDWM_ROOT/runs/gs_g2_ft_appearance500_offset2500"
eval_arm appearance "$APPEARANCE_CFG" "$PANDWM_ROOT/runs/gs_g2_ft_appearance500_offset2500"

if [[ ! -f "$PANDWM_ROOT/cache/stage2_train_g2_seed0_offset2500_500/manifest.json" ]] \
        || ! grep -q '"completed_steps": 500' \
            "$PANDWM_ROOT/cache/stage2_train_g2_seed0_offset2500_500/manifest.json"; then
    wait_for_gpu "cache-stage2-offset2500-500" 70000
    STEPS=500 START_OFFSET=2500 bash "$SCRIPT_DIR/cache_stage2_train_latents_h20.sh"
    unset CUDA_VISIBLE_DEVICES
fi

train_arm appearance_stage2mix "$MIX_CFG" \
    "$PANDWM_ROOT/runs/gs_g2_ft_appearance_stage2mix500_offset2500"
eval_arm appearance_stage2mix "$MIX_CFG" \
    "$PANDWM_ROOT/runs/gs_g2_ft_appearance_stage2mix500_offset2500"

printf '%s campaign complete\n' "$(date --iso-8601=seconds)" | tee -a "$STATUS_LOG"
