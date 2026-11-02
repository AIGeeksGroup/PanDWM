#!/usr/bin/env bash

# Shared paths and network settings for PanDWM on h20.
export PANDWM_ROOT="/path/to/pandwm_root"
export PYTHONPATH="$PANDWM_ROOT/code/PanDWM/src:$PANDWM_ROOT/code/TATS_fvd${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT="https://hf-mirror.com"

pandwm_pick_cuda_device() {
    if [[ -v CUDA_VISIBLE_DEVICES ]]; then
        # Existence alone is not enough: an empty value passes the guard and
        # then torch dies at set_device(0) after the environment is up, and
        # a value inherited from an earlier command in the same shell pins
        # this run to whatever card that was -- silently defeating the
        # free-memory margin this function exists to enforce.
        if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
            printf 'CUDA_VISIBLE_DEVICES is set but not a device list: %s\n' \
                "$CUDA_VISIBLE_DEVICES" >&2
            return 1
        fi
        printf 'Using existing CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" >&2
        return 0
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf 'nvidia-smi is unavailable; set CUDA_VISIBLE_DEVICES manually.\n' >&2
        return 1
    fi

    # Shared machine: pick the single GPU with the most free memory, and
    # refuse to start below a safety margin so other users' jobs never OOM
    # because of us. Override the margin with PANDWM_MIN_FREE_MB.
    local min_free_mb="${PANDWM_MIN_FREE_MB:-50000}"
    local table selection
    table="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)"
    printf 'GPU free memory (MiB):\n%s\n' "$table" >&2
    selection="$(printf '%s\n' "$table" | awk -F ',' -v min="$min_free_mb" '
        {
            idx = $1
            free = $2
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", idx)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", free)
            if (!found || free + 0 > best_free + 0) {
                found = 1
                best_free = free
                best_idx = idx
            }
        }
        END {
            if (found && best_free + 0 >= min + 0) {
                print best_idx
            } else if (found) {
                print "LOW:" best_idx ":" best_free
            }
        }
    ')"

    if [[ -z "$selection" ]]; then
        printf 'No NVIDIA GPU was reported; set CUDA_VISIBLE_DEVICES manually.\n' >&2
        return 1
    fi
    if [[ "$selection" == LOW:* ]]; then
        printf 'Refusing to start: best GPU %s has only %s MiB free (< %s MiB safety margin). Lower PANDWM_MIN_FREE_MB to override.\n' \
            "$(cut -d: -f2 <<<"$selection")" "$(cut -d: -f3 <<<"$selection")" \
            "$min_free_mb" >&2
        return 1
    fi

    export CUDA_VISIBLE_DEVICES="$selection"
    printf 'Selected CUDA_VISIBLE_DEVICES=%s (most free memory, >= %s MiB margin).\n' \
        "$CUDA_VISIBLE_DEVICES" "$min_free_mb" >&2
}
