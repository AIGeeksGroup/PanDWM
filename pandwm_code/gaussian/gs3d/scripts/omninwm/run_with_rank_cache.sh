#!/bin/bash
# Give every torchrun worker an independent Triton/DeepSpeed cache.  The
# official OmniNWM entry point is then executed unchanged via "$@".
set -euo pipefail

: "${LOCAL_RANK:?torchrun did not export LOCAL_RANK}"
: "${OMNINWM_TRITON_CACHE_ROOT:?missing OMNINWM_TRITON_CACHE_ROOT}"

export TRITON_CACHE_DIR=$OMNINWM_TRITON_CACHE_ROOT/rank_$LOCAL_RANK
mkdir -p "$TRITON_CACHE_DIR"
echo "rank=$LOCAL_RANK triton_cache=$TRITON_CACHE_DIR"
exec "$@"
