#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="$PANDWM_ROOT/envs/pandwm"
CFG="${CFG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_h20.json}"
RUN="${RUN:-$PANDWM_ROOT/runs/cd_baseline}"
STEP="${STEP:-10000}"
EMA="${EMA:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29504}"
GENERATOR_SEED="${GENERATOR_SEED:-}"
EVAL_ITEMS="${EVAL_ITEMS:-}"

CKPT_SUFFIX=""
EMA_FLAG=""
if [[ "$EMA" == "1" ]]; then
    CKPT_SUFFIX="_ema"
    EMA_FLAG="--ema"
fi
SOURCE_CKPT="${SOURCE_CKPT:-$PANDWM_ROOT/ckpts/cd_10000${CKPT_SUFFIX}.pth}"

if [[ "$EMA" != "0" && "$EMA" != "1" ]]; then
    printf 'EMA must be 0 or 1, got: %s\n' "$EMA" >&2
    exit 1
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
    printf 'Evaluation config is missing: %s\n' "$CFG" >&2
    exit 1
fi

pandwm_pick_cuda_device

mkdir -p "$RUN/checkpoints" "$PANDWM_ROOT/logs" \
    "$PANDWM_ROOT/torchinductor_cache"

if [[ -n "$EVAL_ITEMS" || -n "$GENERATOR_SEED" ]]; then
    [[ -z "$EVAL_ITEMS" || "$EVAL_ITEMS" =~ ^[1-9][0-9]*$ ]] || {
        printf 'EVAL_ITEMS must be empty or a positive integer: %s\n' \
            "$EVAL_ITEMS" >&2
        exit 1
    }
    [[ -z "$GENERATOR_SEED" || "$GENERATOR_SEED" =~ ^[0-9]+$ ]] || {
        printf 'GENERATOR_SEED must be empty or a non-negative integer: %s\n' \
            "$GENERATOR_SEED" >&2
        exit 1
    }
    mkdir -p "$RUN/eval_configs"
    CFG_SNAPSHOT="$RUN/eval_configs/cd_eval_seed${GENERATOR_SEED:-config}_items${EVAL_ITEMS:-config}.json"
    python3 - "$CFG" "$CFG_SNAPSHOT" "$GENERATOR_SEED" "$EVAL_ITEMS" <<'PY'
import json
import sys

source, output, seed, items = sys.argv[1:]
with open(source, "r", encoding="utf-8") as f:
    config = json.load(f)
if seed:
    config["generator_seed"] = int(seed)
if items:
    config["pipeline"]["inference_config"]["evaluation_item_count"] = int(items)
with open(output, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(output)
PY
    CFG="$CFG_SNAPSHOT"
fi

REQUESTED_CKPT="$RUN/checkpoints/${STEP}${CKPT_SUFFIX}.pth"
if [[ ! -f "$SOURCE_CKPT" ]]; then
    printf 'Stage-2 source checkpoint is missing: %s\n' "$SOURCE_CKPT" >&2
    exit 1
fi
if [[ ! -e "$REQUESTED_CKPT" && ! -L "$REQUESTED_CKPT" ]]; then
    LINK_TARGET="$(realpath --relative-to="$RUN/checkpoints" "$SOURCE_CKPT")"
    ln -s -- "$LINK_TARGET" "$REQUESTED_CKPT"
fi
if [[ ! -L "$REQUESTED_CKPT" || ! -f "$REQUESTED_CKPT" ]]; then
    printf 'Checkpoint path is not a valid symlink: %s\n' \
        "$REQUESTED_CKPT" >&2
    exit 1
fi
if [[ "$(realpath "$REQUESTED_CKPT")" != "$(realpath "$SOURCE_CKPT")" ]]; then
    printf 'Checkpoint symlink has an unexpected target: %s\n' \
        "$REQUESTED_CKPT" >&2
    exit 1
fi
if [[ ! -f "$REQUESTED_CKPT" ]]; then
    printf 'Requested checkpoint is missing: %s\n' "$REQUESTED_CKPT" >&2
    exit 1
fi

eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$ENV_PREFIX"

# evaluate_4step.py has no non-DDP branch. A one-process NCCL group preserves
# its DistributedSampler and torchmetrics synchronization behavior.
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR
export MASTER_PORT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$PANDWM_ROOT/torchinductor_cache"
# Single-GPU eval carries main + EMA + encoders in fp32; expandable segments
# reclaims the ~12G of fragmentation the default allocator strands.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$PANDWM_ROOT/logs/eval_4step_step${STEP}${CKPT_SUFFIX}_${TIMESTAMP}.log"

printf 'PanDWM h20 4-step evaluation\n' | tee "$LOG"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
printf 'Config: %s\n' "$CFG" | tee -a "$LOG"
printf 'Run: %s\n' "$RUN" | tee -a "$LOG"
printf 'Checkpoint: %s\n' "$REQUESTED_CKPT" | tee -a "$LOG"
printf 'Seed/items override: %s / %s\n' \
    "${GENERATOR_SEED:-config}" "${EVAL_ITEMS:-config}" | tee -a "$LOG"
printf 'Log: %s\n' "$LOG" | tee -a "$LOG"

# The entry point unconditionally rewrites its FSDP mesh to [1, 4], which is
# invalid for WORLD_SIZE=1. The runtime shim narrows only that mesh to [1, 1].
# It also stubs the unavailable online stitcher; pre-rendered layout PNGs must
# exist, so calling the stub is a hard failure rather than a silent fallback.
python - "$CFG" "$RUN" "$STEP" "$EMA_FLAG" <<'PY' 2>&1 | tee -a "$LOG"
import os
import sys
import types

import torch.distributed.device_mesh as device_mesh


original_init_device_mesh = device_mesh.init_device_mesh


def init_single_rank_device_mesh(device_type, mesh_shape, *args, **kwargs):
    if os.environ.get("WORLD_SIZE") == "1" and tuple(mesh_shape) == (1, 4):
        mesh_shape = (1, 1)
    return original_init_device_mesh(device_type, mesh_shape, *args, **kwargs)


device_mesh.init_device_mesh = init_single_rank_device_mesh

omninwm = types.ModuleType("omninwm")
omninwm.__path__ = []
omninwm_utils = types.ModuleType("omninwm.utils")
omninwm_utils.__path__ = []
pano_hardseam = types.ModuleType("omninwm.utils.pano_hardseam")


def unavailable_stitcher(*args, **kwargs):
    raise RuntimeError(
        "Online pano stitching is unavailable on h20; verify every bbox and "
        "hdmap timestamp has a pre-rendered PNG"
    )


pano_hardseam.stitch_hardseam = unavailable_stitcher
omninwm.utils = omninwm_utils
omninwm_utils.pano_hardseam = pano_hardseam
sys.modules["omninwm"] = omninwm
sys.modules["omninwm.utils"] = omninwm_utils
sys.modules["omninwm.utils.pano_hardseam"] = pano_hardseam

from dwm_pano import evaluate_4step
from dwm_pano.pipelines import ctsd_pano_consistency_distillation as _cd_mod

# The consistency teacher is only exercised by the training loss; the 4-step
# eval path never calls it. Building it anyway costs a third full-precision
# DiT copy (~13.4G) that does not fit on one GPU next to main + EMA. Alias
# the teacher to the already-built main model instead of constructing it.
_orig_build_aux = \
    _cd_mod.PanoCrossviewTemporalConsistencyDistillation._build_auxiliary_model


def _build_aux_eval_only(self, checkpoint_path, model_load_state_args,
                         requires_grad, role):
    if role == "teacher" and hasattr(self, "model_wrapper"):
        print("eval shim: aliasing consistency teacher to the main model")
        return self.model, self.model_wrapper
    return _orig_build_aux(
        self, checkpoint_path, model_load_state_args, requires_grad, role)


_cd_mod.PanoCrossviewTemporalConsistencyDistillation._build_auxiliary_model = \
    _build_aux_eval_only

config_path, run_path, step, ema_flag = sys.argv[1:]
sys.argv = [
    "evaluate_4step.py",
    "-c",
    config_path,
    "-r",
    run_path,
    "-s",
    step,
]
if ema_flag:
    sys.argv.append(ema_flag)
evaluate_4step.main()
PY
