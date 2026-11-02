#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="$PANDWM_ROOT/envs/pandwm"
# Default to Aliyun: it is the mirror this machine's own pip.conf uses and the
# only one verified to serve torch wheels here (tuna resolved torch to nothing).
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [[ ! -x "$CONDA_BIN" ]]; then
    printf 'Conda executable not found: %s\n' "$CONDA_BIN" >&2
    exit 1
fi

mkdir -p "$PANDWM_ROOT/envs"
if [[ ! -f "$ENV_PREFIX/conda-meta/history" ]]; then
    if [[ -e "$ENV_PREFIX" ]]; then
        printf 'Path exists but is not a conda environment: %s\n' "$ENV_PREFIX" >&2
        exit 1
    fi
    "$CONDA_BIN" create --yes --prefix "$ENV_PREFIX" python=3.11 pip
fi

eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip setuptools wheel

# PyPI torch 2.5.1 wheels use CUDA 12.4 and include the FlexAttention API.
python -m pip install \
    'torch==2.5.1' \
    'torchvision==0.20.1'

# Evaluation-only dependencies. Training-only packages such as wandb,
# bitsandbytes, chamferdist, pandas, and pyarrow are intentionally omitted.
python -m pip install \
    'accelerate==1.4.0' \
    'av==14.0.0' \
    'diffusers==0.31.0' \
    'einops' \
    'fsspec' \
    'ninja' \
    'numpy==1.26.4' \
    'Pillow' \
    'protobuf==4.21.6' \
    'pyquaternion' \
    'safetensors' \
    'scipy' \
    'sentencepiece' \
    'tensorboard' \
    'torch-fidelity' \
    'torchmetrics==1.6.0' \
    'tqdm' \
    'transformers==4.50.0' \
    'transforms3d' \
    'wandb'

# P1 only: install gsplat when the Gaussian decoder work begins.

python - <<'PY'
import torch
import torchmetrics
from torch.nn.attention.flex_attention import BlockMask

if torch.version.cuda != "12.4":
    raise RuntimeError(
        "Expected a CUDA 12.4 torch wheel, got torch.version.cuda="
        + repr(torch.version.cuda)
    )
if tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) < (2, 5):
    raise RuntimeError("PyTorch 2.5 or newer is required")
if torchmetrics.__version__ != "1.6.0":
    raise RuntimeError("torchmetrics must remain pinned to 1.6.0")
print("Environment verification passed")
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("torchmetrics:", torchmetrics.__version__)
print("FlexAttention BlockMask:", BlockMask.__name__)
PY

# Import smoke over the real evaluation module chain (PYTHONPATH from env.sh).
# Catches missing runtime deps now instead of at first eval launch.
python - <<'PY'
import dwm.common
import dwm_pano.evaluate_4step
import dwm_pano.pipelines.ctsd_pano_consistency_distillation
import dwm_pano.datasets.nuscenes_pano
import dwm_pano.metrics.fvd_pano
import pytorch_i3d
print("Eval import chain OK")
PY
