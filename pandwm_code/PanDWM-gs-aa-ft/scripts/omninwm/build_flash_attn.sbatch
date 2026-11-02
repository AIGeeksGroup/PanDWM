#!/bin/bash
#SBATCH --job-name=omni_flash
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
V=$H/omninwm_venv
W=$H/external_models/omninwm_wheelhouse

module load gcc/11.5
module load cuda/12.1
export PIP_NO_INDEX=1
WHEEL=$W/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
WHEEL_SHA=a8719a86d0cd12a105739151f9e45ac8c2ecb840a83ed7a080a88d1645a98440

nvcc --version
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
"$V/bin/python" - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch", torch.__version__, "torch_cuda", torch.version.cuda, "cuda_home", CUDA_HOME)
PY

test -f "$WHEEL"
echo "$WHEEL_SHA  $WHEEL" | sha256sum --check --strict
"$V/bin/pip" install --no-index --no-deps "$WHEEL"

"$V/bin/python" - <<'PY'
import flash_attn
import torch
from flash_attn import flash_attn_func

q = torch.randn(1, 32, 8, 64, device="cuda", dtype=torch.float16)
y = flash_attn_func(q, q, q)
print("flash_attn", flash_attn.__version__, "smoke_shape", tuple(y.shape), "finite", bool(torch.isfinite(y).all()))
PY
