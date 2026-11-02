#!/bin/bash
#SBATCH --job-name=omni_mmcv
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
V=$H/omninwm_venv
W=$H/external_models/omninwm_wheelhouse

module load gcc/11.5
module load cuda/12.1
export CUDA_HOME
CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
BASE_BIN=$(dirname "$(readlink -f "$V/bin/python")")
export PATH=$V/bin:$BASE_BIN:$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export PIP_NO_INDEX=1

nvcc --version
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
"$V/bin/python" - <<'PY'
from torch.utils.cpp_extension import is_ninja_available
assert is_ninja_available(), "venv ninja executable is not visible"
print("ninja_available", is_ninja_available())
PY
"$V/bin/pip" install --no-index --no-build-isolation --no-deps \
  "$W/mmcv-2.2.0.tar.gz"

"$V/bin/python" - <<'PY'
import mmcv
import torch
from mmcv.ops import DeformConv2dPack

m = DeformConv2dPack(4, 4, 3, padding=1).cuda().half()
x = torch.randn(1, 4, 8, 8, device="cuda", dtype=torch.float16)
y = m(x)
print("mmcv", mmcv.__version__, "smoke_shape", tuple(y.shape), "finite", bool(torch.isfinite(y).all()))
PY
