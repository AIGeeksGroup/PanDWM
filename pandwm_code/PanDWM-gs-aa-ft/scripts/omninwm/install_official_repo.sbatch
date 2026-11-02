#!/bin/bash
#SBATCH --job-name=omni_repo
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
V=$H/omninwm_venv
REPO=$H/external_baselines/OmniNWM

module load gcc/11.5
module load cuda/12.1
export CUDA_HOME
CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
BASE_BIN=$(dirname "$(readlink -f "$V/bin/python")")
export PATH=$V/bin:$BASE_BIN:$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8
export PIP_NO_INDEX=1

test "$(sed -n '1p' "$REPO/.git/refs/heads/master")" = \
  "8e2333ee0dfb7be7a477e51a97e644bf7ec53039"
cd "$REPO"
"$V/bin/python" - <<'PY'
from torch.utils.cpp_extension import is_ninja_available
assert is_ninja_available(), "venv ninja executable is not visible"
print("ninja_available", is_ninja_available())
PY
"$V/bin/pip" install -v -e . --no-deps --no-build-isolation

"$V/bin/python" - <<'PY'
import torch
import occ_pool_ext
import omninwm
import omninwm.models
from omninwm.models.mmdit.math import flash_attn_func, liger_rope
print("omninwm_imports_ok", omninwm.__file__, occ_pool_ext.__file__)
PY
