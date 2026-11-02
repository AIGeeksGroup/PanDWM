#!/bin/bash
#SBATCH --job-name=omni_probe
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
echo "PATH=$PATH"
command -v nvcc || true
nvcc --version || true
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
"$H/omninwm_venv/bin/python" - <<'PY'
import sys
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("python", sys.version)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_home", CUDA_HOME)
print("capability", torch.cuda.get_device_capability())
PY
