#!/usr/bin/env bash
#SBATCH --job-name=pandwm_gsplat_smoke
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs4d_hpc3_20261017/logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
# 2026-10-17: build the golden gsplat CUDA extension for H100 (sm_90) and run one rasterization.
set -uo pipefail
GOLDEN=1 source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/gsplat_env_hpc3.sh
nvidia-smi --query-gpu=name --format=csv,noheader; which nvcc ninja; nvcc --version | tail -1
/path/to/pandwm_root/pandwm_venv/bin/python - <<PY
import time, torch, gsplat
t0 = time.time()
N = 2000
means = torch.randn(N, 3, device="cuda") * 2 + torch.tensor([0, 0, 8.0], device="cuda")
quats = torch.nn.functional.normalize(torch.randn(N, 4, device="cuda"), dim=-1)
scales = torch.rand(N, 3, device="cuda") * 0.1
opac = torch.rand(N, device="cuda")
cols = torch.rand(N, 3, device="cuda")
view = torch.eye(4, device="cuda")[None]
K = torch.tensor([[[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]]], device="cuda")
img, alpha, _ = gsplat.rasterization(means, quats, scales, opac, cols, view, K, 320, 240, render_mode="RGB+ED",
                                     eps2d=0.1, rasterize_mode="antialiased")
torch.cuda.synchronize()
print("GSPLAT_SMOKE_OK", tuple(img.shape), float(alpha.mean()), "build+render %.1f s" % (time.time() - t0), gsplat.__version__)
PY
ls -la \$TORCH_EXTENSIONS_DIR/gsplat_cuda 2>/dev/null | head -5
echo SMOKE_END
