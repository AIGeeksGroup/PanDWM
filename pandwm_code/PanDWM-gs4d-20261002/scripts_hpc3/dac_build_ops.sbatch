#!/usr/bin/env bash
#SBATCH --job-name=pandwm_dacbuild
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs4d_hpc3_20261017/logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
# 2026-10-18 (report section 37): build DAC's MultiScaleDeformableAttention op in place (needs a visible GPU:
# its setup.py refuses to build without torch.cuda.is_available()).  Same CUDA 12.8 + gcc 12.5 as gsplat.
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
DAC=/path/to/pandwm_root/pandwm_code/third_party/depth_any_camera
cd $DAC/dac/models/ops
echo "DACBUILD_START node=$(hostname) $(date -Is)"
$PY setup.py build_ext --inplace 2>&1 | tail -5
ls -la *.so
cd $DAC && PYTHONPATH=$DAC:$DAC/dac/models/ops:$PYTHONPATH $PY -c "
import torch, MultiScaleDeformableAttention as M
from dac.models.idisc import IDisc
print('op ok', M.__file__); print('IDisc import ok')"
echo "DACBUILD_END rc=$? $(date -Is)"
