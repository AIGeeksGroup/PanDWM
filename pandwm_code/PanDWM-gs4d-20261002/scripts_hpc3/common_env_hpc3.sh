# 2026-10-17: shared runtime for the 4D GS depth pilot on HPC3 (sourced by the sbatch files).
DR=/path/to/pandwm_root
source $DR/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/gsplat_env_hpc3.sh
GS4D=$DR/pandwm_code/PanDWM-gs4d-20261002
GSAA=$DR/pandwm_code/PanDWM-gs-aa-ft
PY=$DR/pandwm_venv/bin/python
export PYTHONPATH=$GS4D/src:$GSAA/src:$DR/pandwm_code/TATS_fvd:$PYTHONPATH
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=$DR/.cache/torch
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
unset HF_AUTO_UPLOAD_REPO || true
RUNS=$DR/pandwm_runs/gs4d_hpc3_20261017
