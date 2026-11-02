# Shared runtime for the 4D GS jobs on HPC2 (sourced by the sbatch files).
source $HOME/pandwm_env.sh
H=$HOME
GS4D=$H/pandwm_code/PanDWM-gs4d-20261002
GSAA=$H/pandwm_code/PanDWM-gs-aa-ft
export PYTHONPATH=$GS4D/src:$GSAA/src:$H/pandwm_code/TATS_fvd
export TORCH_EXTENSIONS_DIR=$H/.cache/pandwm_torch_extensions
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
unset HF_AUTO_UPLOAD_REPO || true
RUNS=$H/pandwm_runs/gs4d_20261002
