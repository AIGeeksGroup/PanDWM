#!/bin/bash
#SBATCH --job-name=hdrive_eval
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.log
#
# HorizonDrive baseline evaluation (official checkpoints) on the 149 common clips.
# Usage: sbatch --export=ALL,HD_SEED=42 eval_hpc3.sbatch
set -euo pipefail

HD_SEED=${HD_SEED:-42}
HD_HOME=${HD_HOME:-$HOME/horizondrive_repro}
HD_REPO=${HD_REPO:-$HD_HOME/HorizonDrive}
HD_MODELS=${HD_MODELS:-$HD_REPO/models}
HD_ENV=${HD_ENV:-$HOME/horizondrive_venv}
HD_NUSCENES_ROOT=${HD_NUSCENES_ROOT:-$HOME/horizondrive_data/nuScenes/origin}
HD_ANN_PATH=${HD_ANN_PATH:-$HOME/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl}
HD_PANO_RGB_DIR=${HD_PANO_RGB_DIR:-$HOME/pandwm_data/rgb}
HD_PANO_BBOX_DIR=${HD_PANO_BBOX_DIR:-$HOME/pandwm_data/bbox}
HD_PANO_HDMAP_DIR=${HD_PANO_HDMAP_DIR:-$HOME/pandwm_data/hdmap}
HD_OUTPUT_ROOT=${HD_OUTPUT_ROOT:-$HOME/horizondrive_runs}
HD_PANDWM=${HD_PANDWM:-$HOME/pandwm_code/baselines}

HD_CONFIG=$HD_PANDWM/horizondrive/horizondrive_hpc3_common.yaml

export HD_REPO HD_MODELS HD_NUSCENES_ROOT HD_ANN_PATH
export HD_PANO_RGB_DIR HD_PANO_BBOX_DIR HD_PANO_HDMAP_DIR
export HF_HOME=${HF_HOME:-$HD_HOME/cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-$HD_HOME/cache/torch}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HD_PANDWM/../src:$HD_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /path/to/conda/bin/activate pandwm

for path in "$HD_REPO" "$HD_ENV/bin/python" "$HD_CONFIG"; do
  if [ ! -e "$path" ]; then
    echo "missing prerequisite: $path" >&2
    exit 2
  fi
done

HD_RUN_ROOT=$HD_OUTPUT_ROOT/common
mkdir -p "$HD_RUN_ROOT"
cd "$HD_REPO"

HD_EVAL_ENTRY=$HD_PANDWM/horizondrive/eval_common.py
export HD_COMMON_DUMP_DIR=${HD_COMMON_DUMP_DIR:-$HD_RUN_ROOT/job_${SLURM_JOB_ID}/lossless}

NUM_PROCESSES=${SLURM_GPUS_ON_NODE:-1}
if [[ "$NUM_PROCESSES" == *\(* ]]; then
  NUM_PROCESSES=${NUM_PROCESSES%%\(*}
fi
if [[ ! "$NUM_PROCESSES" =~ ^[0-9]+$ ]]; then
  echo "cannot parse SLURM_GPUS_ON_NODE=$NUM_PROCESSES" >&2
  exit 2
fi
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-$("$HD_ENV/bin/python" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)}
export NUM_PROCESSES MAIN_PROCESS_PORT

echo "host=$(hostname) protocol=common seed=$HD_SEED processes=$NUM_PROCESSES"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
for path in "$HD_ANN_PATH" "$HD_PANO_RGB_DIR" \
    "$HD_PANO_BBOX_DIR" "$HD_PANO_HDMAP_DIR"; do
  if [ ! -e "$path" ]; then
    echo "missing common-protocol data: $path" >&2
    exit 2
  fi
done

"$HD_ENV/bin/accelerate" launch \
  --num_processes="$NUM_PROCESSES" \
  --num_machines=1 \
  --main_process_port="$MAIN_PROCESS_PORT" \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  -- \
  "$HD_EVAL_ENTRY" \
  --train_mode=unified \
  --mixed_precision=bf16 \
  --seed="$HD_SEED" \
  --crossview_attn_type=full \
  --low_vram \
  --report_to=tensorboard \
  --config_path="$HD_CONFIG" \
  --pretrained_model_name_or_path="$HD_MODELS/Wan2.1-T2V-1.3B" \
  --output_dir="$HD_RUN_ROOT/job_${SLURM_JOB_ID}" \
  --vae_path="$HD_MODELS/horizondrive-vae.pkl" \
  --transformer_path="$HD_MODELS/horizondrive-dit.safetensors"

echo "completed: $HD_RUN_ROOT/job_${SLURM_JOB_ID}"
