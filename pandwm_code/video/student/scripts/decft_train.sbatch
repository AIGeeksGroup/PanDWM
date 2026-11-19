#!/usr/bin/env bash
# fine-tune the SD3.5 VAE decoder on nuScenes training panoramas (eval_v7/vae_decoder_ft.py), 1 GPU.
# Encoder frozen, so every saved student latent stays decodable.  Extra arguments via DECFT_ARGS.
#   sbatch scripts/decft_train.sbatch
#SBATCH --job-name=pandwm_decft
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/decoder_ft/logs/%x_%j.log
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/eval
W=$DR/pandwm_runs/decoder_ft
PY=$DR/pandwm_venv/bin/python
export PYTHONPATH=$C:$DR/pandwm_code/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=$DR/.cache/torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo "DECFT_JOB_START $(date -Is) node=$(hostname) job=${SLURM_JOB_ID:-x}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$C"
"$PY" "$W/code/vae_decoder_ft.py" \
  --train-config "$DR/pandwm_code/video/student/configs/cfdmd_H_gan_v8_3_hpc3_8gpu.json" \
  --vae "$DR/pandwm_data/stable-diffusion-3.5-medium" --out "$W/${DECFT_RUN:-run2}" ${DECFT_ARGS:-}
rc=$?
echo "DECFT_JOB_END $(date -Is) rc=$rc"
exit $rc
