#!/bin/bash
#SBATCH --job-name=vista_official_smoke
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
SRC=$H/external_baselines/Vista
WEIGHT=$H/external_models/vista/vista.safetensors
OPENCLIP=$H/external_models/open_clip/open_clip_pytorch_model.bin
INPUT=$H/vista_common_front_smoke/clip_0000/00.jpg
RUN=$H/vista_runs/official_smoke_${SLURM_JOB_ID}
EXPECTED_COMMIT=cc9821b4253ca7987c32757613d2fc2448fa9f5d
EXPECTED_SHA256=1b267bb753aa5c5e4d363f6e4c864dd4e343996dedc18b5ab20af6d20be34b98
EXPECTED_OPENCLIP_SHA256=9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4

test -d "$SRC/.git" || { echo "missing official Vista checkout" >&2; exit 2; }
test -f "$WEIGHT" || { echo "missing official Vista weight" >&2; exit 2; }
test -f "$OPENCLIP" || { echo "missing official OpenCLIP weight" >&2; exit 2; }
test -f "$INPUT" || { echo "missing common-protocol condition image" >&2; exit 2; }
test ! -e "$RUN" || { echo "refusing to overwrite $RUN" >&2; exit 2; }
test "$(git -C "$SRC" rev-parse HEAD)" = "$EXPECTED_COMMIT"
echo "$EXPECTED_SHA256  $WEIGHT" | sha256sum --check --strict
echo "$EXPECTED_OPENCLIP_SHA256  $OPENCLIP" | sha256sum --check --strict

mkdir -p "$RUN/code" "$RUN/outputs"
git -C "$SRC" archive --format=tar HEAD | tar -xf - -C "$RUN/code"
mkdir -p "$RUN/code/ckpts" "$RUN/code/image_folder"
ln -s "$WEIGHT" "$RUN/code/ckpts/vista.safetensors"
ln -s "$INPUT" "$RUN/code/image_folder/condition.jpg"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=$H/external_models/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$RUN/code
echo "official_commit=$EXPECTED_COMMIT"
echo "official_weight_sha256=$EXPECTED_SHA256"
echo "protocol=official IMG single-image conditioning; 25 frames; 50 DDIM steps"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
$H/vista_venv/bin/python - <<'PY'
import kornia, open_clip, pytorch_lightning, torch, torchvision
print({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "pytorch_lightning": pytorch_lightning.__version__,
    "kornia": kornia.__version__,
    "open_clip_path": open_clip.__file__,
})
PY

cd "$RUN/code"
$H/vista_venv/bin/python sample.py \
  --dataset IMG \
  --save "$RUN/outputs" \
  --action free \
  --n_rounds 1 \
  --n_frames 25 \
  --n_conds 1 \
  --seed 23 \
  --height 576 \
  --width 1024 \
  --cfg_scale 2.5 \
  --cond_aug 0 \
  --n_steps 50 \
  --rand_gen

find "$RUN/outputs" -type f -printf '%s %p\n' | sort
echo "completed run=$RUN"
