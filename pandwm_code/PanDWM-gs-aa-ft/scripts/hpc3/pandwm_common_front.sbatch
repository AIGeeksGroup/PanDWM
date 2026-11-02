#!/bin/bash
#SBATCH --job-name=pandwm_common
#SBATCH --partition=<partition>
#SBATCH --nodes=1
# 149 clips is not divisible by four.  Keep the formal common protocol on one
# rank so the evaluator does not truncate the final clip for DistributedSampler.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
#
# Re-score PanDWM on the exact HorizonDrive task-matched interchange:
# 149 clips, first 3 frames as context, all 32 predicted frames, CAM_FRONT
# band at 384x768, uint8 round-trip, TATS I3D-400.
set -euo pipefail

ITEMS=${ITEMS:-149}
SEED=${SEED:-42}
INDICES=${INDICES:-}
VIS_DIR=${VIS_DIR:-}
H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
CFG_SRC=$P/configs/ctsd/pandwm/h20/cd_eval_hpc3.json
RUN=$H/pandwm_runs/common_front_items${ITEMS}_seed${SEED}
I3D=$H/pandwm_code/TATS_fvd/i3d_pretrained_400.pt

for path in "$CFG_SRC" "$I3D" "$H/pandwm_data/cd_10000_ema.pth"; do
  test -f "$path" || { echo "missing: $path" >&2; exit 2; }
done

source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$P/src:$H/pandwm_code/TATS_fvd
export PANDWM_4STEP_SAMPLER=stochastic
# The HorizonDrive adapter constructs this exact hard-deduplicated canvas.
# Cross-fading the PanDWM canvas here would change CAM_FRONT GT pixels because
# the 21.6-degree internal seam lies inside the front camera's field of view.
export PANDWM_PANO_BLEND_W=0
if [[ -n "$INDICES" ]]; then
  export PANDWM_EVAL_INDICES="$INDICES"
fi
if [[ -n "$VIS_DIR" ]]; then
  export PANDWM_COMMON_VIS_DIR="$VIS_DIR"
fi
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_HOME=$H/.cache/torch

NUM_PROCESSES=${SLURM_GPUS_ON_NODE:-1}
if [[ "$NUM_PROCESSES" == *\(* ]]; then
  NUM_PROCESSES=${NUM_PROCESSES%%\(*}
fi
test "$NUM_PROCESSES" -ge 1
mkdir -p "$RUN/checkpoints" "$RUN/eval_configs"
ln -sf "$H/pandwm_data/cd_10000_ema.pth" \
  "$RUN/checkpoints/10000_ema.pth"
CFG=$RUN/eval_configs/cfg.json

python - "$CFG_SRC" "$CFG" "$SEED" "$ITEMS" "$NUM_PROCESSES" \
        "$I3D" <<'PY'
import json
import sys

source, target, seed, items, processes, i3d = sys.argv[1:]
config = json.load(open(source, encoding="utf-8"))
config["generator_seed"] = int(seed)
config["global_state"]["device_mesh"]["mesh_shape"] = [1, int(processes)]
pipeline = config["pipeline"]
pipeline["inference_config"]["evaluation_item_count"] = int(items)
# Frozen best four-step schedule used by the s13bl PanDWM arm.
pipeline["denoising_step_list"] = [1000, 625, 250, 42]
pipeline["metrics"] = {
    "n6bridge": {
        "_class_name": (
            "dwm_pano.metrics.nuscenes6_fvd."
            "NuScenes6BandFrechetVideoDistance"),
        "inception_3d_checkpoint_path": i3d,
        "sequence_count": 32,
        "render_height": 384,
        "render_width": 768,
        "quantize_uint8": True,
        "compute_front_fid": True,
    }
}
config.setdefault("informations", {}).update({
    "evaluation_role": "PanDWM task-matched HorizonDrive comparison",
    "common_protocol": {
        "clips": int(items), "fps": 10, "context_frames": 3,
        "prediction_frames_scored": 32,
        "camera": "CAM_FRONT band", "render_size": [384, 768],
        "quantization": "round(clamp(x,0,1)*255)/255",
        "i3d": i3d,
    },
})
json.dump(config, open(target, "w", encoding="utf-8"), indent=2)
print("wrote", target)
PY

echo "host=$(hostname) processes=$NUM_PROCESSES items=$ITEMS seed=$SEED"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
cd "$P"
"$H/pandwm_venv/bin/python" -m torch.distributed.run --standalone \
  --nproc_per_node="$NUM_PROCESSES" \
  "$P/scripts/hpc3/run_pandwm_common.py" \
  --config "$CFG" --run "$RUN"

echo "completed: $RUN"
