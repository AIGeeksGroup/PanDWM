#!/bin/bash
#SBATCH --job-name=pandwm_common
#SBATCH --partition=<partition>
#SBATCH --nodes=1
# Single rank: 149 clips do not split evenly, and DistributedSampler would drop the last clip.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
#
# Score PanDWM on the common-clip protocol: 149 clips, 3 context frames, 32
# predicted frames, front-camera band at 384x768, uint8 round trip, I3D-400.
# Usage: [ITEMS=149] [SEED=42] [INDICES=..] [VIS_DIR=..] sbatch pandwm_common_front.sbatch
set -euo pipefail

ITEMS=${ITEMS:-149}
SEED=${SEED:-42}
INDICES=${INDICES:-}
VIS_DIR=${VIS_DIR:-}
H=/path/to/pandwm_root
P=$H/pandwm_code/baselines
CFG_SRC=$H/pandwm_code/eval/manifests/cd_eval_frozen149_hpc3.json
RUN=$H/pandwm_runs/common_front_items${ITEMS}_seed${SEED}
I3D=$H/pandwm_code/TATS_fvd/i3d_pretrained_400.pt

for path in "$CFG_SRC" "$I3D" "$H/pandwm_data/cd_10000_ema.pth"; do
  test -f "$path" || { echo "missing: $path" >&2; exit 2; }
done

source /path/to/conda/bin/activate pandwm
export PYTHONPATH=$H/pandwm_code/src:$H/pandwm_code/TATS_fvd
export PANDWM_4STEP_SAMPLER=stochastic
# Disable the panorama cross-fade: the seam lies inside the front camera's field of view and
# blending would alter the front-view pixels that are scored.
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
# Four-step sampling schedule.
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
  "$P/common/run_pandwm_common.py" \
  --config "$CFG" --run "$RUN"

echo "completed: $RUN"
