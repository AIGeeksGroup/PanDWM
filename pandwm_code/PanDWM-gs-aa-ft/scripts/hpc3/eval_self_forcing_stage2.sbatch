#!/bin/bash
#SBATCH --job-name=pandwm_sfeval
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

ARM=${ARM:?ARM must be baseline|control|self}
HORIZON=${HORIZON:-short}
SEED=${SEED:-0}
RUN_TAG=${RUN_TAG:-}
ITEMS=${ITEMS:-}
case "$ARM" in baseline|control|self) ;; *) exit 2 ;; esac
case "$HORIZON" in short|long) ;; *) exit 2 ;; esac

DATA_ROOT=/path/to/pandwm_root
CODE_ROOT=$DATA_ROOT/pandwm_code/PanDWM_self_forcing_20260915
BASE_CONFIG=$DATA_ROOT/pandwm_code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_hpc3.json
EVAL_ROOT=$DATA_ROOT/pandwm_runs/self_forcing_stage2_20260915/evaluation
RUN=$EVAL_ROOT/${RUN_TAG}${HORIZON}_${ARM}_seed${SEED}
CHECKPOINT=$DATA_ROOT/pandwm_data/cd_10000_ema.pth
if [ "$ARM" != baseline ]; then
  CHECKPOINT=$DATA_ROOT/pandwm_runs/self_forcing_stage2_20260915/production/$ARM/checkpoints/8000_ema.pth
fi
for path in "$CHECKPOINT" "$BASE_CONFIG"; do
  test -f "$path" || { echo "missing file: $path" >&2; exit 2; }
done
test ! -e "$RUN" || { echo "refusing to overwrite $RUN" >&2; exit 2; }
mkdir -p "$RUN/checkpoints" "$RUN/eval_configs"
ln -s "$CHECKPOINT" "$RUN/checkpoints/10000_ema.pth"
CONFIG=$RUN/eval_configs/config.json

source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$CODE_ROOT/src:$DATA_ROOT/pandwm_code/TATS_fvd
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 500))
export TORCH_HOME=$DATA_ROOT/.cache/torch
unset HF_AUTO_UPLOAD_REPO || true

"$DATA_ROOT/pandwm_venv/bin/python" - \
    "$BASE_CONFIG" "$CONFIG" "$SEED" "$HORIZON" "$CHECKPOINT" "$ITEMS" <<'PY'
import json
import sys

src, dst, seed, horizon, checkpoint, items = sys.argv[1:]
with open(src, encoding="utf-8") as handle:
    config = json.load(handle)
config["generator_seed"] = int(seed)
config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
pipeline = config["pipeline"]
pipeline["denoising_step_list"] = [1000, 625, 250, 42]
pipeline["metrics"] = {
    k: v for k, v in pipeline["metrics"].items() if k in ("fid", "fvd")
}
dataset = config["validation_dataset"]["base_dataset"]
if horizon == "short":
    # Exactly the established 149 valid 35-frame val scenes. Pipeline scores
    # 32 generated frames for FID; PanoPatch-FVD takes the first 16.
    dataset["sequence_length"] = 35
    count, fvd_frames = 149, 16
else:
    # The same timestamp enumerator finds 148 valid 67-frame val scenes.
    # Do not pool these with the 149-scene short-horizon statistic.
    dataset["sequence_length"] = 67
    count, fvd_frames = 148, 64
dataset["fps_stride_tuples"] = [[10, 20]]
pipeline["inference_config"]["evaluation_item_count"] = (
    int(items) if items else count)
pipeline["metrics"]["fvd"]["sequence_count"] = fvd_frames
config.setdefault("informations", {}).update({
    "evaluation_role": "paired from-Stage1 Self-Forcing-inspired CD",
    "checkpoint": checkpoint,
    "horizon": horizon,
    "scene_count": int(items) if items else count,
    "fid_generated_frames": 32 if horizon == "short" else 64,
    "fvd_generated_frames": fvd_frames,
    "sampler": "stochastic",
    "schedule": [1000, 625, 250, 42],
    "pano_blend_width": 128,
})
with open(dst, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print("eval", horizon, "scenes", count, "FVD frames", fvd_frames)
PY

cd "$CODE_ROOT"
"$DATA_ROOT/pandwm_venv/bin/python" - "$CONFIG" "$RUN" <<'PY'
import sys
import types

omninwm = types.ModuleType("omninwm")
omninwm.__path__ = []
utils = types.ModuleType("omninwm.utils")
utils.__path__ = []
hardseam = types.ModuleType("omninwm.utils.pano_hardseam")
def unavailable(*args, **kwargs):
    raise RuntimeError("pre-rendered panorama conditions required")
hardseam.stitch_hardseam = unavailable
omninwm.utils = utils
utils.pano_hardseam = hardseam
sys.modules.update({
    "omninwm": omninwm,
    "omninwm.utils": utils,
    "omninwm.utils.pano_hardseam": hardseam,
})

from dwm_pano import evaluate_4step
from dwm_pano.pipelines import ctsd_pano_consistency_distillation as cd

original = cd.PanoCrossviewTemporalConsistencyDistillation._build_auxiliary_model
def auxiliary(self, checkpoint_path, model_load_state_args, requires_grad, role):
    if role == "teacher" and hasattr(self, "model_wrapper"):
        print("eval shim: alias teacher to main model", flush=True)
        return self.model, self.model_wrapper
    return original(self, checkpoint_path, model_load_state_args,
                    requires_grad, role)
cd.PanoCrossviewTemporalConsistencyDistillation._build_auxiliary_model = auxiliary

config, run = sys.argv[1:]
sys.argv = ["evaluate_4step.py", "-c", config, "-r", run,
            "-s", "10000", "--ema"]
evaluate_4step.main()
PY
echo "completed arm=$ARM horizon=$HORIZON seed=$SEED run=$RUN"
if [ "$ARM" = self ] && [ "$HORIZON" = long ] && \
   [ "$SEED" = 1000 ] && [ -z "$RUN_TAG" ] && [ -z "$ITEMS" ]; then
  "$DATA_ROOT/pandwm_venv/bin/python" \
    "$CODE_ROOT/scripts/hpc3/summarize_self_forcing_metrics.py" \
    --root "$EVAL_ROOT"
fi
