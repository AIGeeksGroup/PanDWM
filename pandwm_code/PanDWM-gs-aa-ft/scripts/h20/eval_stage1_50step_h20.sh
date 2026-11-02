#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="$PANDWM_ROOT/envs/pandwm"
BASE_CFG="${BASE_CFG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_causal_nwa.json}"
H20_CFG="${H20_CFG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_h20.json}"
CHECKPOINT="${CHECKPOINT:-$PANDWM_ROOT/ckpts/stage1_causal_20000.pth}"
RUN="${RUN:-$PANDWM_ROOT/runs/stage1_causal_50step_eval}"
GENERATOR_SEED="${GENERATOR_SEED:-0}"
EVAL_ITEMS="${EVAL_ITEMS:-480}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29529}"

for path in "$BASE_CFG" "$H20_CFG" "$CHECKPOINT"; do
    if [[ ! -f "$path" ]]; then
        printf 'Required file is missing: %s\n' "$path" >&2
        exit 1
    fi
done
if [[ ! "$EVAL_ITEMS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'EVAL_ITEMS must be a positive integer, got: %s\n' "$EVAL_ITEMS" >&2
    exit 1
fi

pandwm_pick_cuda_device

mkdir -p "$RUN" "$PANDWM_ROOT/logs" "$PANDWM_ROOT/torchinductor_cache"
CONFIG_SNAPSHOT="$RUN/stage1_causal_50step_seed${GENERATOR_SEED}_items${EVAL_ITEMS}.json"

python3 - "$BASE_CFG" "$H20_CFG" "$CHECKPOINT" "$CONFIG_SNAPSHOT" \
    "$GENERATOR_SEED" "$EVAL_ITEMS" <<'PY'
import copy
import json
import sys

base_path, h20_path, checkpoint, output, seed, eval_items = sys.argv[1:]
with open(base_path, "r", encoding="utf-8") as f:
    config = json.load(f)
with open(h20_path, "r", encoding="utf-8") as f:
    h20 = json.load(f)

config["generator_seed"] = int(seed)
config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
config["training_dataset"] = copy.deepcopy(h20["training_dataset"])
config["validation_dataset"] = copy.deepcopy(h20["validation_dataset"])
config["training_dataloader"] = copy.deepcopy(h20["training_dataloader"])
config["validation_dataloader"] = copy.deepcopy(h20["validation_dataloader"])
config["preview_dataloader"] = copy.deepcopy(h20["preview_dataloader"])

pipeline = config["pipeline"]
pipeline["pretrained_model_name_or_path"] = \
    h20["pipeline"]["pretrained_model_name_or_path"]
pipeline["model_checkpoint_path"] = checkpoint
pipeline["metrics"] = copy.deepcopy(h20["pipeline"]["metrics"])
pipeline["training_config"]["temporal_attention_mask"] = "causal"
pipeline["inference_config"]["temporal_attention_mask"] = "causal"
pipeline["inference_config"]["inference_steps"] = 50
pipeline["inference_config"]["evaluation_item_count"] = int(eval_items)

config.setdefault("informations", {})["evaluation_role"] = \
    "stage1_causal_teacher_50step"
config["informations"]["checkpoint"] = checkpoint
config["informations"]["source_config"] = base_path

with open(output, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(output)
PY

eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$ENV_PREFIX"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$PANDWM_ROOT/torchinductor_cache"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# DeviceMesh initializes through torch.distributed even for a [1,1] mesh.
# A one-process group also preserves the same DistributedSampler/metric path
# used by the existing Stage-2 evaluation.
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR
export MASTER_PORT

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$PANDWM_ROOT/logs/eval_stage1_50step_seed${GENERATOR_SEED}_items${EVAL_ITEMS}_${TIMESTAMP}.log"

printf 'PanDWM Stage-1 causal teacher 50-step evaluation\n' | tee "$LOG"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
printf 'Config snapshot: %s\n' "$CONFIG_SNAPSHOT" | tee -a "$LOG"
printf 'Checkpoint: %s\n' "$CHECKPOINT" | tee -a "$LOG"
printf 'Seed/items: %s / %s\n' "$GENERATOR_SEED" "$EVAL_ITEMS" | tee -a "$LOG"
printf 'Log: %s\n' "$LOG" | tee -a "$LOG"

python - "$CONFIG_SNAPSHOT" "$RUN" <<'PY' 2>&1 | tee -a "$LOG"
import sys
import types

# h20 stores pre-rendered bbox/HD-map panoramas. Keep the online stitcher a
# hard failure so missing assets cannot silently change the evaluation input.
omninwm = types.ModuleType("omninwm")
omninwm.__path__ = []
omninwm_utils = types.ModuleType("omninwm.utils")
omninwm_utils.__path__ = []
pano_hardseam = types.ModuleType("omninwm.utils.pano_hardseam")


def unavailable_stitcher(*args, **kwargs):
    raise RuntimeError(
        "Online pano stitching is unavailable on h20; a pre-rendered bbox "
        "or HD-map panorama is missing"
    )


pano_hardseam.stitch_hardseam = unavailable_stitcher
omninwm.utils = omninwm_utils
omninwm_utils.pano_hardseam = pano_hardseam
sys.modules["omninwm"] = omninwm
sys.modules["omninwm.utils"] = omninwm_utils
sys.modules["omninwm.utils.pano_hardseam"] = pano_hardseam

from dwm_pano import evaluate

config_path, output_path = sys.argv[1:]
sys.argv = ["evaluate.py", "-c", config_path, "-o", output_path]
evaluate.__name__ = "__main__"
exec(open(evaluate.__file__, "rb").read(), evaluate.__dict__)
PY
