#!/bin/bash
#SBATCH --job-name=pandwm_50
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.log
# 50-step baseline evaluation on HPC3.
# The instrument is aligned item by item with scripts/h20/eval_stage1_50step_h20.sh,
# which produced the teacher numbers FID 10.235 / PanoPatch-FVD 49.045. The config
# merge order, mesh shape, single-GPU launch and frozen protocol fields must match
# that script or the numbers cannot be placed next to 49.045.
#
#   sbatch --export=ALL,ARM=stage0,ITEMS=5      base50.sbatch   # smoke
#   sbatch --export=ALL,ARM=unimlvg,ITEMS=149   base50.sbatch
#   sbatch --export=ALL,ARM=stage1,ITEMS=149    base50.sbatch   # teacher re-check
set -eu
ARM=${ARM:?ARM must be stage0|unimlvg|stage1}
ITEMS=${ITEMS:-149}; SEED=${SEED:-0}; METRICS=${METRICS:-seam_smooth_frameprof}
H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM

case "$ARM" in
  stage0)
    BASE=$P/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_bidirectional_nwa.json
    CKPT=$H/pandwm_data/stage0_bidirectional_20000.pth
    MASK=none   ; ROLE=stage0_bidirectional_teacher_50step ;;
  unimlvg)
    # OpenDWM's released UniMLVG pretrain. Both PanDWM stage configs already list
    # this file as model_checkpoint_path, i.e. it is the initialization PanDWM
    # Stage-0 started from, and it loads under strict=True against the same JSON
    # model block. It has never seen panorama data, so this arm measures what
    # PanDWM's own Stage-0 training added, not method-versus-method.
    BASE=$P/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_bidirectional_nwa.json
    CKPT=$H/pandwm_data/ctsd_unimlvg_tirda_bm_nwa_60k.pth
    MASK=none   ; ROLE=unimlvg_pretrain_60k_50step ;;
  stage1)
    BASE=$P/configs/ctsd/pandwm/ctsd_unimlvg_stage3_tirda_causal_nwa.json
    CKPT=$H/pandwm_data/stage1_causal_20000.pth
    MASK=causal ; ROLE=stage1_causal_teacher_50step ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac
HPC_CFG=$P/configs/ctsd/pandwm/h20/cd_eval_hpc3.json
RUN=$H/pandwm_runs/base50_${ARM}_items${ITEMS}_seed${SEED}

for f in "$BASE" "$HPC_CFG" "$CKPT"; do
  [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done
echo "=== $(hostname) ==="; nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
ls -la "$CKPT"

source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$P/src:$H/pandwm_code/TATS_fvd
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PANDWM_PANO_BLEND_W=0
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 500))
mkdir -p "$RUN/eval_configs"
CFG=$RUN/eval_configs/cfg.json

python - "$BASE" "$HPC_CFG" "$CKPT" "$CFG" "$SEED" "$ITEMS" "$MASK" "$ROLE" \
        "$METRICS" "$RUN" <<'PY'
import copy, json, sys
base_p, hpc_p, ckpt, out, seed, items, mask, role, metrics, run = sys.argv[1:]
config = json.load(open(base_p, encoding="utf-8"))
hpc = json.load(open(hpc_p, encoding="utf-8"))

# merge order identical to eval_stage1_50step_h20.sh
config["generator_seed"] = int(seed)
config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
for k in ("training_dataset", "validation_dataset", "training_dataloader",
          "validation_dataloader", "preview_dataloader"):
    if k in hpc:
        config[k] = copy.deepcopy(hpc[k])

p = config["pipeline"]
p["pretrained_model_name_or_path"] = hpc["pipeline"]["pretrained_model_name_or_path"]
p["model_checkpoint_path"] = ckpt
p["metrics"] = copy.deepcopy(hpc["pipeline"]["metrics"])

# load_state_dict runs with strict=True here (model_load_state_args is absent in
# every pandwm config), so a key or shape mismatch raises instead of silently
# leaving a submodule randomly initialized. Print the report anyway: an empty
# missing/unexpected pair is the positive evidence that the whole checkpoint
# was consumed.
assert "model_load_state_args" not in p, \
    "model_load_state_args is set; strict loading is no longer guaranteed"
p.setdefault("common_config", {})["print_load_state_info"] = True

if mask == "causal":
    p.setdefault("training_config", {})["temporal_attention_mask"] = "causal"
    p["inference_config"]["temporal_attention_mask"] = "causal"
else:
    # Bidirectional arm: verified that this config sets no temporal_attention_mask
    # anywhere. Assert instead of popping, so a future config edit that quietly
    # adds causal masking fails loudly rather than being silently undone here.
    assert "temporal_attention_mask" not in p.get("training_config", {}), \
        "bidirectional base config unexpectedly sets training temporal mask"
    assert "temporal_attention_mask" not in p["inference_config"], \
        "bidirectional base config unexpectedly sets inference temporal mask"
p["inference_config"]["inference_steps"] = 50
p["inference_config"]["evaluation_item_count"] = int(items)

extra = {
    "seam": {"_class_name": "dwm_pano.metrics.seam_profile.PanoSeamProfile",
             "save_path": f"{run}/seam_profile.pt"},
    "smooth": {"_class_name":
               "dwm_pano.metrics.temporal_smoothness.PanoTemporalSmoothness"},
    "frameprof": {"_class_name":
                  "dwm_pano.metrics.frame_profile.PanoFrameProfile"},
}
want = [m for m in metrics.replace("_", ",").split(",") if m]
for name in want:
    assert name in extra, f"unknown metric {name}"
    p["metrics"][name] = extra[name]

config.setdefault("informations", {}).update({
    "evaluation_role": role, "checkpoint": ckpt, "source_config": base_p,
    "protocol": "matches scripts/h20/eval_stage1_50step_h20.sh (1 GPU, "
                "mesh [1,1], 50 steps, 3+16 window, seed 0)"})
json.dump(config, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("role:", role, "| mask:", mask, "| steps: 50 | items:", items,
      "| metrics:", ["fid", "fvd"] + want)
PY

echo "=== ARM=$ARM ROLE=$ROLE ITEMS=$ITEMS SEED=$SEED METRICS=$METRICS ==="
cd "$P"
$H/pandwm_venv/bin/python - "$CFG" "$RUN" <<'PY'
import sys, types
# Same shim as the h20 script: the online stitcher must hard-fail so a missing
# pre-rendered bbox/HD-map panorama cannot silently change the evaluation input.
omninwm = types.ModuleType("omninwm"); omninwm.__path__ = []
u = types.ModuleType("omninwm.utils"); u.__path__ = []
ph = types.ModuleType("omninwm.utils.pano_hardseam")
def unavailable(*a, **k):
    raise RuntimeError("online pano stitching unavailable; pre-rendered PNG required")
ph.stitch_hardseam = unavailable; omninwm.utils = u; u.pano_hardseam = ph
sys.modules.update({"omninwm": omninwm, "omninwm.utils": u,
                    "omninwm.utils.pano_hardseam": ph})

from dwm_pano import evaluate
cfg, run = sys.argv[1:]
sys.argv = ["evaluate.py", "-c", cfg, "-o", run]
evaluate.__name__ = "__main__"
exec(open(evaluate.__file__, "rb").read(), evaluate.__dict__)
PY
echo "=== done: ARM=$ARM ==="
