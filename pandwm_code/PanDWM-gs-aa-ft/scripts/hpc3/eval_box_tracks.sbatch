#!/bin/bash
#SBATCH --job-name=pandwm_boxeval
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
#
# Intended as an afterok dependency of train_box_tracks.sbatch:
#   sbatch --dependency=afterok:<train_job> \
#     --export=ALL,ARM=track,TRAIN_STEPS=600 eval_box_tracks.sbatch
set -euo pipefail

ARM=${ARM:-track}
TRAIN_STEPS=${TRAIN_STEPS:-600}
SEED=${SEED:-0}
ITEMS=${ITEMS:-149}
H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
TRAIN_RUN=$H/pandwm_runs/boxtrack_${ARM}_s${TRAIN_STEPS}_seed${SEED}
CHECKPOINT=$TRAIN_RUN/checkpoints/${TRAIN_STEPS}_ema.pth
RUN=$H/pandwm_runs/boxtrack_eval_${ARM}_s${TRAIN_STEPS}_seed${SEED}
CFG_SRC=$P/configs/ctsd/pandwm/h20/cd_eval_hpc3.json
I3D=$H/pandwm_code/TATS_fvd/i3d_pretrained_400.pt
PKL=$H/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl

case "$ARM" in control|track) ;; *) echo "ARM must be control or track" >&2; exit 2;; esac
for path in "$CHECKPOINT" "$CFG_SRC" "$I3D" "$PKL"; do
  test -f "$path" || { echo "missing file: $path" >&2; exit 2; }
done
test ! -e "$RUN" || { echo "refusing to overwrite $RUN" >&2; exit 2; }
mkdir -p "$RUN/checkpoints" "$RUN/eval_configs"
ln -s "$CHECKPOINT" "$RUN/checkpoints/10000_ema.pth"
CFG=$RUN/eval_configs/cfg.json

source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$P/src:$H/pandwm_code/TATS_fvd
export PANDWM_4STEP_SAMPLER=stochastic
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 500))
export TORCH_HOME=$H/.cache/torch
unset HF_AUTO_UPLOAD_REPO || true

"$H/pandwm_venv/bin/python" - \
  "$CFG_SRC" "$CFG" "$SEED" "$ITEMS" "$I3D" "$PKL" "$ARM" \
  "$CHECKPOINT" <<'PY'
import json
import sys

src, dst, seed, items, i3d, pkl, arm, checkpoint = sys.argv[1:]
c = json.load(open(src, encoding="utf-8"))
c["generator_seed"] = int(seed)
c["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
p = c["pipeline"]
p["inference_config"]["evaluation_item_count"] = int(items)
# Use the frozen schedule selected before this ablation.
p["denoising_step_list"] = [1000, 625, 250, 42]
cls = "dwm_pano.metrics.instance_consistency.InstanceAppearanceConsistency"
p["metrics"] = {
    "fid": {
        "_class_name": "torchmetrics.image.fid.FrechetInceptionDistance",
        "normalize": True,
    },
    "fvd": {
        "_class_name": "dwm_pano.metrics.fvd_pano.PanoPatchFrechetVideoDistance",
        "inception_3d_checkpoint_path": i3d,
        "sequence_count": 16,
    },
    "iacpatch": {
        "_class_name": cls, "pkl_path": pkl, "descriptor": "patch",
        "patch_size": 32, "min_pixels": 8, "backbone_device": "cuda",
    },
    "iacincep": {
        "_class_name": cls, "pkl_path": pkl, "descriptor": "inception",
        "min_pixels": 8, "backbone_device": "cuda",
    },
    "iacvgg": {
        "_class_name": cls, "pkl_path": pkl, "descriptor": "vgg",
        "min_pixels": 8, "backbone_device": "cuda",
    },
    "iacinbig": {
        "_class_name": cls, "pkl_path": pkl, "descriptor": "inception",
        "min_pixels": 32, "backbone_device": "cuda",
    },
}
c.setdefault("informations", {}).update({
    "evaluation_role": "paired 3D-box track consistency ablation",
    "arm": arm,
    "evaluated_checkpoint": checkpoint,
    "schedule": [1000, 625, 250, 42],
    "items": int(items),
})
json.dump(c, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("config", dst, "arm", arm, "checkpoint", checkpoint)
PY

cd "$P"
"$H/pandwm_venv/bin/python" - "$CFG" "$RUN" <<'PY'
import os
import sys
import types

# Pre-rendered condition images are mandatory on HPC3.
omninwm = types.ModuleType("omninwm"); omninwm.__path__ = []
utils = types.ModuleType("omninwm.utils"); utils.__path__ = []
hardseam = types.ModuleType("omninwm.utils.pano_hardseam")
def unavailable(*args, **kwargs):
    raise RuntimeError("online pano stitching unavailable")
hardseam.stitch_hardseam = unavailable
omninwm.utils = utils; utils.pano_hardseam = hardseam
sys.modules.update({
    "omninwm": omninwm, "omninwm.utils": utils,
    "omninwm.utils.pano_hardseam": hardseam,
})

from dwm_pano import evaluate_4step
from dwm_pano.pipelines import ctsd_pano_consistency_distillation as cd

original = cd.PanoCrossviewTemporalConsistencyDistillation._build_auxiliary_model
def auxiliary(self, checkpoint_path, model_load_state_args, requires_grad, role):
    if role == "teacher" and hasattr(self, "model_wrapper"):
        print("eval shim: aliasing teacher to main model", flush=True)
        return self.model, self.model_wrapper
    return original(self, checkpoint_path, model_load_state_args,
                    requires_grad, role)
cd.PanoCrossviewTemporalConsistencyDistillation._build_auxiliary_model = auxiliary

cfg, run = sys.argv[1:]
sys.argv = ["evaluate_4step.py", "-c", cfg, "-r", run,
            "-s", "10000", "--ema"]
evaluate_4step.main()
PY

echo "completed arm=$ARM run=$RUN"
