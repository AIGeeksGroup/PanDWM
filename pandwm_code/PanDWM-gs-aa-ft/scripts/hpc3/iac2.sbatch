#!/bin/bash
#SBATCH --job-name=pandwm_iac2
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=8:00:00
#SBATCH --output=%x_%j.log
# Re-scores the SAME deterministic baseline generation with four instance
# descriptors at once. The "patch" arm must reproduce job 589543 exactly
# (FID 14.596 / FVD 79.272 / iac_ratio 0.928); if it does not, the run is
# void and nothing else on the line is readable.
#   sbatch --export=ALL,ITEMS=149 iac2.sbatch
set -eu
ARM=${ARM:-stochastic}; ITEMS=${ITEMS:-149}; SEED=${SEED:-0}; EQ=${EQ:-0}; PLU=${PLU:-0}
H=/path/to/pandwm_root
RUN=$H/pandwm_runs/iac2_eq${EQ}plu${PLU}_${ARM}_items${ITEMS}_seed${SEED}
CFG_SRC=$H/pandwm_code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_hpc3.json

echo "=== $(hostname) ==="; nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
source /share/anaconda3/bin/activate pytorch271
export PYTHONPATH=$H/pandwm_code/PanDWM/src:$H/pandwm_code/TATS_fvd
export PANDWM_4STEP_SAMPLER=$ARM OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 500))
export TORCH_HOME=$H/.cache/torch
mkdir -p "$RUN/checkpoints" "$RUN/eval_configs"
ln -sf "$H/pandwm_data/cd_10000_ema.pth" "$RUN/checkpoints/10000_ema.pth"

python - "$CFG_SRC" "$RUN/eval_configs/cfg.json" "$SEED" "$ITEMS" "$EQ" "$PLU" <<'PY'
import json, sys
src, dst, seed, items, eq, plu = sys.argv[1:]
c = json.load(open(src))
c["generator_seed"] = int(seed)
c["pipeline"]["inference_config"]["evaluation_item_count"] = int(items)
c["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]

hits = []
def walk(o):
    if isinstance(o, dict):
        if "perspective_modeling_type" in o:
            o["pano_equirect_rays"] = (eq == "1")
            o["pano_plucker_rays"] = (plu == "1")
            hits.append(o["perspective_modeling_type"])
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(c)
assert len(hits) == 1, f"expected exactly one DiT block, got {hits}"
assert hits[0] == "explicit", f"ray encoder inactive: {hits[0]}"

mt = []
def find_metrics(o):
    if isinstance(o, dict):
        if "metrics" in o and isinstance(o["metrics"], dict):
            mt.append(o["metrics"])
        for v in o.values():
            find_metrics(v)
    elif isinstance(o, list):
        for v in o:
            find_metrics(v)
find_metrics(c)
assert len(mt) == 1, f"expected one metrics block, got {len(mt)}"

PKL = ("/path/to/pandwm_root/pandwm_data/"
       "nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl")
CLS = "dwm_pano.metrics.instance_consistency.InstanceAppearanceConsistency"
# Four arms on identical frames and identical boxes. Only the embedding
# differs, except the last one which also raises the box-size floor -- a
# low-order descriptor is worst exactly on tiny crops.
arms = {
    "iacpatch": {"descriptor": "patch", "patch_size": 32, "min_pixels": 8},
    "iacincep": {"descriptor": "inception", "min_pixels": 8},
    "iacvgg":   {"descriptor": "vgg", "min_pixels": 8},
    "iacinbig": {"descriptor": "inception", "min_pixels": 32},
}
for name, extra in arms.items():
    mt[0][name] = dict(
        {"_class_name": CLS, "pkl_path": PKL, "backbone_device": "cuda"},
        **extra)

json.dump(c, open(dst, "w"), ensure_ascii=False, indent=2)
print("config:", dst, "| mesh:", c["global_state"]["device_mesh"]["mesh_shape"],
      "| EQ:", eq, "| PLU:", plu, "| metrics:", sorted(mt[0]))
PY

echo "=== four IAC descriptors | items=$ITEMS | seed=$SEED | EQ=$EQ PLU=$PLU ==="
cd "$H/pandwm_code/PanDWM"
$H/pandwm_venv/bin/python - "$RUN/eval_configs/cfg.json" "$RUN" <<'PY'
import sys, types, os
import torch.distributed.device_mesh as _dm
_orig_init_mesh = _dm.init_device_mesh
def _single_rank_mesh(device_type, mesh_shape, *a, **k):
    if os.environ.get("WORLD_SIZE") == "1" and tuple(mesh_shape) == (1, 4):
        mesh_shape = (1, 1)
    return _orig_init_mesh(device_type, mesh_shape, *a, **k)
_dm.init_device_mesh = _single_rank_mesh

omninwm = types.ModuleType("omninwm"); omninwm.__path__ = []
u = types.ModuleType("omninwm.utils"); u.__path__ = []
p = types.ModuleType("omninwm.utils.pano_hardseam")
def unavailable(*a, **k):
    raise RuntimeError("online pano stitching unavailable; pre-rendered PNG required")
p.stitch_hardseam = unavailable; omninwm.utils = u; u.pano_hardseam = p
sys.modules.update({"omninwm": omninwm, "omninwm.utils": u,
                    "omninwm.utils.pano_hardseam": p})

from dwm_pano import evaluate_4step
from dwm_pano.pipelines import ctsd_pano_consistency_distillation as cd
_orig = cd.PanoCrossviewTemporalConsistencyDistillation._build_auxiliary_model
def _aux(self, checkpoint_path, model_load_state_args, requires_grad, role):
    if role == "teacher" and hasattr(self, "model_wrapper"):
        print("eval shim: aliasing consistency teacher to the main model", flush=True)
        return self.model, self.model_wrapper
    return _orig(self, checkpoint_path, model_load_state_args, requires_grad, role)
cd.PanoCrossviewTemporalConsistencyDistillation._build_auxiliary_model = _aux

cfg, run = sys.argv[1:]
sys.argv = ["evaluate_4step.py", "-c", cfg, "-r", run, "-s", "10000", "--ema"]
evaluate_4step.main()
PY
echo "=== done: four IAC descriptors ==="
