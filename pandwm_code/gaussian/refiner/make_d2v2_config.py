"""Config of the D2 v2 refiner.

Takes the full D2 config written by make_d2_config.py (render-init samples 0.5, render dropout 0.1, lr 1e-5,
8-GPU FSDP) and changes
  init         the D2 v1 checkpoint (render branch already trained), strict load
  render_root  the round-trip pairs written by src/dwm_pano/d2v2_make_pairs.py

    python3 make_d2v2_config.py --out configs/d2v2_8gpu.json
"""

import argparse
import json
from pathlib import Path

DR = "/path/to/pandwm_root"
INIT = f"{DR}/pandwm_runs/d2_refiner/runs/pilot/checkpoints/4000.pth"
ROOT = f"{DR}/pandwm_runs/d2v2/train_pairs"

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--min-items", type=int, default=1000)
args = p.parse_args()
here = Path(__file__).resolve().parent
c = json.loads((here / "configs/d2_full_hpc3_8gpu.json").read_text())
c["pipeline"]["model_checkpoint_path"] = INIT
c["pipeline"]["model_load_state_args"] = {"strict": True}
c["training_dataset"]["render_root"] = ROOT
c["training_dataset"]["min_items"] = int(args.min_items)
c["informations"] = dict(c.get("informations", {}), experiment="render refiner, round-trip renders", init=INIT,
                         render_root=ROOT, objective=c["informations"]["objective"].replace(
                             "degraded GS render", "round-trip GS render"))
Path(args.out).write_text(json.dumps(c, indent=2) + "\n")
print(args.out)
