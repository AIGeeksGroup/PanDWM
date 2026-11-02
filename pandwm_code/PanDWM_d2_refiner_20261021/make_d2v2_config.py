"""Config of the D2 v2 refiner (2026-10-31, D2V2_PLAN_20261031.md section 2).

configs/d2_full_hpc3_8gpu.json (the section 43.3 continuation: render-init samples 0.5, render dropout 0.1, lr 1e-5,
8-GPU FSDP) with
  init         D2 v1 step 4000 (runs/pilot/checkpoints/4000.pth, render branch already trained), strict load
  render_root  the round-trip pairs of PanDWM-gs-aa-ft dwm_pano/d2v2_make_pairs.py (G3b decoder)
    python3 make_d2v2_config.py --out configs/d2v2_hpc3_8gpu.json
"""

import argparse
import json
from pathlib import Path

DR = "/path/to/pandwm_root"
INIT = f"{DR}/pandwm_runs/d2_refiner_20261021/runs/pilot/checkpoints/4000.pth"
ROOT = f"{DR}/pandwm_runs/d2v2_20261031/train_pairs"

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
c["informations"] = dict(c.get("informations", {}), experiment="D2 v2 refiner (D2V2_PLAN_20261031.md)", init=INIT,
                         render_root=ROOT, objective=c["informations"]["objective"].replace(
                             "degraded C2b GS render", "round-trip G3b GS render (d2v2_make_pairs)"))
Path(args.out).write_text(json.dumps(c, indent=2) + "\n")
print(args.out)
