"""Write the fine-tune configs that train the 4D decoder on the test-time geometry.

All variants start from the same base config (DA3 pipeline, camera-time poses, mmdet3d v0.x boxes, lr 2e-5,
frame_stride [2, 3, 4], LOO 0.5) and its step-2000 checkpoint, with the pipeline class switched to pipeline_relabel
(every frame's Gaussians are slid along their rays to the step's jointly reconstructed DA3 ranges before any loss):
  G0  = bound 0, offsets kept (the test-time "scale" relabel)
  G1  = G0 + no_offset (centres exactly on the DA3 ray points)
  G2  = G1 + perceptual weights 0.5 on own / composite / LOO
  G3a = G0 + G2's perceptual weights (free offsets)
  G3b = in-plane offsets bounded to 1 decoder pixel + G2's perceptual weights
Usage: python3 make_ft3_configs.py [out_dir]
"""
import copy
import json
import sys
from pathlib import Path

H = Path.home()
R1 = H / "pandwm_code/PanDWM-gs-ft2/configs/gs4dd_hpc2_ft2_R1.json"
INIT = H / "pandwm_runs/gs_ft2/R1_from_hpc3/checkpoints/2000.pth"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else H / "pandwm_code/gaussian/finetune/configs"
base = json.loads(R1.read_text())
assert base["pipeline"]["_class_name"] == "dwm_pano.gs4d.pipeline_da3.PanoGaussianDecoder4DDa3Pipeline"
assert base["training_dataset"]["base_dataset"]["pose_source"] == "camera_time"
assert INIT.is_file(), INIT
OUT.mkdir(parents=True, exist_ok=True)


def arm(note, relabel, gs4d=None, depth=None):
    cfg = copy.deepcopy(base)
    p = cfg["pipeline"]
    p["_class_name"] = "dwm_pano.gs4d.pipeline_relabel.PanoGaussianDecoder4DRelabelPipeline"
    p["initial_decoder_checkpoint_path"] = str(INIT)
    p["gs4d_relabel"] = relabel
    p["gs4d"].update(gs4d or {})
    p["gs4d_depth"].update(depth or {})
    cfg["informations"] = {"note": note}
    return cfg


ARMS = {
    "G0": arm("round-3 G0: R1 step 2000 + train on the test-time DA3 relabel (bound 0, offsets kept)",
              {"bound": 0.0}),
    "G1": arm("round-3 G1: G0 + no xyz offset (centres on the DA3 ray points)",
              {"bound": 0.0, "no_offset": True}),
    "G2": arm("round-3 G2: G1 + LPIPS 0.5 on own / composite / LOO",
              {"bound": 0.0, "no_offset": True, "loo_perceptual": 0.5},
              gs4d={"weight_composite_perceptual": 0.5}),
}
# G3a = G0 + G2's LPIPS weights (free offsets); G3b = in-plane offsets bounded to 1 px
# (da3_depth.plane_offsets) + G2's LPIPS weights.
ARMS["G3a"] = arm("round-3 G3a: G0 (free offsets) + LPIPS 0.5 on own / composite / LOO",
                  {"bound": 0.0, "loo_perceptual": 0.5}, gs4d={"weight_composite_perceptual": 0.5})
ARMS["G3b"] = arm("round-3 G3b: in-plane offsets bounded to 1 decoder pixel + LPIPS 0.5 on own / composite / LOO",
                  {"bound": 0.0, "plane_offset_px": 1.0, "loo_perceptual": 0.5},
                  gs4d={"weight_composite_perceptual": 0.5})
for name in ("G2", "G3a", "G3b"):
    ARMS[name]["pipeline"]["training_config"]["weight_perceptual"] = 0.5

for name, cfg in ARMS.items():
    path = OUT / f"gs4dd_hpc2_ft3_{name}.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(path)
