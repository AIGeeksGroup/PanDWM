"""C2 config for HPC3: the 4D decoder retrained with Depth Anything 3 pseudo-depth (2026-10-20, report section 41).

Base: configs/gs4dd_hpc3_F1.json (coverage recipe + metre L1 0.002 / 0.002 + log-depth L1 0.1 / 0.1, far cut 99 m).
Changes, nothing else:
  * class dwm_pano_4d.pipeline_da3.PanoGaussianDecoder4DDa3Pipeline: the depth labels of every batch come from DA3
    run on the batch's own frames (ground scale), not from label files; gs4d_depth.depth_optional = false;
  * warm start from the coverage run step 3000 (init/gs4d_cov_3000.pth), the decoder of review packages v2 / v3;
  * training data = the 695-scene set only (full train minus the 5 held-out labelled scenes), no p2 copies.
  python3 make_c2_config.py   -> configs/gs4dd_hpc3_C2.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DR = "/path/to/pandwm_root"
RUNS = f"{DR}/pandwm_runs/gs4d_hpc3_20261017"
DA3 = f"{DR}/pandwm_code/third_party/depth_anything_3"


def main():
    c = json.load(open(HERE / "configs/gs4dd_hpc3_F1.json"))
    p = c["pipeline"]
    assert p["_class_name"] == "dwm_pano_4d.pipeline_depth.PanoGaussianDecoder4DDepthPipeline"
    assert p["gs4d_depth"] == {"weight_log_range": 0.1, "weight_log_render": 0.1, "far_cut_m": 99.0,
                               "depth_optional": True}
    assert p["training_config"]["weight_render_depth"] == 0.002 and p["training_config"]["weight_range_l1"] == 0.002
    p["_class_name"] = "dwm_pano_4d.pipeline_da3.PanoGaussianDecoder4DDa3Pipeline"
    p["gs4d_depth"]["depth_optional"] = False
    p["gs4d_da3"] = {"repo_src": f"{DA3}/repo/src", "anyview_dir": f"{DA3}/weights/DA3-LARGE-1.1",
                     "metric_dir": f"{DA3}/weights/DA3METRIC-LARGE", "scale": "ground", "sky_threshold": 0.3,
                     "min_range_m": 0.5, "far_cut_m": 99.0, "sky_fill_m": 100.0, "crop_hw": [392, 784]}
    p["initial_decoder_checkpoint_path"] = f"{RUNS}/init/gs4d_cov_3000.pth"
    full = c["training_dataset"]["datasets"][0]
    assert full["base_dataset"]["pkl_path"].endswith("nuscenes_interp_12Hz_infos_train_minus_holdout5.pkl")
    c["training_dataset"] = full
    c.setdefault("informations", {})["c2_da3_20261020"] = (
        "from coverage step 3000 (review v2/v3 decoder); data = full train minus holdout5 (695 scenes); depth labels "
        "of every batch from Depth Anything 3 on the batch's frames (ground scale, sky / >= 99 m filled at 100 m); "
        "metre L1 0.002/0.002, log L1 0.1/0.1 as F1")
    out = HERE / "configs/gs4dd_hpc3_C2.json"
    json.dump(c, open(out, "w"), indent=2)
    print("wrote", out.name)


if __name__ == "__main__":
    main()
