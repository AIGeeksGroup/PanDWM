"""Depth-pilot configs for HPC3 (2026-10-17, report section 34).

Base: configs/gs4d_hpc2.json, the recipe of the coverage run whose step 3000 is the decoder of the 149-scene review
renders (black background + coverage regulariser 0.1).  Changes, nothing else:
  * every /path/to/pandwm_root path -> /path/to/pandwm_root (HPC3 has the same layout);
  * class dwm_pano_4d.pipeline_depth.PanoGaussianDecoder4DDepthPipeline (= the 4D pipeline when the log terms are 0);
  * warm start from that decoder, init/gs4d_cov_3000.pth (md5 84344b6b);
  * training data = the G2 depth subset nuscenes_interp_12Hz_infos_train_p2.pkl (80 scenes, 18638 frames, every
    frame has a depth label in pandwm_data/train/depth);
  * per arm: C  no depth term (as the coverage run);
             D1 metre L1 0.002 / 0.002 (G2's) + log-depth L1 0.1 / 0.1 (range / render);
             D2 metre L1 0.002 / 0.002 + log-depth L1 0.5 / 0.5.
  python3 make_depth_pilot_configs.py   -> configs/gs4dd_hpc3_{C,D1,D2}.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD, NEW = "/path/to/pandwm_root", "/path/to/pandwm_root"
ARMS = {"C": (0.0, 0.0, 0.0, 0.0), "D1": (0.002, 0.002, 0.1, 0.1), "D2": (0.002, 0.002, 0.5, 0.5)}


def rewrite(o):
    if isinstance(o, dict):
        return {k: rewrite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rewrite(v) for v in o]
    if isinstance(o, str):
        return o.replace(OLD, NEW)
    return o


def main():
    base = rewrite(json.load(open(HERE / "configs/gs4d_hpc2.json")))
    assert base["pipeline"]["gs4d"].get("weight_coverage") == 0.1, "base must be the coverage recipe"
    for arm, (w_depth, w_range, w_log_range, w_log_render) in ARMS.items():
        c = json.loads(json.dumps(base))
        p = c["pipeline"]
        p["_class_name"] = "dwm_pano_4d.pipeline_depth.PanoGaussianDecoder4DDepthPipeline"
        p["initial_decoder_checkpoint_path"] = f"{NEW}/pandwm_runs/gs4d_hpc3_20261017/init/gs4d_cov_3000.pth"
        p["training_config"]["range_dir"] = f"{NEW}/pandwm_data/train/depth"
        p["training_config"]["weight_render_depth"] = w_depth
        p["training_config"]["weight_range_l1"] = w_range
        p["gs4d_depth"] = {"weight_log_range": w_log_range, "weight_log_render": w_log_render, "far_cut_m": 99.0}
        c["training_dataset"]["base_dataset"]["pkl_path"] = \
            f"{NEW}/pandwm_gs_aa_experiment/inputs/nuscenes_interp_12Hz_infos_train_p2.pkl"
        c.setdefault("informations", {})["depth_pilot_20261017"] = (
            f"arm {arm}: from 4D coverage run step 3000; G2 depth subset p2 (80 scenes, every frame labelled); "
            f"metre L1 {w_depth}/{w_range}, log-depth L1 {w_log_range}/{w_log_render} (range/render), far cut 99 m")
        out = HERE / f"configs/gs4dd_hpc3_{arm}.json"
        json.dump(c, open(out, "w"), indent=2)
        print("wrote", out.name)


if __name__ == "__main__":
    main()
