"""Build the depth-supervised 4D decoder config from the depth pilot config.

Starts from the pilot config (coverage recipe, metre L1 0.002 / 0.002, log-depth L1 0.1 / 0.1, class
dwm_pano_4d.pipeline_depth.PanoGaussianDecoder4DDepthPipeline) and changes:
  * warm start from the pilot decoder checkpoint (init/gs4dd_D1_1000.pth);
  * gs4d_depth.depth_optional = true: depth terms only on batches whose frames all have a label;
  * training data = ConcatDataset of the full training set minus the 5 held-out labelled scenes (695 scenes) and the
    labelled p2 subset repeated P2_REPEAT times, so a sizeable share of the batches carries depth.
Usage: python3 make_formal_config.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = "/path/to/pandwm_root/pandwm_runs/gs4d_hpc3"
P2 = "/path/to/pandwm_root/pandwm_gs_aa_experiment/inputs/nuscenes_interp_12Hz_infos_train_p2.pkl"
P2_REPEAT = 4


def main():
    c = json.load(open(HERE / "configs/gs4dd_hpc3_D1.json"))
    p = c["pipeline"]
    assert p["_class_name"] == "dwm_pano_4d.pipeline_depth.PanoGaussianDecoder4DDepthPipeline"
    assert p["gs4d_depth"] == {"weight_log_range": 0.1, "weight_log_render": 0.1, "far_cut_m": 99.0}
    p["initial_decoder_checkpoint_path"] = f"{RUNS}/init/gs4dd_D1_1000.pth"
    p["gs4d_depth"]["depth_optional"] = True
    adapter = c["training_dataset"]
    assert adapter["base_dataset"]["pkl_path"] == P2
    full = json.loads(json.dumps(adapter))
    full["base_dataset"]["pkl_path"] = f"{RUNS}/data/nuscenes_interp_12Hz_infos_train_minus_holdout5.pkl"
    c["training_dataset"] = {"_class_name": "torch.utils.data.ConcatDataset",
                             "datasets": [full] + [json.loads(json.dumps(adapter)) for _ in range(P2_REPEAT)]}
    c.setdefault("informations", {})["formal_F1"] = (
        f"from depth pilot D1 step 1000; data = full train minus holdout5 (695 scenes) + p2 x {P2_REPEAT}; "
        "depth terms (metre L1 0.002/0.002, log L1 0.1/0.1, far cut 99 m) only on fully labelled batches")
    out = HERE / "configs/gs4dd_hpc3_F1.json"
    json.dump(c, open(out, "w"), indent=2)
    print("wrote", out.name)


if __name__ == "__main__":
    main()
