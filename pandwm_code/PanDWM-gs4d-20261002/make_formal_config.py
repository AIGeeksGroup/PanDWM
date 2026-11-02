"""Formal depth-supervised 4D decoder config for HPC3 (2026-10-18, report section 38; the user chose option A).

Base: configs/gs4dd_hpc3_D1.json (the depth pilot arm D1: coverage recipe + metre L1 0.002 / 0.002 + log-depth L1
0.1 / 0.1, class dwm_pano_4d.pipeline_depth.PanoGaussianDecoder4DDepthPipeline).  Changes, nothing else:
  * warm start from the pilot's D1 step 1000 (init/gs4dd_D1_1000.pth, md5 7ed1d4f8);
  * gs4d_depth.depth_optional = true: depth terms only on batches whose frames all have a label;
  * training data = ConcatDataset of
      - the full 700-scene training set minus the 5 held-out labelled scenes (data/..._train_minus_holdout5.pkl,
        695 scenes; the same rgb_dir as the coverage run), and
      - the labelled p2 subset (80 scenes) repeated P2_REPEAT times,
    so roughly 40 % of the batches carry depth (the 81 labelled scenes inside the full set + the p2 copies).
    The 5 held-out scenes (full-set indices 16, 23, 43, 106, 108; data/..._train_holdout5.pkl) are seen by neither.
  python3 make_formal_config.py   -> configs/gs4dd_hpc3_F1.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = "/path/to/pandwm_root/pandwm_runs/gs4d_hpc3_20261017"
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
    c.setdefault("informations", {})["formal_F1_20261018"] = (
        f"from depth pilot D1 step 1000; data = full train minus holdout5 (695 scenes) + p2 x {P2_REPEAT}; "
        "depth terms (metre L1 0.002/0.002, log L1 0.1/0.1, far cut 99 m) only on fully labelled batches")
    out = HERE / "configs/gs4dd_hpc3_F1.json"
    json.dump(c, open(out, "w"), indent=2)
    print("wrote", out.name)


if __name__ == "__main__":
    main()
