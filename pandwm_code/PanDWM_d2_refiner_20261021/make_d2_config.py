"""Configs for the D2 refiner (2026-10-21, D2_PLAN_20261021.md Task 4).

Base: the HPC3 teacher v3 control config (PanDWM-teacher-v3-20261015/configs/teacher_v3_control.json: Stage-1
causal teacher stage1_causal_20000.pth, 8-GPU FSDP, full fine-tuning, lr 1e-5, the 700-scene train set).
Changes:
  pipeline   dwm_pano_d2.pipeline.PanoD2RefinerSD (robust_config -> d2_config), strict=False load (the only
             missing keys must be render_embed.*), generation_task_ratio 0.3 (no-history windows),
             3D-box / HD-map condition ratio 0.5 (inference drops both)
  model      dwm_pano_d2.model.D2RenderDiT, 17 render channels, scale 10
  data       D2RenderDataset over the train set in 35-frame clips, one start every 2 s ([[10, 2.0]])
    python3 make_d2_config.py --out configs/d2_pilot_hpc3_8gpu.json
    python3 make_d2_config.py --out configs/d2_smoke_hpc3_2gpu.json --gpus 2
    python3 make_d2_config.py --out configs/d2_full_hpc3_8gpu.json --min-items 300 --render-init-prob 0.5
"""

import argparse
import json
from pathlib import Path

DR = "/path/to/pandwm_root"
BASE = Path(__file__).resolve().parents[1] / "PanDWM-teacher-v3-20261015/configs/teacher_v3_control.json"
RENDER_ROOT = f"{DR}/pandwm_runs/d2_refiner_20261021/train_pairs"
MANIFEST = f"{DR}/pandwm_code/PanDWM_long_horizon_20260928/inputs/split_195_scene_manifest.json"
TEACHER = f"{DR}/pandwm_data/stage1_causal_20000.pth"


def build(base, gpus, render_root, lr, seed, min_items, render_init_prob=0.0, render_init_min_sigma=0.5):
    c = json.loads(json.dumps(base))
    c["global_state"]["device_mesh"]["mesh_shape"] = [1, int(gpus)]
    c["generator_seed"] = int(seed)
    c["optimizer"]["lr"] = float(lr)
    pc = c["pipeline"]
    pc["_class_name"] = "dwm_pano_d2.pipeline.PanoD2RefinerSD"
    pc.pop("robust_config", None)
    pc["d2_config"] = {"render_dropout": 0.1, "max_start": 16, "forbidden_scene_manifest": MANIFEST,
                       "render_init_prob": float(render_init_prob), "render_init_min_sigma": float(render_init_min_sigma)}
    pc["model_checkpoint_path"] = TEACHER
    pc["model_load_state_args"] = {"strict": False}
    pc["common_config"]["print_load_state_info"] = True
    m = pc["model"]
    m["_class_name"] = "dwm_pano_d2.model.D2RenderDiT"
    m["render_condition_channels"] = 17
    m["render_condition_scale"] = 10.0
    tc = pc["training_config"]
    tc.update({"generation_task_ratio": 0.3, "3dbox_condition_ratio": 0.5, "hdmap_condition_ratio": 0.5,
               "reference_frame_count": 3, "all_reference_visible_ratio": 1.0, "reference_visible_rate": 1.0,
               "disable_reference_frame_loss": True, "temporal_attention_mask": "causal"})
    tc.pop("history_noise", None)
    td = c["training_dataset"]
    if td["_class_name"] != "dwm.datasets.common.DatasetAdapter":
        raise ValueError("unexpected training_dataset class in the base config")
    bd = td["base_dataset"]
    if bd["split"] != "train" or "train" not in bd["pkl_path"]:
        raise ValueError("the base training dataset must be the train split")
    bd["sequence_length"] = 35
    bd["fps_stride_tuples"] = [[10, 2.0]]
    c["training_dataset"] = {"_class_name": "dwm_pano_d2.data.D2RenderDataset", "base": td,
                             "render_root": render_root, "min_items": int(min_items)}
    c["informations"] = {
        "experiment": "D2 refiner (PanDWM-d2-refiner-20261021/D2_PLAN_20261021.md)",
        "init": TEACHER, "lr": float(lr), "gpus": int(gpus),
        "objective": "Stage-1 flow matching on one 19-frame window (start 0..16) of a 35-frame train clip, "
                     "conditioned on the clip's degraded C2b GS render (16 latent ch + coverage)",
        "render_root": render_root}
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--render-root", default=RENDER_ROOT)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-items", type=int, default=1)
    ap.add_argument("--render-init-prob", type=float, default=0.0, help="section 43.3 continuation: 0.5")
    ap.add_argument("--render-init-min-sigma", type=float, default=0.5)
    args = ap.parse_args()
    c = build(json.loads(Path(args.base).read_text()), args.gpus, args.render_root, args.lr, args.seed,
              args.min_items, args.render_init_prob, args.render_init_min_sigma)
    Path(args.out).write_text(json.dumps(c, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
