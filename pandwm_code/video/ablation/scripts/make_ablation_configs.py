"""Write the ablation configs from the final run's config and print every key that differs.

Variants (all share A_ctrl's re-anchored schedules, see CONTROL below):
  A_ctrl    full recipe: the temporal-head and motion-match schedules start at step 0 with no ramp.
  B_noadv   A without the adversarial term (gan weights 0).
  C_rechist A with PanoContextForcedDMDv8_4R (src/dwm_pano/student/pipeline_v8_4r.py) and recorded_history true.
  D_notemp  A with disc_temporal false (spatial heads only).

  python3 make_ablation_configs.py <final run config> <out dir>
"""
import json
import sys
from pathlib import Path

NEW_KEYS = ("recorded_history",)          # knob of the C subclass; absent from the base config
CONTROL = {"temporal_adv_from": 0, "temporal_adv_ramp": 0, "motion_match_from": 0, "motion_match_ramp": 0}
VARIANTS = {
    "A_ctrl": {},
    "B_noadv": {"gan_generator_weight": 0.0, "gan_discriminator_weight": 0.0},
    "C_rechist": {"_class_name": "dwm_pano.student.pipeline_v8_4r.PanoContextForcedDMDv8_4R",
                  "recorded_history": True},
    "D_notemp": {"disc_temporal": False},
}


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def main():
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    base = json.loads(src.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    fb = flat(base)
    for name, extra in VARIANTS.items():
        cfg = json.loads(json.dumps(base))
        for key, value in {**CONTROL, **extra}.items():
            if key not in cfg["pipeline"] and key not in NEW_KEYS:
                raise KeyError(f"{key} not in the I config pipeline section")
            cfg["pipeline"][key] = value
        path = out_dir / f"abl_{name}_v8_4_hpc3_8gpu.json"
        path.write_text(json.dumps(cfg, indent=4, ensure_ascii=False) + "\n")
        fc = flat(cfg)
        diff = {k: (fb.get(k, "<absent>"), fc.get(k, "<absent>")) for k in sorted(set(fb) | set(fc))
                if fb.get(k, "<absent>") != fc.get(k, "<absent>")}
        print(f"{path.name}: {len(diff)} keys differ from {src.name}")
        for k, (a, b) in diff.items():
            print(f"    {k}: {json.dumps(a)} -> {json.dumps(b)}")


if __name__ == "__main__":
    main()
