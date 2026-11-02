"""Write the ablation configs (2026-10-21) from the final run's config and print every key that differs.

Source: PanDWM-cfdmd-A-20261008/configs/cfdmd_I_gan_v8_4_hpc3_8gpu.json (identical to
pandwm_runs/cfdmd_20261013_I/gan/log/config_used_job_714919.json, the run that produced the locked v84i4250 EMA).

  A_ctrl    full recipe from the consistency-distilled student: the I config with the two schedules that were anchored
            to the staged run's history (temporal heads from step 1300, ramp 150; motion match from step 1600,
            ramp 100) re-anchored at step 0 with no ramp, so both terms are at the strength they had in the final stage
            from the start.  Everything else is the I config as is (gan_warmup [100, 200] and ema_start_step 200 are
            anchored at step 0 of the chain, i.e. at the start from the student, and are kept).
  B_noadv   A + gan_generator_weight 0, gan_discriminator_weight 0 (use_gan False: no heads, no head training, no
            adversarial term; the critic step is flow matching only, as in A).
  C_rechist A + pipeline class PanoContextForcedDMDv8_4R (src/dwm_pano_cfdmd_abl/pipeline_v8_4r.py, = v8.4 with the
            knob off) + recorded_history true (option (b), user decision 2026-10-21, ABLATION_20261021.md section 1.8).
            The earlier option (a) (context_forcing_prob 0) is not a single switch and was not used.
  D_notemp  A + disc_temporal false (spatial heads only).

  python3 make_ablation_configs.py <I config> <out dir>
"""
import json
import sys
from pathlib import Path

NEW_KEYS = ("recorded_history",)          # knob of the C subclass; absent from the I config
CONTROL = {"temporal_adv_from": 0, "temporal_adv_ramp": 0, "motion_match_from": 0, "motion_match_ramp": 0}
VARIANTS = {
    "A_ctrl": {},
    "B_noadv": {"gan_generator_weight": 0.0, "gan_discriminator_weight": 0.0},
    "C_rechist": {"_class_name": "dwm_pano_cfdmd_abl.pipeline_v8_4r.PanoContextForcedDMDv8_4R",
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
