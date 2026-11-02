"""Flicker of the ablation arms on Dev-78 from flicker_stats.py outputs (2026-10-21). Pure numpy.

Definitions = flicker_analyze.py (PanDWM-baselines-20261019/flicker, FLICKER_20261020.md section 0.1), applied to the
78 Dev-78 clips:
  c = mean absolute grey-level change over the 30 within-window steps of the 32 generated frames (stats win.full);
  ONE recorded reference for every arm: the recorded (top) half of the Dev-149 eval run of v84i4250 seed 42
    (pandwm_runs/eval149_hpc3_v83_20261012/v84i4250, the reference of the Val-150 flicker column);
  r_i = c_gen,i / c_ref,i;  static (dev rule) = clips with c_ref <= the 1/3 quantile of c_ref over the 78 clips;
  flicker = mean of r_i over the static clips (the table column);  also: pooled ratio over all clips, sharpness ratio;
  per arm: per-seed values, mean over seeds; 95% interval = mean +- 1.96 SE, delete-one-scene jackknife of the seed
  mean (Dev-78 has one clip per scene), static membership fixed;  pairs A:B = mean_A - mean_B with its jackknife.
  python3 abl_flicker.py --manifest dev78.json --ref stats_ref.json --arm A=s42.json,s1000.json ... \
      --pair A:B ... --out flicker.json
"""
import argparse
import json

import numpy as np


def load(path):
    return {int(k): v for k, v in json.load(open(path))["clips"].items()}


def arr(st, clips, side, key="win"):
    if key == "win":
        return np.array([st[c][side]["win"]["full"] for c in clips])
    return np.array([st[c][side][key] for c in clips])


def stats_one(g, ref, sets, keep):
    r = g["c"] / ref["c"]
    out = {}
    for k, m in sets.items():
        mm = m & keep
        out[f"flicker_{k}"] = float(r[mm].mean())
        out[f"pooled_{k}"] = float(g["c"][mm].sum() / ref["c"][mm].sum())
    out["sharp"] = float(g["s"][keep].mean() / ref["s"][keep].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arm", action="append", required=True, help="TAG=stats[,stats...]")
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = json.load(open(a.manifest))["clips"]
    clips = [int(c["test_index"]) for c in man]
    scene = np.array([c["scene_token"] for c in man])
    scenes = sorted(set(scene))
    ref_st = load(a.ref)
    ref = {"c": arr(ref_st, clips, "r"), "s": arr(ref_st, clips, "r", "sharp")}
    t = np.quantile(ref["c"], [1 / 3, 2 / 3])
    sets = {"static_dev": ref["c"] <= t[0], "all": np.ones(len(clips), bool)}
    arms = {}
    for spec in a.arm:
        tag, files = spec.split("=", 1)
        arms[tag] = []
        for f in files.split(","):
            st = load(f)
            missing = [c for c in clips if c not in st]
            if missing:
                raise RuntimeError(f"{f}: missing clips {missing[:5]}")
            arms[tag].append({"file": f, "c": arr(st, clips, "g"), "s": arr(st, clips, "g", "sharp"),
                              "own_ref_c": arr(st, clips, "r")})
    full = np.ones(len(clips), bool)

    def seed_mean(tag, keep, key):
        return float(np.mean([stats_one(g, ref, sets, keep)[key] for g in arms[tag]]))

    def jack(fn):
        v = np.array([fn(scene != sc) for sc in scenes])
        n = len(v)
        return float(np.sqrt((n - 1) / n * ((v - v.mean()) ** 2).sum()))

    keys = ["flicker_static_dev", "flicker_all", "pooled_static_dev", "pooled_all", "sharp"]
    out = {"clips": len(clips), "scenes": len(scenes), "static_clips": int(sets["static_dev"].sum()),
           "tertiles_ref": [float(x) for x in t], "ref": a.ref, "arms": {}, "pairs": {}}
    for tag, gs in arms.items():
        row = {"files": [g["file"] for g in gs]}
        for k in keys:
            per = [stats_one(g, ref, sets, full)[k] for g in gs]
            m = float(np.mean(per))
            se = jack(lambda keep, k=k: seed_mean(tag, keep, k))
            row[k] = {"per_seed": per, "mean": m, "sd": float(np.std(per, ddof=1)) if len(per) > 1 else None,
                      "se_jackknife": se, "ci95": [m - 1.96 * se, m + 1.96 * se]}
        # check: own recorded half instead of the fixed reference (should differ by codec noise only)
        own = [float((g["c"] / g["own_ref_c"])[sets["static_dev"]].mean()) for g in gs]
        row["flicker_static_dev_own_ref"] = own
        out["arms"][tag] = row
    for spec in a.pair:
        x, y = spec.split(":")
        row = {}
        for k in keys:
            d = seed_mean(x, full, k) - seed_mean(y, full, k)
            se = jack(lambda keep, k=k: seed_mean(x, keep, k) - seed_mean(y, keep, k))
            row[k] = {"diff": d, "se_jackknife": se, "ci95": [d - 1.96 * se, d + 1.96 * se]}
        out["pairs"][spec] = row
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"static (dev rule) clips {out['static_clips']} of {len(clips)}; ref tertiles {t[0]:.3f} / {t[1]:.3f}")
    for tag, row in out["arms"].items():
        r = row["flicker_static_dev"]
        print(f"[flicker] {tag:<12} static {r['mean']:.4f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] seeds "
              f"{' / '.join(f'{v:.4f}' for v in r['per_seed'])}  all(pooled) {row['pooled_all']['mean']:.4f}  "
              f"sharp {row['sharp']['mean']:.4f}  own-ref static {' / '.join(f'{v:.4f}' for v in row['flicker_static_dev_own_ref'])}")
    for spec, row in out["pairs"].items():
        r = row["flicker_static_dev"]
        print(f"[flicker pair] {spec:<20} static {r['diff']:+.4f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]")
    print("ABL_FLICKER_DONE", flush=True)


if __name__ == "__main__":
    main()
