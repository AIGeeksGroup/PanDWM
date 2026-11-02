"""FID / FVD of 35-frame clip sets from clip_feats.py features (2026-10-18; Test-S main table, ladder, setting G).

Metrics (the frozen-protocol instruments, computed from saved features; with all clips they equal
compute_eval149_metrics.py up to float rounding):
  FID   = torchmetrics Frechet distance of Inception features, all 32 generated frames of every clip;
  FID-1, FID-2 = the same on window 1 (frames 3-18) / window 2 (19-34) only (point values);
  FVD-1, FVD-2 = PanoPatchFrechetVideoDistance on window 1 / window 2 (6 camera patches x 400 I3D logits, pooled).
Per arm: the value of every seed, the mean and SD over seeds, and for FID / FVD-1 / FVD-2 a 95% interval of the seed
mean, point +- 1.96 SE from a delete-one-scene jackknife (all clips of a scene left out together; generated and
recorded frames alike).  Pairs A:B: paired difference of the seed means with its jackknife interval, and P(A - B < 0)
from a paired bootstrap over scenes (--boot draws; 0 skips it).
Why not bootstrap intervals (2026-10-18, before any test result): resampling scenes with replacement duplicates clips,
which biases Frechet distances upwards; on Dev-149 the locked model's FID is 10.13 but its 1000-draw percentile
interval was [13.5, 15.6].  The jackknife has no such bias; in a paired difference the bias largely cancels, so the
bootstrap is kept only for the sign probability.
Floor: the scenes are split in random halves (--splits); recorded half 1 vs recorded half 2, and each arm's half-1
clips vs recorded half 2 (and vice versa), the same n; excess = arm - floor.
  python3 tests_analyze.py --real REAL.npz --arm ours=s42.npz,s1000.npz,s2026.npz --arm b0=b0_s42.npz \
      --pair ours:b0 --boot 1000 --out report.json
"""
import argparse
import json

import numpy as np
import torch

from long_analyze import DEV, Stats, frechet

METRICS = ("FID", "FID-1", "FID-2", "FVD-1", "FVD-2")
BOOT = ("FID", "FVD-1", "FVD-2")


def load(path):
    z = np.load(path, allow_pickle=False)
    return {"inc": z["inc"], "i3d": z["i3d"], "test_index": z["test_index"].tolist(),
            "scene": [str(s) for s in z["scene_token"]], "seed": int(z["seed"][0])}


class Rows:
    """Feature rows of one metric on DEV (float64) with the scene of each row; stats(w) weights rows by scene."""

    def __init__(self, x, row_scene):
        self.x = torch.as_tensor(np.ascontiguousarray(x, np.float64), device=DEV)
        self.scene = torch.as_tensor(np.asarray(row_scene), device=DEV)

    def stats(self, scene_weights):
        w = torch.as_tensor(np.asarray(scene_weights, np.float64), device=DEV)[self.scene]
        st = Stats(self.x.shape[1])
        st.n = float(w.sum())
        st.s = w @ self.x
        st.ss = (self.x * w[:, None]).T @ self.x
        return st


def metric_rows(feats, scene_id):
    sc = np.array([scene_id[s] for s in feats["scene"]])
    inc, i3d = feats["inc"], feats["i3d"]
    return {
        "FID": Rows(inc.reshape(-1, 2048), np.repeat(sc, 32)),
        "FID-1": Rows(inc[:, :16].reshape(-1, 2048), np.repeat(sc, 16)),
        "FID-2": Rows(inc[:, 16:].reshape(-1, 2048), np.repeat(sc, 16)),
        "FVD-1": Rows(i3d[:, 0].reshape(-1, 400), np.repeat(sc, 6)),
        "FVD-2": Rows(i3d[:, 1].reshape(-1, 400), np.repeat(sc, 6)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--arm", action="append", required=True, help="TAG=seedfile[,seedfile...]")
    ap.add_argument("--pair", action="append", default=[], help="A:B")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    real = load(a.real)
    order = list(dict.fromkeys(real["scene"]))
    scene_id = {s: i for i, s in enumerate(order)}
    n_sc = len(order)
    real_rows = metric_rows(real, scene_id)
    arms = {}
    for spec in a.arm:
        tag, files = spec.split("=", 1)
        arms[tag] = []
        for f in files.split(","):
            g = load(f)
            if g["test_index"] != real["test_index"] or g["scene"] != real["scene"]:
                raise RuntimeError(f"{f}: clips differ from the real features")
            arms[tag].append({"seed": g["seed"], "file": f, "rows": metric_rows(g, scene_id)})
        seeds = [s["seed"] for s in arms[tag]]
        if len(set(seeds)) != len(seeds):
            raise RuntimeError(f"{tag}: repeated seed {seeds}")
    ones = [1] * n_sc
    out = {"scenes": n_sc, "clips": len(real["test_index"]), "boot_for_sign_only": a.boot, "ci": "jackknife", "arms": {}, "pairs": {}, "floor": {}}

    def value(tag, metric, w_gen, w_real=None):
        r = real_rows[metric].stats(w_gen if w_real is None else w_real)
        return float(np.mean([frechet(s["rows"][metric].stats(w_gen), r) for s in arms[tag]]))

    for tag, runs in arms.items():
        row = {"seeds": [r["seed"] for r in runs], "files": [r["file"] for r in runs]}
        for m in METRICS:
            r = real_rows[m].stats(ones)
            vals = [frechet(run["rows"][m].stats(ones), r) for run in runs]
            row[m] = {"per_seed": vals, "mean": float(np.mean(vals)),
                      "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
        out["arms"][tag] = row
        print(f"[point] {tag} " + " ".join(f"{m}={row[m]['mean']:.3f}" for m in METRICS), flush=True)
    jack = {tag: {m: [] for m in BOOT} for tag in arms}
    for j in range(n_sc):
        w = [1] * n_sc
        w[j] = 0
        for m in BOOT:
            r = real_rows[m].stats(w)
            for tag in arms:
                jack[tag][m].append(float(np.mean([frechet(s["rows"][m].stats(w), r) for s in arms[tag]])))
        if (j + 1) % 20 == 0:
            print(f"[jackknife] {j + 1}/{n_sc}", flush=True)

    def jack_se(values):
        v = np.asarray(values)
        return float(np.sqrt((len(v) - 1) / len(v) * ((v - v.mean()) ** 2).sum()))

    for tag in arms:
        for m in BOOT:
            se = jack_se(jack[tag][m])
            mean = out["arms"][tag][m]["mean"]
            out["arms"][tag][m]["se_jackknife"] = se
            out["arms"][tag][m]["ci95"] = [mean - 1.96 * se, mean + 1.96 * se]
    rng = np.random.default_rng(20261018)
    draws = [np.bincount(rng.integers(0, n_sc, n_sc), minlength=n_sc) for _ in range(a.boot if a.pair else 0)]
    boot = {tag: {m: [] for m in BOOT} for tag in arms}
    for b, w in enumerate(draws):
        for m in BOOT:
            r = real_rows[m].stats(w)
            for tag in arms:
                boot[tag][m].append(float(np.mean([frechet(s["rows"][m].stats(w), r) for s in arms[tag]])))
        if (b + 1) % 100 == 0:
            print(f"[boot] {b + 1}/{len(draws)}", flush=True)
    for spec in a.pair:
        x, y = spec.split(":")
        row = {}
        for m in BOOT:
            diff = out["arms"][x][m]["mean"] - out["arms"][y][m]["mean"]
            se = jack_se(np.array(jack[x][m]) - np.array(jack[y][m]))
            d = np.array(boot[x][m]) - np.array(boot[y][m])
            row[m] = {"diff": diff, "se_jackknife": se, "ci95": [diff - 1.96 * se, diff + 1.96 * se],
                      "p_lt_0_boot": float((d < 0).mean()) if len(d) else None}
        for m in ("FID-1", "FID-2"):
            row[m] = {"diff": out["arms"][x][m]["mean"] - out["arms"][y][m]["mean"]}
        out["pairs"][spec] = row
    rng = np.random.default_rng(7)
    floor = {m: [] for m in METRICS}
    half = {tag: {m: [] for m in METRICS} for tag in arms}
    for _ in range(a.splits):
        perm = rng.permutation(n_sc)
        w1 = np.zeros(n_sc)
        w1[perm[:n_sc // 2]] = 1
        w2 = 1 - w1
        for m in METRICS:
            r1, r2 = real_rows[m].stats(w1), real_rows[m].stats(w2)
            floor[m].append(frechet(r1, r2))
            for tag, runs in arms.items():
                half[tag][m].append(float(np.mean([
                    0.5 * (frechet(s["rows"][m].stats(w1), r2) + frechet(s["rows"][m].stats(w2), r1))
                    for s in runs])))
    out["floor"] = {m: float(np.mean(floor[m])) for m in METRICS}
    out["floor_note"] = (f"recorded half vs recorded half, mean over {a.splits} random scene splits; arms: generated "
                         "half vs the other recorded half, same splits and n; excess = arm - floor")
    for tag in arms:
        out["arms"][tag]["half_n"] = {m: float(np.mean(half[tag][m])) for m in METRICS}
        out["arms"][tag]["excess_over_floor"] = {m: out["arms"][tag]["half_n"][m] - out["floor"][m] for m in METRICS}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{'arm':<20}" + "".join(f"{m:>24}" for m in METRICS))
    for tag in arms:
        cells = []
        for m in METRICS:
            r = out["arms"][tag][m]
            sd = f"+-{r['sd']:.2f}" if r["sd"] is not None else ""
            ci = f" [{r['ci95'][0]:.1f},{r['ci95'][1]:.1f}]" if "ci95" in r else ""
            cells.append(f"{r['mean']:.2f}{sd}{ci}")
        print(f"{tag:<20}" + "".join(f"{c:>24}" for c in cells))
    print(f"{'floor (half n)':<20}" + "".join(f"{out['floor'][m]:>24.2f}" for m in METRICS))
    for tag in arms:
        print(f"{'excess ' + tag:<20}" + "".join(f"{out['arms'][tag]['excess_over_floor'][m]:>24.2f}" for m in METRICS))
    for spec, row in out["pairs"].items():
        cells = []
        for m in METRICS:
            r = row[m]
            p = f" p<0={r['p_lt_0_boot']:.2f}" if r.get("p_lt_0_boot") is not None else ""
            ci = f" [{r['ci95'][0]:+.1f},{r['ci95'][1]:+.1f}]{p}" if "ci95" in r else ""
            cells.append(f"{r['diff']:+.2f}{ci}")
        print(f"{'pair ' + spec:<20}" + "".join(f"{c:>24}" for c in cells))
    print("TESTS_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    main()
