"""Long-video quality over time from long_feats.py features (2026-10-14; a diagnostic, not a gate).

All arms are compared on the SAME videos and windows: the (scene, seed, window) cells every listed arm has.  The real
frames of every cell come from one reference arm (--real-from, default the first arm), as in long_fid.py, so a number
here with all 24 videos and windows 1-6 / 7-12 reproduces long_fid.py.
  1. FID (Inception, torchmetrics features) generated vs real per time bucket, and KID x 1000 per window (unbiased,
     cubic polynomial kernel, the full sets; KID is unbiased at small n, FID is not).
  2. FVD (6 camera patches x I3D logits, the 149-clip instrument) per bucket of 16-frame windows.
  3. Seed-vs-seed reference (arms with two seeds per scene): FID / FVD between the seed-a and seed-b videos of the same
     scenes, next to each seed vs the real frames at the same n.  If the model's videos varied like real futures do,
     a real video would be "one more seed", so gen-vs-real ~ gen-vs-gen; the excess gen-vs-real minus gen-vs-gen is the
     part that is not explained by the videos inventing other, plausible content (a heuristic split).
  4. Uncertainty over scenes (all of a scene's seeds together): delete-one-scene jackknife SE of each arm's FID per
     bucket, of its late - early rise and of its difference to the first arm; paired-bootstrap P(difference < 0).
  5. Realism vs an UNPAIRED real pool (--pool NAME: feats_pool_NAME.npz, e.g. the real frames of the 78 non-test
     149-clip eval clips, 78 scenes): FID / FVD of each seed's videos vs the pool, next to the recorded frames of the
     same scenes and windows vs the pool (the floor at equal n).  Matching the recorded content does not help here, so
     excess = generated - recorded is the realism gap only.
  python3 long_analyze.py <feat_dir> <out.json> --arms A B ... [--real-from A] [--seeds 42] [--boot 300] [--pool P]
"""
import argparse
import json

import numpy as np
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUCKETS = [("w1-3", 1, 3), ("w4-6", 4, 6), ("w7-9", 7, 9), ("w10-12", 10, 12), ("early w1-6", 1, 6),
           ("late w7-12", 7, 12)]


def load(feat_dir, tag):
    z = np.load(f"{feat_dir}/feats_{tag}.npz")
    first, last = int(z["first_window"]), int(z["windows"])
    vids = [(int(a), int(b)) for a, b in zip(z["scenes"], z["seeds"])]
    return {"tag": tag, "first": first, "last": last, "vids": vids, "index": {v: i for i, v in enumerate(vids)},
            "inc_gen": z["inc_gen"], "inc_real": z["inc_real"], "i3d_gen": z["i3d_gen"], "i3d_real": z["i3d_real"]}


def cell(arm, kind, vid, w):
    """Features of window w (1-indexed) of video vid: inc (16, 2048) or i3d (6, 400)."""
    i, k = arm["index"][vid], w - arm["first"]
    if kind == "inc_gen" or kind == "inc_real":
        return arm[kind][i, 16 * k:16 * k + 16]
    return arm[kind][i, k]


class Stats:
    """Sufficient statistics (n, sum, sum of outer products) in float64 on DEV."""

    def __init__(self, d):
        self.n, self.s, self.ss = 0, torch.zeros(d, dtype=torch.float64, device=DEV), \
            torch.zeros(d, d, dtype=torch.float64, device=DEV)

    def add(self, x, weight=1):
        x = torch.as_tensor(np.asarray(x, np.float64), device=DEV).reshape(-1, self.s.shape[0])
        self.n += weight * x.shape[0]
        self.s += weight * x.sum(0)
        self.ss += weight * (x.T @ x)
        return self

    def plus(self, other, weight=1):
        out = Stats(self.s.shape[0])
        out.n, out.s, out.ss = self.n + weight * other.n, self.s + weight * other.s, self.ss + weight * other.ss
        return out

    def moments(self):
        mu = self.s / self.n
        cov = (self.ss - self.n * torch.outer(mu, mu)) / (self.n - 1)
        return mu, cov


def frechet(a, b):
    """torchmetrics' Frechet distance; trace sqrt(S1 S2) through symmetric eigensolvers (same value, stabler)."""
    mu1, s1 = a.moments()
    mu2, s2 = b.moments()
    ev, vec = torch.linalg.eigh(s1)
    root = (vec * ev.clamp_min(0).sqrt()) @ vec.T
    m = root @ s2 @ root
    tr = torch.linalg.eigvalsh((m + m.T) / 2).clamp_min(0).sqrt().sum()
    return float((mu1 - mu2).square().sum() + s1.trace() + s2.trace() - 2 * tr)


def frechet_eigvals(a, b):
    mu1, s1 = a.moments()
    mu2, s2 = b.moments()
    return float((mu1 - mu2).square().sum() + s1.trace() + s2.trace()
                 - 2 * torch.linalg.eigvals(s1 @ s2).sqrt().real.sum())


def kid(x, y):
    x = torch.as_tensor(np.asarray(x, np.float64), device=DEV).reshape(-1, 2048)
    y = torch.as_tensor(np.asarray(y, np.float64), device=DEV).reshape(-1, 2048)
    d = x.shape[1]
    kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
    n, m = x.shape[0], y.shape[0]
    return float((kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
                 + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1)) - 2 * kxy.mean()) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feat_dir")
    ap.add_argument("out")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--real-from", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="restrict to these seeds")
    ap.add_argument("--scenes", type=int, nargs="*", default=None, help="restrict to these scenes")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--buckets", nargs="*", default=None,
                    help="NAME=LO-HI window buckets (default w1-3 ... late w7-12); names starting with early / late / "
                         "all get intervals")
    a = ap.parse_args()
    arms = {t: load(a.feat_dir, t) for t in a.arms}
    ref = arms[a.real_from] if a.real_from in arms else load(a.feat_dir, a.real_from) if a.real_from else \
        arms[a.arms[0]]
    vids = set.intersection(*[set(arm["vids"]) for arm in arms.values()])
    if a.seeds:
        vids = {v for v in vids if v[1] in a.seeds}
    if a.scenes:
        vids = {v for v in vids if v[0] in a.scenes}
    first = max(arm["first"] for arm in arms.values())
    last = min(arm["last"] for arm in arms.values())
    vids = sorted(vids)
    ref_vids = set(ref["vids"])
    real_vid = {}
    for v in vids:   # the real frames of a video: the reference arm's copy of the same (scene, seed), else any seed
        real_vid[v] = v if v in ref_vids else next(r for r in ref["vids"] if r[0] == v[0])
    scenes = sorted({v[0] for v in vids})
    seeds = sorted({v[1] for v in vids})
    windows = list(range(first, last + 1))
    spec = BUCKETS if not a.buckets else [(b.split("=")[0], int(b.split("=")[1].split("-")[0]),
                                           int(b.split("=")[1].split("-")[1])) for b in a.buckets]
    buckets = [(name, lo, hi) for name, lo, hi in spec if lo >= first and hi <= last]
    if first > 1 or last < 12:
        buckets.append((f"all w{first}-{last}", first, last))
    print(f"{len(vids)} videos ({len(scenes)} scenes x seeds {seeds}), windows {first}-{last}, arms {a.arms}, "
          f"real frames from {ref['tag']}", flush=True)
    out = {"arms": a.arms, "real_from": ref["tag"], "videos": vids, "windows": [first, last], "fid": {}, "kid": {},
           "fvd": {}, "seed_split": {}, "bootstrap": {}}

    # per (scene, window) statistics, so buckets and bootstrap resamples are sums of them
    inc = {t: {} for t in a.arms}
    vd = {t: {} for t in a.arms}
    inc_real, vd_real = {}, {}
    for sc in scenes:
        for w in windows:
            r = Stats(2048)
            rv = Stats(400)
            for v in vids:
                if v[0] == sc:
                    r.add(cell(ref, "inc_real", real_vid[v], w))
                    rv.add(cell(ref, "i3d_real", real_vid[v], w))
            inc_real[sc, w], vd_real[sc, w] = r, rv
            for t, arm in arms.items():
                g, gv = Stats(2048), Stats(400)
                for v in vids:
                    if v[0] == sc:
                        g.add(cell(arm, "inc_gen", v, w))
                        gv.add(cell(arm, "i3d_gen", v, w))
                inc[t][sc, w], vd[t][sc, w] = g, gv

    def total(table, lo, hi, weights=None):
        acc = None
        for sc in scenes:
            wt = 1 if weights is None else weights.get(sc, 0)
            if wt == 0:
                continue
            for w in range(lo, hi + 1):
                acc = table[sc, w] if acc is None and wt == 1 else (Stats(table[sc, w].s.shape[0]).plus(
                    table[sc, w], wt) if acc is None else acc.plus(table[sc, w], wt))
        return acc

    check = (frechet(total(inc[a.arms[0]], *buckets[-1][1:]), total(inc_real, *buckets[-1][1:])),
             frechet_eigvals(total(inc[a.arms[0]], *buckets[-1][1:]), total(inc_real, *buckets[-1][1:])))
    print(f"frechet check (eigh vs eigvals) {check[0]:.4f} {check[1]:.4f}", flush=True)

    print("\n== 1. FID generated vs real per bucket (equal n across arms)")
    print("  arm            " + " ".join(f"{b[0]:>11s}" for b in buckets))
    for t in a.arms:
        out["fid"][t] = {name: frechet(total(inc[t], lo, hi), total(inc_real, lo, hi)) for name, lo, hi in buckets}
        print(f"  {t:14s} " + " ".join(f"{out['fid'][t][b[0]]:11.2f}" for b in buckets), flush=True)
    rr = {}
    if first <= 6 and last >= 7:
        rr["real early vs real late"] = frechet(total(inc_real, max(first, 1), 6), total(inc_real, 7, min(last, 12)))
        print(f"  real early vs real late (content shift of the recorded video): {rr['real early vs real late']:.2f}")
    out["fid_real_vs_real"] = rr

    print("\n== KID x 1000 per window, generated vs real (unbiased)")
    print("  arm            " + " ".join(f"{w:6d}" for w in windows))
    for t, arm in arms.items():
        out["kid"][t] = []
        for w in windows:
            g = np.concatenate([cell(arm, "inc_gen", v, w) for v in vids])
            r = np.concatenate([cell(ref, "inc_real", real_vid[v], w) for v in vids])
            out["kid"][t].append(kid(g, r))
        print(f"  {t:14s} " + " ".join(f"{x:6.2f}" for x in out["kid"][t]), flush=True)

    print("\n== 2. FVD (I3D, 6 camera patches) generated vs real per bucket of 16-frame windows")
    print("  arm            " + " ".join(f"{b[0]:>11s}" for b in buckets))
    for t in a.arms:
        out["fvd"][t] = {name: frechet(total(vd[t], lo, hi), total(vd_real, lo, hi)) for name, lo, hi in buckets}
        print(f"  {t:14s} " + " ".join(f"{out['fvd'][t][b[0]]:11.2f}" for b in buckets), flush=True)
    print(f"  samples per bucket: {len(vids)} videos x windows x 6 patches (e.g. {len(vids) * 6 * 6} for 6 windows)")

    if len(seeds) == 2 and all(any(v == (sc, sd) for v in vids) for sc in scenes for sd in seeds):
        sa, sb = seeds
        print(f"\n== 3. Seed-vs-seed reference (seed {sa} vs {sb}, {len(scenes)} scenes; same n on both sides)")
        print("  arm            bucket        FID gen-gen  FID gen-real  excess | FVD gen-gen  FVD gen-real  excess")
        for t, arm in arms.items():
            res = {}
            for name, lo, hi in buckets:
                g = {}
                gv = {}
                for sd in seeds:
                    s, sv = Stats(2048), Stats(400)
                    for sc in scenes:
                        for w in range(lo, hi + 1):
                            s.add(cell(arm, "inc_gen", (sc, sd), w))
                            sv.add(cell(arm, "i3d_gen", (sc, sd), w))
                    g[sd], gv[sd] = s, sv
                r, rv = Stats(2048), Stats(400)
                for sc in scenes:
                    for w in range(lo, hi + 1):
                        r.add(cell(ref, "inc_real", real_vid[(sc, sa)], w))
                        rv.add(cell(ref, "i3d_real", real_vid[(sc, sa)], w))
                gg, gr = frechet(g[sa], g[sb]), 0.5 * (frechet(g[sa], r) + frechet(g[sb], r))
                vg, vr = frechet(gv[sa], gv[sb]), 0.5 * (frechet(gv[sa], rv) + frechet(gv[sb], rv))
                res[name] = {"fid_gen_gen": gg, "fid_gen_real": gr, "fid_excess": gr - gg,
                             "fvd_gen_gen": vg, "fvd_gen_real": vr, "fvd_excess": vr - vg}
                print(f"  {t:14s} {name:12s} {gg:11.2f} {gr:13.2f} {gr - gg:7.2f} | {vg:11.2f} {vr:13.2f} "
                      f"{vr - vg:7.2f}", flush=True)
            out["seed_split"][t] = res

    if a.boot > 0 and len(scenes) >= 4:
        # Intervals: delete-one-scene jackknife SE (point +- 1.96 SE).  A bootstrap that resamples scenes WITH
        # replacement duplicates scenes, which shifts FID upward (fewer distinct frames), so its percentiles can miss
        # the point; it is kept only for P(difference < 0) of paired arm differences, where the shift cancels.
        main_b = [b for b in buckets if b[0].startswith(("early", "late", "all"))]
        names = [b[0] for b in main_b]
        # per-scene bucket sums once, so a resample is a weighted sum over scenes only
        per_scene = {}
        for name, lo, hi in main_b:
            per_scene[name, "real"] = {sc: total(inc_real, lo, hi, {sc: 1}) for sc in scenes}
            for t in a.arms:
                per_scene[name, t] = {sc: total(inc[t], lo, hi, {sc: 1}) for sc in scenes}

        def wsum(table, wts):
            acc = None
            for sc, wt in wts.items():
                if wt:
                    acc = Stats(table[sc].s.shape[0]).plus(table[sc], wt) if acc is None else acc.plus(table[sc], wt)
            return acc

        jack = {t: {n: [] for n in names} for t in a.arms}
        for drop in scenes:
            wts = {sc: (0 if sc == drop else 1) for sc in scenes}
            for name, lo, hi in main_b:
                real = wsum(per_scene[name, "real"], wts)
                for t in a.arms:
                    jack[t][name].append(frechet(wsum(per_scene[name, t], wts), real))
        rng = np.random.default_rng(0)
        draws = {t: {n: [] for n in names} for t in a.arms}
        for _ in range(a.boot):
            pick = rng.choice(len(scenes), len(scenes), replace=True)
            wts = {}
            for p_ in pick:
                wts[scenes[p_]] = wts.get(scenes[p_], 0) + 1
            for name, lo, hi in main_b:
                real = wsum(per_scene[name, "real"], wts)
                for t in a.arms:
                    draws[t][name].append(frechet(wsum(per_scene[name, t], wts), real))
        print(f"\n== 4. Uncertainty over scenes ({len(scenes)} scenes): FID +- 1.96 jackknife SE; "
              f"P(<0) from {a.boot} paired bootstrap resamples")

        def se(values):
            v = np.asarray(values, np.float64)
            return float(np.sqrt((len(v) - 1) / len(v) * np.square(v - v.mean()).sum()))

        base = a.arms[0]
        for t in a.arms:
            res = {}
            for name in names:
                p0 = out["fid"][t][name]
                e = se(jack[t][name])
                res[name] = {"point": p0, "se_jack": e, "ci": [p0 - 1.96 * e, p0 + 1.96 * e]}
                if t != base:
                    pd = p0 - out["fid"][base][name]
                    e = se(np.asarray(jack[t][name]) - np.asarray(jack[base][name]))
                    diff = np.asarray(draws[t][name]) - np.asarray(draws[base][name])
                    res[name]["minus_" + base] = {"point": pd, "se_jack": e, "ci": [pd - 1.96 * e, pd + 1.96 * e],
                                                  "p_below_0_boot": float((diff < 0).mean())}
            early_n = next((n for n in names if n.startswith("early")), None)
            late_n = next((n for n in names if n.startswith("late")), None)
            if early_n and late_n:
                pr = out["fid"][t][late_n] - out["fid"][t][early_n]
                e = se(np.asarray(jack[t][late_n]) - np.asarray(jack[t][early_n]))
                res["late_minus_early"] = {"point": pr, "se_jack": e, "ci": [pr - 1.96 * e, pr + 1.96 * e]}
            out["bootstrap"][t] = res
            line = "  ".join(f"{n} {res[n]['point']:.2f} [{res[n]['ci'][0]:.2f}, {res[n]['ci'][1]:.2f}]"
                             for n in names)
            print(f"  {t:14s} {line}")
            if "late_minus_early" in res:
                r = res["late_minus_early"]
                print(f"  {'':14s} late - early {r['point']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]")
            if t != base:
                for n in names:
                    d = res[n]["minus_" + base]
                    print(f"  {'':14s} {n} minus {base}: {d['point']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] "
                          f"P(<0) {d['p_below_0_boot']:.2f}", flush=True)
    if a.pool:
        z = np.load(f"{a.feat_dir}/feats_pool_{a.pool}.npz")
        p_inc, p_vd = Stats(2048).add(z["inc_real"]), Stats(400).add(z["i3d_real"])
        print(f"\n== 5. Realism vs the unpaired real pool '{a.pool}' ({z['inc_real'].shape[0]} clips, "
              f"{z['inc_real'].shape[0] * 32} frames); per seed, same n as the recorded frames")
        print("  arm            bucket        FID gen-pool  FID rec-pool  excess | FVD gen-pool  FVD rec-pool  excess")
        out["pool"] = {"name": a.pool, "arms": {}}
        for t, arm in arms.items():
            res = {}
            for name, lo, hi in buckets:
                fg, vg = [], []
                for sd in seeds:
                    g, gv = Stats(2048), Stats(400)
                    for sc in scenes:
                        for w in range(lo, hi + 1):
                            g.add(cell(arm, "inc_gen", (sc, sd), w))
                            gv.add(cell(arm, "i3d_gen", (sc, sd), w))
                    fg.append(frechet(g, p_inc))
                    vg.append(frechet(gv, p_vd))
                r, rv = Stats(2048), Stats(400)
                for sc in scenes:
                    for w in range(lo, hi + 1):
                        r.add(cell(ref, "inc_real", real_vid[(sc, seeds[0])], w))
                        rv.add(cell(ref, "i3d_real", real_vid[(sc, seeds[0])], w))
                fr, vr = frechet(r, p_inc), frechet(rv, p_vd)
                res[name] = {"fid_gen_pool": float(np.mean(fg)), "fid_rec_pool": fr, "fid_excess": float(np.mean(fg)) - fr,
                             "fvd_gen_pool": float(np.mean(vg)), "fvd_rec_pool": vr, "fvd_excess": float(np.mean(vg)) - vr}
                print(f"  {t:14s} {name:12s} {np.mean(fg):12.2f} {fr:13.2f} {np.mean(fg) - fr:7.2f} | "
                      f"{np.mean(vg):12.2f} {vr:13.2f} {np.mean(vg) - vr:7.2f}", flush=True)
            out["pool"]["arms"][t] = res
        print("  KID x 1000 per window vs the pool (generated, all videos; then the recorded frames)")
        pool_x = z["inc_real"].reshape(-1, 2048)
        out["pool"]["kid"] = {}
        for t, arm in list(arms.items()) + [("recorded", None)]:
            row = []
            for w in windows:
                if arm is None:
                    g = np.concatenate([cell(ref, "inc_real", real_vid[(sc, seeds[0])], w) for sc in scenes])
                else:
                    g = np.concatenate([cell(arm, "inc_gen", v, w) for v in vids])
                row.append(kid(g, pool_x))
            out["pool"]["kid"][t] = row
            print(f"  {t:14s} " + " ".join(f"{x:6.2f}" for x in row), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("LONG_ANALYZE_DONE")


if __name__ == "__main__":
    main()
