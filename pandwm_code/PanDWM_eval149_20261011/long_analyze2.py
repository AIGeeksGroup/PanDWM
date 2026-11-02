"""Long-video quality over time from long_feats.py features, memory-light (2026-10-14; long_analyze.py's successor).

long_analyze.py kept one 2048 x 2048 float64 statistic per (scene, window, arm): ~20 GB for 12 scenes, ~130 GB for
Dev-76-L.  This keeps the raw features (float32) and computes moments on demand on the GPU, with the same Frechet
distance (torchmetrics' formula through symmetric eigensolvers), so on the same videos it reproduces long_analyze.py.
All arms are compared on the same (scene, seed, window) cells; real frames come from --real-from (default: first arm).
  1. FID / FVD generated vs recorded per window bucket (equal n across arms); KID x 1000 per window.
  2. Seed-vs-seed reference (two seeds per scene): gen-gen vs gen-real at the same n, excess = gen-real - gen-gen.
  3. Uncertainty: delete-one-group jackknife over scenes (groups of --jack-group scenes) -> point +- 1.96 SE, for each
     arm, its late - early rise, and its difference to the first arm; paired bootstrap P(difference < 0).
  4. Realism vs an unpaired real pool, either --pool NAME (feats_pool_NAME.npz; drop pool clips that show the evaluated
     scenes with --pool-exclude-clips) or --pool-split: the scenes are split in two halves and each half's generated
     and recorded frames are compared with the OTHER half's recorded frames of all windows (disjoint scenes, same
     pipeline, all time offsets); the two directions are averaged.  excess = generated - recorded vs the same pool.
  python3 long_analyze2.py <feat_dir> <out.json> --arms A B ... [--buckets NAME=LO-HI ...] [--pool-split]
"""
import argparse
import json

import numpy as np
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_BUCKETS = ["w1-3=1-3", "w4-6=4-6", "w7-9=7-9", "w10-12=10-12", "early w1-6=1-6", "late w7-12=7-12"]


def load(feat_dir, tag):
    z = np.load(f"{feat_dir}/feats_{tag}.npz")
    vids = [(int(a), int(b)) for a, b in zip(z["scenes"], z["seeds"])]
    return {"tag": tag, "first": int(z["first_window"]), "last": int(z["windows"]), "vids": vids,
            "index": {v: i for i, v in enumerate(vids)}, "inc_gen": z["inc_gen"], "inc_real": z["inc_real"],
            "i3d_gen": z["i3d_gen"], "i3d_real": z["i3d_real"]}


def cells(arm, kind, vid, windows):
    """Features of the listed windows (1-indexed) of one video, stacked: inc (16 * n, 2048) or i3d (6 * n, 400)."""
    i = arm["index"][vid]
    if kind.startswith("inc"):
        return np.concatenate([arm[kind][i, 16 * (w - arm["first"]):16 * (w - arm["first"]) + 16] for w in windows])
    return np.concatenate([arm[kind][i, w - arm["first"]] for w in windows])


def moments(x):
    x = torch.as_tensor(np.ascontiguousarray(x), device=DEV, dtype=torch.float64)
    mu = x.mean(0)
    xc = x - mu
    return mu, xc.T @ xc / (x.shape[0] - 1)


def frechet(a, b):
    (mu1, s1), (mu2, s2) = a, b
    ev, vec = torch.linalg.eigh(s1)
    root = (vec * ev.clamp_min(0).sqrt()) @ vec.T
    m = root @ s2 @ root
    tr = torch.linalg.eigvalsh((m + m.T) / 2).clamp_min(0).sqrt().sum()
    return float((mu1 - mu2).square().sum() + s1.trace() + s2.trace() - 2 * tr)


def kid(x, y):
    x = torch.as_tensor(np.ascontiguousarray(x), device=DEV, dtype=torch.float64)
    y = torch.as_tensor(np.ascontiguousarray(y), device=DEV, dtype=torch.float64)
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
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--scenes", type=int, nargs="*", default=None)
    ap.add_argument("--buckets", nargs="*", default=None, help="NAME=LO-HI; names starting with early / late get "
                                                               "intervals and the late - early rise")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--jack-group", type=int, default=1, help="scenes per jackknife group")
    ap.add_argument("--pool", default=None)
    ap.add_argument("--pool-exclude-clips", type=int, nargs="*", default=None)
    ap.add_argument("--pool-split", action="store_true")
    ap.add_argument("--no-kid", action="store_true")
    a = ap.parse_args()
    arms = {t: load(a.feat_dir, t) for t in a.arms}
    ref = arms[a.real_from] if a.real_from in arms else (load(a.feat_dir, a.real_from) if a.real_from else
                                                          arms[a.arms[0]])
    vids = set.intersection(*[set(arm["vids"]) for arm in arms.values()])
    if a.seeds:
        vids = {v for v in vids if v[1] in a.seeds}
    if a.scenes:
        vids = {v for v in vids if v[0] in a.scenes}
    vids = sorted(vids)
    first = max(arm["first"] for arm in arms.values())
    last = min(arm["last"] for arm in arms.values())
    real_vid = {v: (v if v in ref["index"] else next(r for r in ref["vids"] if r[0] == v[0])) for v in vids}
    scenes = sorted({v[0] for v in vids})
    seeds = sorted({v[1] for v in vids})
    windows = list(range(first, last + 1))
    spec = [(b.split("=")[0], int(b.split("=")[1].split("-")[0]), int(b.split("=")[1].split("-")[1]))
            for b in (a.buckets or DEFAULT_BUCKETS)]
    buckets = [(n, lo, hi) for n, lo, hi in spec if lo >= first and hi <= last]
    if first > 1 or last < 12:
        buckets.append((f"all w{first}-{last}", first, last))
    print(f"{len(vids)} videos ({len(scenes)} scenes x seeds {seeds}), windows {first}-{last}, arms {a.arms}, "
          f"real frames from {ref['tag']}", flush=True)
    out = {"arms": a.arms, "real_from": ref["tag"], "videos": vids, "windows": [first, last],
           "buckets": [list(b) for b in buckets], "fid": {}, "fvd": {}, "kid": {}, "seed_split": {}, "uncertainty": {},
           "pool": {}}

    def gen(t, kind, vs, ws):
        return np.concatenate([cells(arms[t], kind, v, ws) for v in vs])

    def rec(kind, vs, ws):
        return np.concatenate([cells(ref, kind, real_vid[v], ws) for v in vs])

    def wr(lo, hi):
        return list(range(lo, hi + 1))

    print("\n== 1. FID / FVD generated vs recorded per bucket (equal n across arms)")
    print("  FID  arm           " + " ".join(f"{b[0]:>11s}" for b in buckets))
    rec_m = {b[0]: (moments(rec("inc_real", vids, wr(b[1], b[2]))), moments(rec("i3d_real", vids, wr(b[1], b[2]))))
             for b in buckets}
    for t in a.arms:
        out["fid"][t], out["fvd"][t] = {}, {}
        for name, lo, hi in buckets:
            out["fid"][t][name] = frechet(moments(gen(t, "inc_gen", vids, wr(lo, hi))), rec_m[name][0])
            out["fvd"][t][name] = frechet(moments(gen(t, "i3d_gen", vids, wr(lo, hi))), rec_m[name][1])
        print(f"  FID  {t:13s} " + " ".join(f"{out['fid'][t][b[0]]:11.2f}" for b in buckets), flush=True)
    for t in a.arms:
        print(f"  FVD  {t:13s} " + " ".join(f"{out['fvd'][t][b[0]]:11.2f}" for b in buckets), flush=True)

    if not a.no_kid:
        print("\n== KID x 1000 per window, generated vs recorded")
        print("  arm            " + " ".join(f"{w:6d}" for w in windows))
        for t in a.arms:
            out["kid"][t] = [kid(gen(t, "inc_gen", vids, [w]), rec("inc_real", vids, [w])) for w in windows]
            print(f"  {t:14s} " + " ".join(f"{x:6.2f}" for x in out["kid"][t]), flush=True)

    if len(seeds) == 2 and all((sc, sd) in set(vids) for sc in scenes for sd in seeds):
        sa, sb = seeds
        print(f"\n== 2. Seed-vs-seed reference (seed {sa} vs {sb}, {len(scenes)} scenes)")
        print("  arm            bucket        FID gen-gen  gen-rec  excess | FVD gen-gen  gen-rec  excess")
        for t in a.arms:
            res = {}
            for name, lo, hi in buckets:
                va, vb = [(sc, sa) for sc in scenes], [(sc, sb) for sc in scenes]
                r_i, r_v = moments(rec("inc_real", va, wr(lo, hi))), moments(rec("i3d_real", va, wr(lo, hi)))
                ga, gb = moments(gen(t, "inc_gen", va, wr(lo, hi))), moments(gen(t, "inc_gen", vb, wr(lo, hi)))
                va_, vb_ = moments(gen(t, "i3d_gen", va, wr(lo, hi))), moments(gen(t, "i3d_gen", vb, wr(lo, hi)))
                gg, gr = frechet(ga, gb), 0.5 * (frechet(ga, r_i) + frechet(gb, r_i))
                vg, vr = frechet(va_, vb_), 0.5 * (frechet(va_, r_v) + frechet(vb_, r_v))
                res[name] = {"fid_gen_gen": gg, "fid_gen_rec": gr, "fid_excess": gr - gg, "fvd_gen_gen": vg,
                             "fvd_gen_rec": vr, "fvd_excess": vr - vg}
                print(f"  {t:14s} {name:12s} {gg:11.2f} {gr:8.2f} {gr - gg:7.2f} | {vg:11.2f} {vr:8.2f} {vr - vg:7.2f}",
                      flush=True)
            out["seed_split"][t] = res

    main_b = [b for b in buckets if b[0].startswith(("early", "late", "all"))]
    if a.boot >= 0 and len(scenes) >= 4 and main_b:
        groups = [scenes[i:i + a.jack_group] for i in range(0, len(scenes), a.jack_group)]
        names = [b[0] for b in main_b]

        def add_stats(table, vs):
            for name, lo, hi in main_b:
                r = moments(rec("inc_real", vs, wr(lo, hi)))
                for t in a.arms:
                    table[t][name].append(frechet(moments(gen(t, "inc_gen", vs, wr(lo, hi))), r))

        jack = {t: {n: [] for n in names} for t in a.arms}
        for g in groups:
            add_stats(jack, [v for v in vids if v[0] not in g])
        draws = {t: {n: [] for n in names} for t in a.arms}
        rng = np.random.default_rng(0)
        for _ in range(max(a.boot, 0)):
            pick = rng.choice(len(scenes), len(scenes), replace=True)
            add_stats(draws, [v for p in pick for v in vids if v[0] == scenes[p]])

        def se(values):
            v = np.asarray(values, np.float64)
            return float(np.sqrt((len(v) - 1) / len(v) * np.square(v - v.mean()).sum()))

        print(f"\n== 3. Uncertainty: FID +- 1.96 jackknife SE ({len(groups)} groups of <= {a.jack_group} scenes); "
              f"P(<0) from {a.boot} paired bootstrap resamples")
        base = a.arms[0]
        early_n = next((n for n in names if n.startswith("early")), None)
        late_n = next((n for n in names if n.startswith("late")), None)
        for t in a.arms:
            res = {}
            for n in names:
                p0, e = out["fid"][t][n], se(jack[t][n])
                res[n] = {"point": p0, "se_jack": e, "ci": [p0 - 1.96 * e, p0 + 1.96 * e]}
                if t != base:
                    pd = p0 - out["fid"][base][n]
                    e = se(np.asarray(jack[t][n]) - np.asarray(jack[base][n]))
                    diff = np.asarray(draws[t][n]) - np.asarray(draws[base][n])
                    res[n]["minus_" + base] = {"point": pd, "se_jack": e, "ci": [pd - 1.96 * e, pd + 1.96 * e],
                                               "p_below_0_boot": float((diff < 0).mean()) if a.boot else None}
            if early_n and late_n:
                pr = out["fid"][t][late_n] - out["fid"][t][early_n]
                e = se(np.asarray(jack[t][late_n]) - np.asarray(jack[t][early_n]))
                res["late_minus_early"] = {"point": pr, "se_jack": e, "ci": [pr - 1.96 * e, pr + 1.96 * e]}
            out["uncertainty"][t] = res
            print(f"  {t:14s} " + "  ".join(f"{n} {res[n]['point']:.2f} [{res[n]['ci'][0]:.2f}, {res[n]['ci'][1]:.2f}]"
                                           for n in names))
            if "late_minus_early" in res:
                r = res["late_minus_early"]
                print(f"  {'':14s} late - early {r['point']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]")
            if t != base:
                for n in names:
                    d = res[n]["minus_" + base]
                    pb = "" if d["p_below_0_boot"] is None else f" P(<0) {d['p_below_0_boot']:.2f}"
                    print(f"  {'':14s} {n} minus {base}: {d['point']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]{pb}",
                          flush=True)

    if a.pool or a.pool_split:
        if a.pool_split:
            halves = [scenes[0::2], scenes[1::2]]
            plan = [(halves[0], halves[1]), (halves[1], halves[0])]
            label = f"split halves of {len(scenes)} scenes (each half vs the other half's recorded frames, all windows)"
        else:
            z = np.load(f"{a.feat_dir}/feats_pool_{a.pool}.npz")
            keep = [i for i, c in enumerate(z["clips"]) if not a.pool_exclude_clips or int(c) not in
                    set(a.pool_exclude_clips)]
            pool_inc, pool_vd = z["inc_real"][keep].reshape(-1, 2048), z["i3d_real"][keep].reshape(-1, 400)
            plan = [(scenes, None)]
            label = f"pool '{a.pool}' ({len(keep)} clips, {pool_inc.shape[0]} frames)"
        print(f"\n== 4. Realism vs an unpaired real reference: {label}; per seed, same n as the recorded frames")
        print("  arm            bucket        FID gen  FID rec  excess | FVD gen  FVD rec  excess")
        res_all = {t: {} for t in a.arms}
        for name, lo, hi in buckets:
            acc = {t: {"fg": [], "fr": [], "vg": [], "vr": []} for t in a.arms}
            for evals, other in plan:
                if other is None:
                    p_i, p_v = moments(pool_inc), moments(pool_vd)
                else:
                    pv = [v for v in vids if v[0] in other and v[1] == seeds[0]]
                    p_i, p_v = moments(rec("inc_real", pv, windows)), moments(rec("i3d_real", pv, windows))
                rv = [v for v in vids if v[0] in evals and v[1] == seeds[0]]
                fr = frechet(moments(rec("inc_real", rv, wr(lo, hi))), p_i)
                vr = frechet(moments(rec("i3d_real", rv, wr(lo, hi))), p_v)
                for t in a.arms:
                    for sd in seeds:
                        gv = [v for v in vids if v[0] in evals and v[1] == sd]
                        acc[t]["fg"].append(frechet(moments(gen(t, "inc_gen", gv, wr(lo, hi))), p_i))
                        acc[t]["vg"].append(frechet(moments(gen(t, "i3d_gen", gv, wr(lo, hi))), p_v))
                    acc[t]["fr"].append(fr)
                    acc[t]["vr"].append(vr)
            for t in a.arms:
                fg, frr = float(np.mean(acc[t]["fg"])), float(np.mean(acc[t]["fr"]))
                vg, vrr = float(np.mean(acc[t]["vg"])), float(np.mean(acc[t]["vr"]))
                res_all[t][name] = {"fid_gen": fg, "fid_rec": frr, "fid_excess": fg - frr, "fvd_gen": vg,
                                    "fvd_rec": vrr, "fvd_excess": vg - vrr}
        for t in a.arms:
            for name, _, _ in buckets:
                r = res_all[t][name]
                print(f"  {t:14s} {name:12s} {r['fid_gen']:8.2f} {r['fid_rec']:8.2f} {r['fid_excess']:7.2f} | "
                      f"{r['fvd_gen']:8.2f} {r['fvd_rec']:8.2f} {r['fvd_excess']:7.2f}", flush=True)
        out["pool"] = {"label": label, "arms": res_all}
    json.dump(out, open(a.out, "w"), indent=1)
    print("LONG_ANALYZE2_DONE")


if __name__ == "__main__":
    main()
