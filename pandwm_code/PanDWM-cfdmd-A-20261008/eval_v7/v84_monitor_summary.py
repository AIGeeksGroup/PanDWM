"""Summarise the per-step V84 lines of a v8.4 training log (2026-10-13).

  python3 v84_monitor_summary.py <train.log> [--bins 4001-4050 4051-4150 4151-4250]

Each V84 line holds, averaged over ranks: k, the HF energy ratio (hf) and low-pass change ratio (mot) of every rollout
window vs its real frames, and on student steps the trained window's HF ratio (sf_ratio) and the teacher direction
tdir=<proj>/<energy ratio> (proj > 0: the DMD target adds the student's own detail; < 0: removes it).
Prints, per step bin: the HF / change ratio by rollout depth (window 1 = from the real history), and the teacher
direction and trained-window HF ratio by k.
"""
import argparse
import math
import re
from collections import defaultdict

LINE = re.compile(r"V84 step=(\d+) k=(\d+) hf=\[([^\]]*)\] mot=\[([^\]]*)\] sf_ratio=(\S+) .*tdir=(\S+)")


def num(s):
    try:
        v = float(s)
    except ValueError:
        return math.nan
    return v


def mean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--bins", nargs="*", default=None)
    a = ap.parse_args()
    rows = []
    for line in open(a.log, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        step, k = int(m.group(1)), int(m.group(2))
        hf = [num(x) for x in m.group(3).split()]
        mot = [num(x) for x in m.group(4).split()]
        sf = num(m.group(5)) if m.group(5) != "-" else math.nan
        tdir = m.group(6)
        proj, er = (num(tdir.split("/")[0]), num(tdir.split("/")[1])) if "/" in tdir else (math.nan, math.nan)
        rows.append((step, k, hf, mot, sf, proj, er))
    if not rows:
        print("NO_V84_LINES")
        return
    steps = [r[0] for r in rows]
    bins = a.bins or [f"{min(steps)}-{max(steps)}"]
    print(f"{len(rows)} V84 lines, steps {min(steps)}..{max(steps)}")
    for b in bins:
        lo, hi = (int(x) for x in b.split("-"))
        sel = [r for r in rows if lo <= r[0] <= hi]
        if not sel:
            continue
        by_w_hf, by_w_mot = defaultdict(list), defaultdict(list)
        by_k_proj, by_k_er, by_k_sf, by_k_n = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(int)
        for _, k, hf, mot, sf, proj, er in sel:
            by_k_n[k] += 1
            for w, v in enumerate(hf):
                by_w_hf[w + 1].append(v)
            for w, v in enumerate(mot):
                by_w_mot[w + 1].append(v)
            by_k_proj[k].append(proj)
            by_k_er[k].append(er)
            by_k_sf[k].append(sf)
        ws = sorted(by_w_hf)
        print(f"== steps {lo}-{hi}: {len(sel)} steps")
        print("  rollout window   " + " ".join(f"{w:6d}" for w in ws))
        print("  HF ratio         " + " ".join(f"{mean(by_w_hf[w]):6.3f}" for w in ws))
        print("  change ratio     " + " ".join(f"{mean(by_w_mot[w]):6.3f}" for w in ws))
        print("  n                " + " ".join(f"{len(by_w_hf[w]):6d}" for w in ws))
        ks = sorted(by_k_n)
        print("  k                " + " ".join(f"{k:6d}" for k in ks))
        print("  steps            " + " ".join(f"{by_k_n[k]:6d}" for k in ks))
        print("  teacher proj     " + " ".join(f"{mean(by_k_proj[k]):+6.3f}" for k in ks))
        print("  teacher E ratio  " + " ".join(f"{mean(by_k_er[k]):6.3f}" for k in ks))
        print("  trained-win HF   " + " ".join(f"{mean(by_k_sf[k]):6.3f}" for k in ks))
        deep = [p for _, k, _, _, _, p, _ in sel if k >= 7]
        shallow = [p for _, k, _, _, _, p, _ in sel if k <= 4]
        print(f"  teacher proj k<=4 {mean(shallow):+.4f} (n={sum(math.isfinite(x) for x in shallow)}), "
              f"k>=7 {mean(deep):+.4f} (n={sum(math.isfinite(x) for x in deep)})")
    print("V84_SUMMARY_DONE")


if __name__ == "__main__":
    main()
