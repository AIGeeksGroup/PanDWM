"""Same-step paired comparison of the V84 training monitors of two branches that share the start, data and rollout
depth sequence (2026-10-16, e.g. branch L vs branch I from seeds/I_4250; standard library only, login-node safe).
  python3 pair_v84_monitors.py <log A> <log B> [--from 4251] [--to 4500] [--bins 4251-4350 4351-4500]
Per window position (1 = first rollout window): mean of (A - B) for the latent low-pass change ratio `mot` and the HF
energy ratio `hf` over the steps present in both logs, with the standard error over steps; deep = positions 5-9.
Also prints A's mm3 rollout keys (mm3_roll_lf_ratio, mm3_roll_cap_frac, mm3_roll_dmd_ratio) when present."""
import argparse
import math
import re
import statistics as st


def v84(path):
    out = {}
    for line in open(path, errors="replace"):
        if not line.startswith("V84 step="):
            continue
        step = int(re.search(r"step=(\d+)", line).group(1))
        k = int(re.search(r" k=(\d+)", line).group(1))
        hf = [float(v) for v in re.search(r"hf=\[([^\]]*)\]", line).group(1).split()]
        mot = [float(v) for v in re.search(r"mot=\[([^\]]*)\]", line).group(1).split()]
        out[step] = (k, hf, mot)
    return out


def keys(path, names):
    vals = {n: [] for n in names}
    for line in open(path, errors="replace"):
        for n in names:
            m = re.match(rf"\s*{n}:\s*([-+0-9.eE]+|nan)", line)
            if m:
                vals[n].append(float(m.group(1)))
    return vals


def summary(diffs):
    if not diffs:
        return "   -   "
    se = st.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
    return f"{st.mean(diffs):+.4f} (se {se:.4f}, n {len(diffs)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--from", dest="lo", type=int, default=0)
    ap.add_argument("--to", dest="hi", type=int, default=10 ** 9)
    ap.add_argument("--bins", nargs="*", default=[])
    a = ap.parse_args()
    A, B = v84(a.a), v84(a.b)
    common = sorted(s for s in set(A) & set(B) if a.lo <= s <= a.hi)
    bad = [s for s in common if A[s][0] != B[s][0]]
    print(f"steps in A {len(A)}, in B {len(B)}, common {len(common)} ({common[0] if common else '-'}.."
          f"{common[-1] if common else '-'}); depth k differs on {len(bad)} common steps")
    bins = [(int(x.split("-")[0]), int(x.split("-")[1])) for x in a.bins] or [(a.lo, a.hi)]
    for lo, hi in bins:
        steps = [s for s in common if lo <= s <= hi and A[s][0] == B[s][0]]
        print(f"== steps {lo}-{hi} ({len(steps)} paired)")
        print(f"  {'pos':>3s}  {'mot A':>6s} {'mot B':>6s}  mot A-B                        hf A-B")
        deep_m = []
        for pos in range(1, 11):
            dm = [A[s][2][pos - 1] - B[s][2][pos - 1] for s in steps if len(A[s][2]) >= pos]
            dh = [A[s][1][pos - 1] - B[s][1][pos - 1] for s in steps if len(A[s][1]) >= pos]
            ma = [A[s][2][pos - 1] for s in steps if len(A[s][2]) >= pos]
            mb = [B[s][2][pos - 1] for s in steps if len(B[s][2]) >= pos]
            if not dm:
                continue
            if 5 <= pos <= 9:
                deep_m += dm
            print(f"  {pos:3d}  {st.mean(ma):6.3f} {st.mean(mb):6.3f}  {summary(dm):30s} {summary(dh)}")
        print(f"  deep (positions 5-9) mot A-B: {summary(deep_m)}")
    kv = keys(a.a, ["mm3_roll_lf_ratio", "mm3_roll_cap_frac", "mm3_roll_dmd_ratio", "mm3_roll_on"])
    for n, v in kv.items():
        if v:
            print(f"A {n}: n {len(v)}, mean {st.mean(v):.4f}, last {v[-1]:.4f}")


if __name__ == "__main__":
    main()
