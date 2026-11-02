"""Report section 34 gate for the 4D GS depth pilot (2026-10-17).

  python3 depth_pilot_gate.py <eval dir with cov3000_h100.json and depth_pilot_{C,D1,D2}_1000.json>

Pre-registered rule (D vs C, same eval: 25 dev clips, gap 4, H100):
  main:   psnr_pooled D - C >= +0.3 dB
  guards: psnr_own D >= C - 0.3 dB;  coverage_pooled D >= 0.99
Also prints the paired per-clip difference with a 95% t interval (report only, not part of the rule).
"""
import json
import math
import sys
from pathlib import Path

T975 = 2.064  # t(0.975, 24)


def load(path):
    d = json.loads(Path(path).read_text())
    return d["mean"], {r["clip_index"]: r for r in d["rows"]}


def paired(a, b, key):
    common = sorted(set(a) & set(b))
    diff = [b[i][key] - a[i][key] for i in common]
    n = len(diff)
    mean = sum(diff) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diff) / (n - 1)) if n > 1 else float("nan")
    half = T975 * sd / math.sqrt(n)
    return mean, mean - half, mean + half, sum(x > 0 for x in diff), n


def main():
    root = Path(sys.argv[1])
    base_m, base_r = load(root / "cov3000_h100.json")
    c_m, c_r = load(root / "depth_pilot_C_1000.json")
    print(f"{'arm':<6}{'psnr_own':>10}{'psnr_pooled':>13}{'cov_pooled':>12}{'psnr_single':>13}{'cov_single':>12}")
    for name, m in [("base", base_m), ("C", c_m)]:
        print(f"{name:<6}{m['psnr_own']:>10.3f}{m['psnr_pooled']:>13.3f}{m['coverage_pooled']:>12.4f}"
              f"{m['psnr_single']:>13.3f}{m['coverage_single']:>12.4f}")
    verdicts = {}
    for arm in ("D1", "D2"):
        p = root / f"depth_pilot_{arm}_1000.json"
        if not p.is_file():
            print(f"{arm:<6} (missing)")
            continue
        m, r = load(p)
        print(f"{arm:<6}{m['psnr_own']:>10.3f}{m['psnr_pooled']:>13.3f}{m['coverage_pooled']:>12.4f}"
              f"{m['psnr_single']:>13.3f}{m['coverage_single']:>12.4f}")
        main_d = m["psnr_pooled"] - c_m["psnr_pooled"]
        own_ok = m["psnr_own"] >= c_m["psnr_own"] - 0.3
        cov_ok = m["coverage_pooled"] >= 0.99
        ok = main_d >= 0.3 and own_ok and cov_ok
        verdicts[arm] = (ok, main_d)
        for key in ("psnr_pooled", "psnr_single", "psnr_own"):
            mean, lo, hi, wins, n = paired(c_r, r, key)
            print(f"    {arm}-C {key:<12} {mean:+.3f} dB  95% [{lo:+.3f}, {hi:+.3f}]  better on {wins}/{n} clips")
        print(f"    gate: pooled {main_d:+.3f} (>= +0.3: {main_d >= 0.3}), own guard {own_ok}, "
              f"coverage guard {cov_ok} -> {'PASS' if ok else 'FAIL'}")
    passed = [a for a, (ok, _) in verdicts.items() if ok]
    if passed:
        best = max(passed, key=lambda a: verdicts[a][1])
        print(f"VERDICT: PASS ({', '.join(passed)}); best = {best} -> extend {best} to 3000 steps")
    elif len(verdicts) == 2:
        print("VERDICT: FAIL (neither arm passes) -> render-side fixes only; propose multi-layer / multi-frame GS")


if __name__ == "__main__":
    main()
