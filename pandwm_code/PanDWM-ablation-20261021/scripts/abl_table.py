"""ASCII table of the ablation from tests_analyze.py and abl_flicker.py outputs (2026-10-21).
  python3 abl_table.py analysis.json flicker.json CONTROL > TABLE.txt
Pairs are CONTROL:variant, i.e. control minus variant (negative = control lower).
"""
import json
import sys


def cell(r):
    ci = r.get("ci95")
    sd = f" sd {r['sd']:.2f}" if r.get("sd") is not None else ""
    return f"{r['mean']:.2f} [{ci[0]:.2f},{ci[1]:.2f}]{sd}" if ci else f"{r['mean']:.2f}{sd}"


def main():
    an, fl, ctrl = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
    cols = ("FID", "FVD-1", "FVD-2")
    print(f"Dev-78 ({an['clips']} clips, {an['scenes']} scenes), mean over seeds, 95% delete-one-scene jackknife")
    print(f"flicker = mean per-clip change ratio on the {fl['static_clips']} static clips (dev rule), fixed reference")
    print(f"{'arm':<12}" + "".join(f"{c:>34}" for c in cols) + f"{'Flicker':>34}")
    for tag, row in an["arms"].items():
        f = fl["arms"].get(tag, {}).get("flicker_static_dev")
        fc = (f"{f['mean']:.3f} [{f['ci95'][0]:.3f},{f['ci95'][1]:.3f}]" if f else "n/a")
        print(f"{tag:<12}" + "".join(f"{cell(row[c]):>34}" for c in cols) + f"{fc:>34}")
    print("per seed (" + ", ".join(str(s) for s in next(iter(an["arms"].values()))["seeds"]) + "):")
    for tag, row in an["arms"].items():
        f = fl["arms"].get(tag, {}).get("flicker_static_dev", {}).get("per_seed", [])
        print(f"  {tag:<10} " + "  ".join(f"{c} " + "/".join(f"{v:.2f}" for v in row[c]["per_seed"]) for c in cols)
              + "  flicker " + "/".join(f"{v:.3f}" for v in f))
    print(f"paired differences {ctrl} - variant (negative = {ctrl} lower), [95% jackknife], P(diff<0) by scene bootstrap:")
    for spec, row in an["pairs"].items():
        fr = fl["pairs"].get(spec, {}).get("flicker_static_dev")
        cells = []
        for c in cols:
            r = row[c]
            p = f" p<0={r['p_lt_0_boot']:.2f}" if r.get("p_lt_0_boot") is not None else ""
            cells.append(f"{r['diff']:+.2f} [{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}]{p}")
        fcell = f"{fr['diff']:+.3f} [{fr['ci95'][0]:+.3f},{fr['ci95'][1]:+.3f}]" if fr else "n/a"
        print(f"  {spec:<20} " + "  ".join(f"{c} {v}" for c, v in zip(cols, cells)) + f"  flicker {fcell}")
    print("floor (recorded half vs half, half n): " + " ".join(f"{k} {v:.2f}" for k, v in an["floor"].items()))


if __name__ == "__main__":
    main()
