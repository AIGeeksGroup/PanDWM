"""Print the H-milestone comparison for watch_v83.sh:  python summary_v83.py <tag> <previous tag>"""
import json
import sys

E = "/path/to/pandwm_root/pandwm_runs/eval149_hpc3_v83_20261012"
tag, ptag = sys.argv[1], sys.argv[2]
rows = ((ptag, json.load(open(f"{E}/metrics/report_{ptag}.json"))["arms"][ptag]),
        (tag, json.load(open(f"{E}/metrics/report.json"))["arms"][tag]))
for name, v in rows:
    print(f"{name} FID {v['fid_all']:.2f} FVD {v['fvd_w1']:.1f} / {v['fvd_w2']:.1f} sum {v['fvd_w1'] + v['fvd_w2']:.1f}")
d = json.load(open(f"{E}/flicker_diag_{tag}.json"))
for a in d["arms"]:
    s = d["summary"][a]
    tert = " ".join(f"{x:.2f}" for x in s["by_motion"])
    print(f"{a} win {s['win_full']:.3f} bnd {s['bnd_full']:.3f} sharp {s['sharp']:.3f} tertiles {tert}")
print("pair", json.dumps(d["pair"]))
print(open(f"{E}/realism.log").read().splitlines()[0][:300])
