"""Round-3 summary and gates (GS_FT3_PLAN_20261030.md section 4) on HPC3: recon (scores/<arm>_4d.json) and shift
(shift/<arm>/report.json) of the GS-Dev arms, against V0 = R0 + v4 inference (E0 recon, FT3B_V0 shift rerun).
    python3 ft3_summary.py [arms...]          (default FT3B_V0 FT3B_V2 FT3_G0 FT3_G1 FT3_G2)
    python3 ft3_summary.py --test [arms...]   (Test-S 71: tab5_gen_gs_ft3_20261030/report_ours4d_<arm>.json)
"""
import json
import os
import sys

B = "/path/to/pandwm_root/pandwm_runs/gs_dev_hpc3"
T = "/path/to/pandwm_root/pandwm_runs/tab5_gen_gs_ft3_20261030"
RECON = {"FT3B_V0": "E0", "FT3B_V2": "CD_V2"}
D = (0, 1, 2, 4)


def recon(name):
    f = "%s/scores/%s_4d.json" % (B, RECON.get(name, name))
    if not os.path.exists(f):
        return None
    r = json.load(open(f))
    a = r["image"]
    d = r["depth"]["all"]["static"]
    return {"full": a["all"]["psnr_fill"], "far": sum(a[g]["psnr_fill"] for g in ("half", "back", "fwd")) / 3,
            "next": a["next"]["psnr_fill"], "absrel": d["abs_rel"], "d1": d["delta1"]}


def shift(path):
    if not os.path.exists(path):
        return None
    r = json.load(open(path))
    out = {}
    for d in D:
        a = r["arms"]["4d_%d" % d]
        out[d] = (a["fid"], a["fvd_w1"], a["coverage"], a.get("sharpness", float("nan")))
    out["depth"] = (r.get("depth") or {}).get("all")
    out["clips"] = r["clips"]
    return out


def line(name, rc, sh):
    s = name
    if rc:
        s += " | recon full %.2f far %.2f next %.2f absrel %.3f d1 %.3f" % (rc["full"], rc["far"], rc["next"],
                                                                           rc["absrel"], rc["d1"])
    if sh:
        s += "\n    shift(%d) " % sh["clips"] + "  ".join("%dm %.2f/%.1f cov%.3f sh%.3f" % ((d,) + sh[d]) for d in D)
        s += "  | mean FVD %.1f" % (sum(sh[d][1] for d in D) / 4)
        if sh["depth"]:
            s += "\n    gen-scene depth absrel %.3f d1 %.3f" % (sh["depth"]["abs_rel"], sh["depth"]["delta1"])
    return s


def gates(rc, sh, rc0, sh0):
    g1 = sum(sh[d][1] for d in D) / 4 < sum(sh0[d][1] for d in D) / 4
    g2 = all(sh[d][0] <= sh0[d][0] + 1.0 and sh[d][1] <= sh0[d][1] + 10.0 for d in D)
    g3 = rc is not None and rc["d1"] >= 0.678 and rc["far"] >= 18.10
    g4 = all(sh[d][3] >= sh0[d][3] - 0.02 and sh[d][2] >= sh0[d][2] - 0.01 for d in D)
    return "gates: 1 meanFVD %s | 2 no-regress %s | 3 d1/far %s | 4 sharp/cov %s -> %s (5 = visual check)" % (
        g1, g2, g3, g4, "PASS" if g1 and g2 and g3 and g4 else "FAIL")


args = sys.argv[1:]
test = bool(args) and args[0] == "--test"
if test:
    args = args[1:]
arms = args or ["FT3B_V0", "FT3B_V2", "FT3_G0", "FT3_G1", "FT3_G2"]
if test:
    sh0 = shift("%s/report_ours4d_FT3B_V0.json" % T)
    for a in arms:
        sh = shift("%s/report_ours4d_%s.json" % (T, a))
        print(line("TEST " + a, None, sh))
else:
    rc0, sh0 = recon("FT3B_V0"), shift("%s/shift/FT3B_V0/report.json" % B)
    for a in arms:
        rc, sh = recon(a), shift("%s/shift/%s/report.json" % (B, a))
        print(line(a, rc, sh))
        if a != "FT3B_V0" and rc and sh and sh0:
            print("    " + gates(rc, sh, rc0, sh0))
