import json, os
B = "/path/to/pandwm_root/pandwm_runs/gs_dev_hpc3/shift"
for arm in ("FT3B_V0", "FT3_G3b", "NF_N3", "RS_G3b"):
    f = "%s/%s/report.json" % (B, arm)
    if not os.path.exists(f):
        print(arm, "missing"); continue
    r = json.load(open(f))
    print(arm, "clips", r["clips"])
    for k in sorted(r["arms"]):
        a = r["arms"][k]
        print("   %-10s FID %6.2f FVD %6.1f cov %s sharp %s" % (k, a["fid"], a["fvd_w1"],
              "%.3f" % a["coverage"] if a.get("coverage") is not None else "-",
              "%.3f" % a["sharpness"] if a.get("sharpness") is not None else "-"))
    st = r.get("scene_stats") or {}
    print("   stats", {k: round(v, 4) for k, v in st.items() if k.startswith("magcull") and "repair" in k})
