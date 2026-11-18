"""Per video: recorded window-1 luma (night if < 70), and generated/recorded luma per window from long_color-style
reads; plus the history-latent channel-mean drift from summary.json (no video decode needed for the latter)."""
import json
import sys

import numpy as np

root, arm = sys.argv[1], sys.argv[2]
scenes = [0, 14, 21, 32, 41, 49, 62, 82, 10, 31, 52, 72]
print(f"== {arm}: history latent per-channel mean, distance to the GT context (window 1 history), per window")
rows = []
for sc in scenes:
    for sd in (42, 43):
        s = json.load(open(f"{root}/{arm}_scene{sc}_seed{sd}/summary.json"))
        w = s["windows"]
        m0 = np.array(w[0]["history_latent_raw_stats"]["mean"])
        s0 = np.array(w[0]["history_latent_raw_stats"]["std"])
        dm = [float(np.abs(np.array(x["history_latent_raw_stats"]["mean"]) - m0).mean()) for x in w]
        ds = [float((np.array(x["history_latent_raw_stats"]["std"]) / s0).mean()) for x in w]
        rows.append((sc, sd, dm, ds, m0))
dm = np.array([r[2] for r in rows]); ds = np.array([r[3] for r in rows])
print("  window              " + " ".join(f"{i:6d}" for i in range(1, 13)))
print("  |mean - mean_w1|    " + " ".join(f"{x:6.3f}" for x in dm.mean(0)))
print("  std / std_w1        " + " ".join(f"{x:6.3f}" for x in ds.mean(0)))
# which latent channels drift, signed, averaged over videos (window 12 history vs window 1)
sig = np.array([np.array(json.load(open(f"{root}/{arm}_scene{r[0]}_seed{r[1]}/summary.json"))["windows"][-1]
                         ["history_latent_raw_stats"]["mean"]) - r[4] for r in rows])
print("  signed channel drift w12-w1 (mean over videos):", " ".join(f"{x:+.3f}" for x in sig.mean(0)))
print("  same sign in >= 80% of videos:", int(((np.sign(sig) == np.sign(sig.mean(0))).mean(0) >= 0.8).sum()), "of 16")
