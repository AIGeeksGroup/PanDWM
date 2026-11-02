"""Cut a window range out of a long_feats.py feature file (2026-10-15), e.g. the student's own window 7 of its
Dev-76-L rollouts as a baseline next to teachers that generated only window 7 from the same history:
  python3 feats_slice.py <in feats_TAG.npz> <out feats_NEW.npz> --first 7 --last 7
Frames: inc_* are (n, 16 * windows, 2048) per video; I3D: i3d_* are (n, windows, 6, 400)."""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True)
    a = ap.parse_args()
    d = dict(np.load(a.src))
    fw, lw = int(d["first_window"]), int(d["windows"])
    if not fw <= a.first <= a.last <= lw:
        raise SystemExit(f"windows {a.first}-{a.last} not inside {fw}-{lw}")
    w0, w1 = a.first - fw, a.last - fw + 1
    out = {k: v for k, v in d.items() if k not in ("inc_gen", "inc_real", "i3d_gen", "i3d_real",
                                                  "first_window", "windows")}
    for k in ("inc_gen", "inc_real"):
        out[k] = d[k][:, 16 * w0:16 * w1]
    for k in ("i3d_gen", "i3d_real"):
        out[k] = d[k][:, w0:w1]
    out["first_window"], out["windows"] = a.first, a.last
    np.savez(a.dst, **out)
    print("wrote", a.dst, {k: v.shape for k, v in out.items() if hasattr(v, "shape")})


if __name__ == "__main__":
    main()
