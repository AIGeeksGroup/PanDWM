"""Concatenate clip_feats.py feature files of disjoint clip sets, e.g. Dev-149 + scene-0018 = 150-scene val (2026-10-19).
  python3 merge_feats.py OUT.npz A.npz B.npz [...]
All parts must come from the same seed (or all be real features); test indices must not repeat.
"""
import os
import sys

import numpy as np


def main():
    out, parts = sys.argv[1], sys.argv[2:]
    z = [np.load(p, allow_pickle=False) for p in parts]
    seeds = {int(s) for q in z for s in q["seed"]}
    if len(seeds) != 1:
        raise RuntimeError(f"mixed seeds {seeds}")
    ti = np.concatenate([q["test_index"] for q in z])
    if len(set(ti.tolist())) != len(ti):
        raise RuntimeError("repeated test_index")
    tmp = out + ".tmp.npz"
    np.savez(tmp, inc=np.concatenate([q["inc"] for q in z]), i3d=np.concatenate([q["i3d"] for q in z]),
             test_index=ti, scene_token=np.concatenate([q["scene_token"] for q in z]),
             seed=np.concatenate([q["seed"] for q in z]),
             manifest="+".join(str(q["manifest"]) for q in z), arm_root="+".join(str(q["arm_root"]) for q in z))
    os.replace(tmp, out)
    print(f"MERGED {out} clips={len(ti)} seed={seeds.pop()} from {parts}")


if __name__ == "__main__":
    main()
