"""tests_analyze.py for six-camera band features (cam_band_feats.py, inc [n, 32, 6, 2048]) (2026-10-20).

tests_analyze.py expects one Inception row per frame (inc [n, 32, 2048]). Here every frame has six camera images, so
FID pools all 6 cameras x 32 frames of a clip (FID-1 / FID-2: 6 x 16 of window 1 / 2), every image row tagged with its
clip's scene so the delete-one-scene jackknife, the scene bootstrap and the half-split floor drop all six cameras of a
scene together. i3d [n, 2, 6, 400] already has tests_analyze.py's shape (6 camera samples per window). Everything else
(point values, seed means, jackknife intervals, pairs, half-n floor) is tests_analyze.main unchanged (the module is
imported and its two functions replaced at run time; tests_analyze.py itself is not modified). Panorama features
(3-dim inc) are refused, so the two instruments cannot be mixed in one report.
  python3 cam_band_analyze.py --real feats_rec448.npz --arm ours=a.npz,b.npz,c.npz --arm uni=u.npz --pair ours:uni \
      --boot 1000 --out analysis.json
"""
import numpy as np

import tests_analyze as ta

_metric_rows_pano = ta.metric_rows


def metric_rows(feats, scene_id):
    inc, i3d = feats["inc"], feats["i3d"]
    if inc.ndim == 3:
        return _metric_rows_pano(feats, scene_id)
    if inc.ndim != 4 or inc.shape[1] != 32 or i3d.shape[1:] != (2, inc.shape[2], 400):
        raise ValueError(f"unexpected camera features inc {inc.shape} i3d {i3d.shape}")
    v = inc.shape[2]
    sc = np.array([scene_id[s] for s in feats["scene"]])
    return {
        "FID": ta.Rows(inc.reshape(-1, 2048), np.repeat(sc, 32 * v)),
        "FID-1": ta.Rows(inc[:, :16].reshape(-1, 2048), np.repeat(sc, 16 * v)),
        "FID-2": ta.Rows(inc[:, 16:].reshape(-1, 2048), np.repeat(sc, 16 * v)),
        "FVD-1": ta.Rows(i3d[:, 0].reshape(-1, 400), np.repeat(sc, v)),
        "FVD-2": ta.Rows(i3d[:, 1].reshape(-1, 400), np.repeat(sc, v)),
    }


_load_pano = ta.load


def load(path):
    out = _load_pano(path)
    if out["inc"].ndim != 4:
        raise ValueError(f"{path}: not six-camera band features (inc {out['inc'].shape})")
    return out


ta.metric_rows = metric_rows
ta.load = load

if __name__ == "__main__":
    ta.main()
