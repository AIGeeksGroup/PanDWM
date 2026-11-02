"""CPU tests for dwm_pano/d2v2_make_pairs.py helpers (2026-10-31, D2V2_PLAN_20261031.md).

    PYTHONPATH=src python3 -m pytest -q tests/test_d2v2_pairs.py   (needs the dwm / dwm_pano imports; HPC)
"""

import numpy as np
from pyquaternion import Quaternion

from dwm_pano.d2v2_make_pairs import DY_CHOICES, _pose, d2v2_shift_infos, draw_shift


def _info(rng):
    return {"ego2global_translation": rng.normal(size=3).tolist(),
            "ego2global_rotation": Quaternion(axis=[0, 0, 1], angle=float(rng.uniform(-3, 3))).elements.tolist(),
            "lidar2ego_translation": [0.9, 0.0, 1.8],
            "lidar2ego_rotation": Quaternion(axis=[0, 0, 1], angle=-1.57).elements.tolist(), "token": "x"}


def test_shift_moves_the_ego_along_its_own_y_and_keeps_the_lidar_frame():
    rng = np.random.default_rng(0)
    infos = [_info(rng) for _ in range(3)]
    out = d2v2_shift_infos(infos, 2.5)
    for a, b in zip(infos, out):
        ea = _pose(a["ego2global_translation"], a["ego2global_rotation"])
        eb = _pose(b["ego2global_translation"], b["ego2global_rotation"])
        local = np.linalg.inv(ea) @ eb
        assert np.allclose(local[:3, :3], np.eye(3)) and np.allclose(local[:3, 3], [0.0, 2.5, 0.0])
        la = ea @ _pose(a["lidar2ego_translation"], a["lidar2ego_rotation"])
        lb = eb @ _pose(b["lidar2ego_translation"], b["lidar2ego_rotation"])
        assert np.allclose(la, lb)                                       # LiDAR points / gt_boxes stay in place
        assert b["token"] == "x" and a["ego2global_translation"] != b["ego2global_translation"]


def test_draw_shift_is_deterministic_and_covers_both_signs():
    draws = [draw_shift(7, i, 0.15) for i in range(2000)]
    assert draws == [draw_shift(7, i, 0.15) for i in range(2000)]
    plain = sum(m == "plain" for m, _ in draws) / len(draws)
    assert 0.10 < plain < 0.20
    dys = [d for m, d in draws if m == "roundtrip"]
    assert set(abs(d) for d in dys) == set(DY_CHOICES) and min(dys) < 0 < max(dys)
