"""Box extraction must agree with the repo's own box-corner convention.

Needs the PanDWM-gs-aa-ft ``src`` on the path (for dwm_pano); runs on CPU:
  PYTHONPATH=src:../PanDWM-gs-aa-ft/src python3 tests/test_dataset_boxes.py
"""

import sys
import types
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE.parent / "PanDWM-gs-aa-ft" / "src"))
# pano_via_stitch imports an external stitcher at module level; stub it.
for name in ("omninwm", "omninwm.utils", "omninwm.utils.pano_hardseam"):
    module = sys.modules.setdefault(name, types.ModuleType(name))
    module.__path__ = []
sys.modules["omninwm.utils.pano_hardseam"].stitch_hardseam = lambda *a, **k: None

from dwm_pano.datasets.pano_via_stitch import _ego_corners_from_boxes  # noqa: E402
from dwm_pano_4d.dataset import frame_boxes_in_ego  # noqa: E402


def synthetic_info():
    return {
        "gt_boxes": np.array([[10.0, 2.0, -0.5, 4.5, 1.9, 1.6, 0.3],
                              [-6.0, -3.0, -0.8, 0.6, 0.7, 1.8, -1.2],
                              [3.0, 8.0, -1.0, 0.5, 0.5, 1.0, 0.0]]),
        "gt_box_ids": ["car-a", "ped-b", "cone-c"],
        "gt_names": ["car", "pedestrian", "traffic_cone"],
        "lidar2ego_rotation": list(Quaternion(axis=[0, 0, 1], angle=np.pi / 2).elements),
        "lidar2ego_translation": [0.94, 0.0, 1.84],
    }


def test_centers_rotations_match_repo_corners():
    info = synthetic_info()
    tokens, centers, rotations, half = frame_boxes_in_ego(
        info, {"car", "pedestrian"})
    assert tokens == ["car-a", "ped-b"]                 # cone filtered out
    corners = _ego_corners_from_boxes(
        info["gt_boxes"][:2], Quaternion(info["lidar2ego_rotation"]),
        np.asarray(info["lidar2ego_translation"]))
    for i in range(2):
        assert np.allclose(corners[i].mean(axis=0), centers[i], atol=1e-4)
        local = (corners[i] - centers[i]) @ rotations[i]   # ego -> box axes
        assert np.allclose(np.abs(local), half[i][None], atol=1e-4)
        assert np.allclose(rotations[i].T @ rotations[i], np.eye(3), atol=1e-9)


def test_empty_and_unlisted_classes():
    info = synthetic_info()
    assert frame_boxes_in_ego(info, {"bus"})[0] == []
    info["gt_boxes"] = np.zeros((0, 7))
    info["gt_box_ids"], info["gt_names"] = [], []
    assert frame_boxes_in_ego(info, {"car"})[0] == []


if __name__ == "__main__":
    test_centers_rotations_match_repo_corners()
    test_empty_and_unlisted_classes()
    print("ok dataset box tests")
