"""CPU unit tests for dwm_pano_4d.geometry (no cluster data needed).

  PYTHONPATH=src python3 -m pytest -q tests  (or: python3 tests/test_geometry.py)
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dwm_pano_4d.geometry import (  # noqa: E402
    box_pose, points_in_boxes, quat_mul, relative_ego_transform,
    rotation_to_quat, transform_gaussians)


def quat_to_rotation(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def random_rotations(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(n, 4, generator=g, dtype=torch.float64), dim=-1)
    return quat_to_rotation(q)


def test_rotation_quat_roundtrip_including_180_degrees():
    rotations = random_rotations(500)
    flips = torch.stack([torch.diag(torch.tensor(d, dtype=torch.float64))
                         for d in ([1, -1, -1], [-1, 1, -1], [-1, -1, 1])])
    rotations = torch.cat([rotations, flips, torch.eye(3, dtype=torch.float64)[None]])
    back = quat_to_rotation(rotation_to_quat(rotations))
    assert torch.allclose(back, rotations, atol=1e-9)


def test_quat_mul_matches_matrix_product():
    a, b = random_rotations(50, 1), random_rotations(50, 2)
    qa, qb = rotation_to_quat(a), rotation_to_quat(b)
    assert torch.allclose(quat_to_rotation(quat_mul(qa, qb)), a @ b, atol=1e-9)


def test_transform_gaussians_moves_means_and_orientations():
    g = torch.zeros(10, 14, dtype=torch.float64)
    g[:, 0:3] = torch.randn(10, 3, dtype=torch.float64)
    g[:, 7:11] = rotation_to_quat(random_rotations(10, 3))
    g[:, 11:14] = torch.rand(10, 3, dtype=torch.float64)
    yaw = 0.7
    m = torch.eye(4, dtype=torch.float64)
    m[:3, :3] = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0],
                              [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    m[:3, 3] = torch.tensor([3.0, -2.0, 0.5])
    out = transform_gaussians(g, m)
    assert torch.allclose(out[:, 0:3], g[:, 0:3] @ m[:3, :3].T + m[:3, 3])
    assert torch.allclose(quat_to_rotation(out[:, 7:11]),
                          m[:3, :3] @ quat_to_rotation(g[:, 7:11]), atol=1e-9)
    assert torch.equal(out[:, 3:7], g[:, 3:7]) and torch.equal(out[:, 11:14], g[:, 11:14])


def test_relative_ego_transform_composes():
    def pose(yaw, t):
        p = torch.eye(4, dtype=torch.float64)
        p[:3, :3] = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0],
                                  [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        p[:3, 3] = torch.tensor(t)
        return p
    e2g_s, e2g_t = pose(0.3, [10.0, 5.0, 0.0]), pose(0.5, [14.0, 6.0, 0.0])
    point_s = torch.tensor([2.0, 1.0, 0.5, 1.0], dtype=torch.float64)
    world = e2g_s @ point_s
    point_t = relative_ego_transform(e2g_s, e2g_t) @ point_s
    assert torch.allclose(e2g_t @ point_t, world)


def test_points_in_boxes_assigns_owner_and_margin():
    centers = torch.tensor([[5.0, 0.0, 1.0], [-5.0, 2.0, 1.0]])
    rotations = torch.stack([torch.eye(3), torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])])
    half = torch.tensor([[2.0, 1.0, 0.8], [2.0, 1.0, 0.8]])
    points = torch.tensor([[5.0, 0.0, 1.0], [6.9, 0.0, 1.0], [7.2, 0.0, 1.0],
                           [-5.0, 3.9, 1.0], [-5.0, 2.0, 5.0], [0.0, 0.0, 0.0]])
    owner = points_in_boxes(points, centers, rotations, half, margin=0.0)
    assert owner.tolist() == [0, 0, -1, 1, -1, -1]
    owner = points_in_boxes(points, centers, rotations, half, margin=0.3)
    assert owner.tolist()[:3] == [0, 0, 0]
    owner = points_in_boxes(points, centers, rotations, half,
                            valid=torch.tensor([False, True]), margin=0.0)
    assert owner.tolist() == [-1, -1, -1, 1, -1, -1]


def test_box_motion_carries_a_point_rigidly():
    c_s, c_t = torch.tensor([10.0, 0.0, 1.0]), torch.tensor([14.0, 1.0, 1.0])
    yaw = 0.4
    r_t = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0.0],
                        [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    move = box_pose(c_t, r_t) @ torch.linalg.inv(box_pose(c_s, torch.eye(3)))
    corner_s = torch.tensor([11.0, 0.5, 1.0, 1.0])
    corner_t = move @ corner_s
    assert torch.allclose(corner_t[:3], c_t + r_t @ torch.tensor([1.0, 0.5, 0.0]), atol=1e-6)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
