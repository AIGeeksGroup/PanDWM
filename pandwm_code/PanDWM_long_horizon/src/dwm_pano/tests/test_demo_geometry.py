"""Geometry guards for the end-to-end GS demo."""
import torch

from dwm_pano.demo_world_model import (
    NUSCENES_CAM_ORDER,
    nuscenes_camera_stack,
)


def test_demo_converts_and_resizes_raw_nuscenes_intrinsics():
    cameras = {}
    for index, name in enumerate(NUSCENES_CAM_ORDER):
        cameras[name] = {
            "sensor2ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "sensor2ego_translation": [float(index), 0.0, 1.5],
            "camera_intrinsics": [
                [1000.0, 0.0, 800.0 + index],
                [0.0, 1000.0, 450.0 + index],
                [0.0, 0.0, 1.0],
            ],
        }

    transforms, intrinsics = nuscenes_camera_stack(
        cameras, torch.device("cpu"), target_size=(360, 640))
    assert transforms.shape == (6, 4, 4)
    assert intrinsics.shape == (6, 3, 3)
    assert transforms[:, 0, 3].tolist() == list(map(float, range(6)))
    assert torch.allclose(
        intrinsics[:, 0, 2],
        (torch.arange(6, dtype=torch.float32) + 800.5) * 0.4)
    assert torch.allclose(
        intrinsics[:, 1, 2],
        (torch.arange(6, dtype=torch.float32) + 450.5) * 0.4)
