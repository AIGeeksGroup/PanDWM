"""Stitch six generated camera views into the 256x4114 panorama canvas of the paper's instrument (2026-10-19).

Baselines generate six real cameras; the paper scores every method in the panoramic band, so their views go through
the OneBEV hard-seam mapping that made the recorded panoramas (dwm_pano.calibrated_stitch, projection model
"onebev_legacy", legacy offsets at native 900x1600, fixed Voronoi owners) onto exactly the canvas axes the model and
the recorded canvas use (half_canvas_pano_axes(half_size=(256, 2304), right_trim_px=494)).
  stitch_six(views, calib) -> (uint8 [T, 3, 256, 4114], bool [256, 4114] valid)
views: uint8 [T, 6, 3, h, w] in OpenDWM camera order (FL, F, FR, BR, B, BL); calib: per-camera sensor-to-ego 4x4
(OpenCV camera axes), raw OpenCV intrinsics at image_size (W, H), in the same order.
"""
import torch

from dwm_pano.calibrated_stitch import differentiable_hard_seam_stitch, half_canvas_pano_axes, prepare_gsplat_intrinsics
from dwm_pano.datasets.pano_projection import CAM_ORDER

ODWM_ORDER = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT")
HALF_SIZE, RIGHT_TRIM = (256, 2304), 494


@torch.no_grad()
def stitch_six(views, calib, device="cpu", views_order=ODWM_ORDER):
    if views.ndim != 5 or views.shape[1] != 6:
        raise ValueError(f"views must be [T, 6, 3, h, w], got {tuple(views.shape)}")
    order = [list(views_order).index(c) for c in CAM_ORDER]
    h, w = views.shape[-2:]
    imgs = views[:, order].to(device, torch.float32) / 255.0
    c2e = torch.as_tensor(calib["camera_transforms"], dtype=torch.float64)[order].to(device)
    K = torch.as_tensor(calib["camera_intrinsics"], dtype=torch.float64)[order]
    size = torch.as_tensor(calib["image_size"])[order]  # (W, H) per camera
    Kg = torch.stack([prepare_gsplat_intrinsics(K[i], (int(size[i, 1]), int(size[i, 0])), (h, w)) for i in range(6)])
    rows, cols = half_canvas_pano_axes(half_size=HALF_SIZE, right_trim_px=RIGHT_TRIM, dtype=torch.float64)
    pano, valid = differentiable_hard_seam_stitch(
        imgs, c2e, Kg.to(device), pano_rows=rows.to(device), pano_columns=cols.to(device),
        projection_model="onebev_legacy")
    out = (pano.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    return out, valid[0, 0].cpu() if valid.ndim == 4 else valid.cpu()
