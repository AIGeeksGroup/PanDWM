"""DAC (Depth Any Camera, github yuliangguo/depth_any_camera @ 371ee299, weights dac_swinl_outdoor) metric range
for the 360 x 33.3 deg pano band, used to place decoder Gaussians (2026-10-18, report section 37).

The pano inference is dwm_pano/infer_pano_depth_dac.py's: resample to square-angular pixels near DAC's canonical
density (about 2765 x 256), run 90-degree longitude tiles every 60 degrees, Hann-blend, resize back to 600 x 9600.
DAC reads RGB, so for a generated frame the latents are VAE-decoded first and the two halves are put back on the
9600-wide pano grid (the 1152-column overlap is averaged).

The DAC range is the distance from the viewpoint of the pixel's content, so it is used as the range along the
decoder's exact content ray (from the owning camera's origin); no source-camera-z correction is applied.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from dwm_pano.datasets.pano_projection import PANO_H, PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE, PANO_W
from dwm_pano.gs_geometry import PANO_LAT_HALF_RAD
from dwm_pano.infer_pano_depth_dac import (
    column_to_azimuth_rad,
    compute_tile_start_columns,
    make_hann_window,
    predict_tile_range,
    resample_pano_to_square_angular,
)


def build_dac(dac_root, config_file, model_file, device):
    """infer_pano_depth_dac.build_dac_model without the backbone's ImageNet download (the nodes are offline): the
    encoder is built with pretrained=False and the DAC checkpoint then fills every weight (strict load)."""
    import json
    if dac_root not in sys.path:
        sys.path.append(dac_root)
    from dac.models.idisc import IDisc
    from dac.models.idisc_erp import IDiscERP
    with open(config_file, "r") as handle:
        config = json.load(handle)
    config["model"]["pixel_encoder"]["pretrained"] = False
    model = {"IDisc": IDisc, "IDiscERP": IDiscERP}[config["model_name"]].build(config)
    try:
        state = torch.load(model_file, map_location="cpu", weights_only=True)
    except Exception as error:                     # checkpoints with training state need the full unpickler
        print(f"[dac] weights_only load failed ({type(error).__name__}); loading the full checkpoint", flush=True)
        state = torch.load(model_file, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    return model.to(device).eval(), config


class DacRanger:
    def __init__(self, dac_root, config_file, model_file, device, tile_fov_deg=90.0, tile_stride_deg=60.0,
                 size_multiple=32):
        ops = os.path.join(dac_root, "dac", "models", "ops")
        if ops not in sys.path:
            sys.path.insert(0, ops)                    # the compiled MultiScaleDeformableAttention module
        self.model, self.config = build_dac(dac_root, config_file, model_file, device)
        self.device = device
        self.cano_density = self.config["cano_sz"][0] / np.pi
        self.tile_fov_deg, self.tile_stride_deg, self.size_multiple = tile_fov_deg, tile_stride_deg, size_multiple

    @torch.no_grad()
    def pano_range(self, pano):
        """(600, 9600, 3) float32 RGB in [0, 1] -> (600, 9600) float32 range in metres."""
        scaled, density = resample_pano_to_square_angular(pano, self.cano_density, 1.0, self.size_multiple)
        height, width = scaled.shape[:2]
        scale_back = density / self.cano_density
        starts, tile_width = compute_tile_start_columns(width, self.tile_fov_deg, self.tile_stride_deg,
                                                        self.size_multiple)
        hann = make_hann_window(tile_width)
        range_sum = np.zeros((height, width), dtype=np.float64)
        weight_sum = np.zeros(width, dtype=np.float64)
        for start in starts:
            columns = (start + np.arange(tile_width)) % width
            longitude = [column_to_azimuth_rad(start + 0.5, width),
                         column_to_azimuth_rad(start + tile_width - 0.5, width)]
            tile = predict_tile_range(self.model, self.config["model_name"], scaled[:, columns],
                                      [-PANO_LAT_HALF_RAD, PANO_LAT_HALF_RAD], longitude, self.device)
            range_sum[:, columns] += tile * scale_back * hann[None, :]
            weight_sum[columns] += hann
        import cv2
        return cv2.resize((range_sum / weight_sum[None, :]).astype(np.float32), dsize=(PANO_W, PANO_H),
                          interpolation=cv2.INTER_LINEAR)


def pano_from_halves(halves):
    """(2, 3, h, w) halves in [0, 1] (decoder layout) -> (600, 9600, 3) float32 pano; the overlap is averaged."""
    out = np.zeros((PANO_H, PANO_W, 3), dtype=np.float32)
    weight = np.zeros((1, PANO_W, 1), dtype=np.float32)
    for half, (c0, c1) in zip(halves, (PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE)):
        big = F.interpolate(half[None].float(), size=(PANO_H, c1 - c0), mode="bilinear", antialias=False,
                            align_corners=False)[0]
        out[:, c0:c1] += big.permute(1, 2, 0).cpu().numpy()
        weight[:, c0:c1] += 1.0
    return out / weight


def range_to_decoder_grid(range_full, height, width, device):
    """(600, 9600) range -> (2, height, width) on the decoder's half grids (area-averaged like the label loader)."""
    full = torch.from_numpy(np.ascontiguousarray(range_full))[None, None]
    halves = [F.interpolate(full[:, :, :, c0:c1], size=(height, width), mode="bilinear", antialias=True)[0, 0]
              for c0, c1 in (PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE)]
    return torch.stack(halves).to(device)


def relabel_gaussians(g, pred_range, new_range, valid, origins, dirs):
    """Move each Gaussian of a valid pixel along its ray to new_range (xyz offset dropped, scales follow the
    range ratio).  g (N, 14) in decoder order; pred_range / new_range / valid (V, H, W).  Returns (g', range')."""
    use = valid.reshape(-1)
    r_pred, r_new = pred_range.reshape(-1).float(), new_range.reshape(-1).float()
    out = g.clone()
    out[use, 0:3] = (origins[use] + dirs[use] * r_new[use, None]).to(g.dtype)
    ratio = (r_new / r_pred.clamp_min(1e-3)).clamp(0.25, 4.0)
    out[use, 11:14] = g[use, 11:14] * ratio[use, None].to(g.dtype)
    return out, torch.where(valid, new_range.float(), pred_range.float())
