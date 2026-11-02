"""Metric range estimation for a full 9600x600 pano via Depth-Any-Camera (DAC).

The pano is resampled once to square-angular pixels at the DAC canonical ERP
density, split into overlapping longitude tiles, run through the DAC model,
blended with Hann windows, and upsampled back to the native pano grid. The
output npz key "range_m" holds Euclidean meters along each unit pano ray
(600x9600), matching --depth-npz of debug_pano_gs_geometry.py.

One-time setup:
    HF_HUB_DISABLE_XET=1 hf download yuliangguo/depth-any-camera \
        dac_swinl_outdoor.pt dac_swinl_outdoor.json \
        --local-dir <dac_root>/checkpoints
    cd <dac_root>/dac/models/ops && TORCH_CUDA_ARCH_LIST=9.0 \
        python setup.py build install
Run (GPU node):
    PYTHONPATH=src python src/dwm_pano/infer_pano_depth_dac.py \
        --pano-path data/val_rgb/<timestamp>.jpg --dac-root <dac_root> \
        --config-file <dac_root>/checkpoints/dac_swinl_outdoor.json \
        --model-file <dac_root>/checkpoints/dac_swinl_outdoor.pt \
        --out-dir <dir>
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from dwm_pano.datasets.pano_projection import PANO_H, PANO_W
from dwm_pano.gs_geometry import PANO_LAT_HALF_RAD

# Normalization constants tied to the DAC checkpoints (demo_dac_outdoor.py).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_parser():
    parser = argparse.ArgumentParser(
        description="DAC metric range estimation on a full RGB pano.")
    parser.add_argument(
        "--pano-path", type=str, required=True,
        help="Path to one full 9600x600 RGB pano jpg.")
    parser.add_argument(
        "--dac-root", type=str, required=True,
        help="Root of the depth_any_camera repository (for the dac package).")
    parser.add_argument(
        "--config-file", type=str, required=True,
        help="DAC checkpoint json (e.g. dac_swinl_outdoor.json).")
    parser.add_argument(
        "--model-file", type=str, required=True,
        help="DAC checkpoint weights (e.g. dac_swinl_outdoor.pt).")
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Directory for the range npz and visualization png.")
    parser.add_argument(
        "--tile-fov-deg", type=float, default=90.0,
        help="Longitude window of each ERP tile fed to the model.")
    parser.add_argument(
        "--tile-stride-deg", type=float, default=60.0,
        help="Longitude stride between tile centers.")
    parser.add_argument(
        "--forward-scale", type=float, default=1.0,
        help="Tile density relative to the DAC canonical ERP density.")
    parser.add_argument(
        "--size-multiple", type=int, default=32,
        help="Round tile height and width down to this multiple.")
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device for model inference.")
    parser.add_argument(
        "--vis-max-range-m", type=float, default=80.0,
        help="Range clip for the colormap visualization.")
    parser.add_argument(
        "--report-percentiles", type=float, nargs="+",
        default=[5.0, 50.0, 95.0],
        help="Range percentiles printed in the numeric report.")
    parser.add_argument(
        "--report-edge-rows", type=int, default=30,
        help="Row count for the bottom/top median report lines.")
    parser.add_argument(
        "--align-tile-scales", action="store_true",
        help="Rescale each tile to the mosaic median over the overlap.")
    return parser


def build_dac_model(dac_root, config_file, model_file, device):
    sys.path.append(dac_root)
    from dac.models.idisc import IDisc
    from dac.models.idisc_erp import IDiscERP
    model_classes = {
        "IDisc": IDisc,
        "IDiscERP": IDiscERP,
    }
    with open(config_file, "r") as config_handle:
        config = json.load(config_handle)
    model_class = model_classes[config["model_name"]]
    model = model_class.build(config)
    model.load_pretrained(model_file)
    model = model.to(device)
    model.eval()
    return model, config


def resample_pano_to_square_angular(pano_float, cano_density, forward_scale,
                                    size_multiple):
    """Resize to square-angular pixels near the canonical ERP density."""
    latitude_span = 2.0 * PANO_LAT_HALF_RAD
    unrounded_height = latitude_span * cano_density * forward_scale
    scaled_height = round(unrounded_height / size_multiple) * size_multiple
    angular_density = scaled_height / latitude_span
    scaled_width = round(angular_density * 2.0 * np.pi)
    scaled_pano = cv2.resize(
        pano_float, dsize=(scaled_width, scaled_height),
        interpolation=cv2.INTER_AREA)
    return scaled_pano, angular_density


def compute_tile_start_columns(scaled_width, tile_fov_deg, tile_stride_deg,
                               size_multiple):
    unrounded_width = scaled_width * tile_fov_deg / 360.0
    tile_width = int(unrounded_width / size_multiple) * size_multiple
    stride_columns = round(scaled_width * tile_stride_deg / 360.0)
    if tile_width <= stride_columns:
        raise RuntimeError("tile window must be wider than the tile stride")
    tile_count = math.ceil(scaled_width / stride_columns)
    start_columns = []
    for tile_index in range(tile_count):
        center_column = tile_index * stride_columns
        start_column = (center_column - tile_width // 2) % scaled_width
        start_columns.append(start_column)
    return start_columns, tile_width


def column_to_azimuth_rad(column, scaled_width):
    return (column - scaled_width / 2.0) / scaled_width * 2.0 * np.pi


def make_hann_window(tile_width):
    phase = 2.0 * np.pi * (np.arange(tile_width) + 0.5) / tile_width
    return 0.5 - 0.5 * np.cos(phase)


def predict_tile_range(model, model_name, tile_float, latitude_range,
                       longitude_range, device):
    tile_tensor = TF.to_tensor(tile_float.astype(np.float32))
    image = TF.normalize(tile_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    image = image.unsqueeze(0).to(device)
    # Only the ERP model variant conditions on angular ranges (demo scripts).
    with torch.no_grad():
        if model_name == "IDiscERP":
            latitude = torch.tensor(
                [latitude_range], dtype=torch.float32, device=device)
            longitude = torch.tensor(
                [longitude_range], dtype=torch.float32, device=device)
            prediction, _, _ = model(image, latitude, longitude)
        else:
            prediction, _, _ = model(image)
    return prediction[0, 0].float().cpu().numpy()


def main():
    args = create_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    model, config = build_dac_model(
        args.dac_root, args.config_file, args.model_file, device)
    cano_density = config["cano_sz"][0] / np.pi

    pano_image = Image.open(args.pano_path)
    pano_float = np.asarray(pano_image, dtype=np.float32) / 255.0
    if pano_float.shape != (PANO_H, PANO_W, 3):
        raise RuntimeError(
            f"expected full {PANO_W}x{PANO_H} pano, got {pano_float.shape}")

    scaled_pano, angular_density = resample_pano_to_square_angular(
        pano_float, cano_density, args.forward_scale, args.size_multiple)
    scaled_height, scaled_width = scaled_pano.shape[:2]
    prediction_scale = angular_density / cano_density
    print(f"[dac] scaled pano {scaled_width}x{scaled_height} "
          f"density={angular_density:.1f}px/rad "
          f"scale-back={prediction_scale:.5f}")

    start_columns, tile_width = compute_tile_start_columns(
        scaled_width, args.tile_fov_deg, args.tile_stride_deg,
        args.size_multiple)
    latitude_range = [-PANO_LAT_HALF_RAD, PANO_LAT_HALF_RAD]
    hann_window = make_hann_window(tile_width)

    range_sum = np.zeros((scaled_height, scaled_width), dtype=np.float64)
    weight_sum = np.zeros(scaled_width, dtype=np.float64)
    for start_column in start_columns:
        column_indices = (start_column + np.arange(tile_width)) % scaled_width
        tile = scaled_pano[:, column_indices]
        azimuth_first = column_to_azimuth_rad(
            start_column + 0.5, scaled_width)
        azimuth_last = column_to_azimuth_rad(
            start_column + tile_width - 0.5, scaled_width)
        longitude_range = [azimuth_first, azimuth_last]
        tile_range = predict_tile_range(
            model, config["model_name"], tile, latitude_range,
            longitude_range, device)
        tile_range = tile_range * prediction_scale
        if args.align_tile_scales:
            overlap_mask = weight_sum[column_indices] > 0.0
            if overlap_mask.any():
                overlap_columns = column_indices[overlap_mask]
                mosaic_sum = range_sum[:, overlap_columns]
                mosaic_weight = weight_sum[overlap_columns][None, :]
                mosaic = mosaic_sum / mosaic_weight
                ratio = float(np.median(mosaic / tile_range[:, overlap_mask]))
                tile_range = tile_range * ratio
                print(f"[dac] tile@{start_column:4d} scale-align "
                      f"x{ratio:.4f}")
        print(f"[dac] tile@{start_column:4d} range "
              f"[{tile_range.min():.2f},{tile_range.max():.2f}]m")
        range_sum[:, column_indices] += tile_range * hann_window[None, :]
        weight_sum[column_indices] += hann_window

    range_scaled = range_sum / weight_sum[None, :]
    range_native = cv2.resize(
        range_scaled, dsize=(PANO_W, PANO_H),
        interpolation=cv2.INTER_LINEAR)
    range_native = range_native.astype(np.float32)

    pano_name = os.path.splitext(os.path.basename(args.pano_path))[0]
    npz_path = os.path.join(args.out_dir, f"{pano_name}_range_m.npz")
    np.savez_compressed(npz_path, range_m=range_native)

    normalized_range = np.clip(range_native / args.vis_max_range_m, 0.0, 1.0)
    range_visualization = cv2.applyColorMap(
        (normalized_range * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    visualization_path = os.path.join(
        args.out_dir, f"{pano_name}_range_vis.png")
    cv2.imwrite(visualization_path, range_visualization)

    percentile_values = np.percentile(range_native, args.report_percentiles)
    summary_parts = []
    for level, value in zip(args.report_percentiles, percentile_values):
        summary_parts.append(f"p{level:g}={value:.2f}")
    bottom_rows = range_native[-args.report_edge_rows:]
    top_rows = range_native[:args.report_edge_rows]
    print("[dac] range percentiles (m): " + " ".join(summary_parts))
    print(f"[dac] bottom-{args.report_edge_rows}-rows median "
          f"{float(np.median(bottom_rows)):.2f} m (near road, expect a few)")
    print(f"[dac] top-{args.report_edge_rows}-rows median "
          f"{float(np.median(top_rows)):.2f} m (expect far)")
    print(f"[dac] saved {npz_path}")


if __name__ == "__main__":
    main()
