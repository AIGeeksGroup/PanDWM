"""Quantify cross-tile depth discontinuities in a pano range map.

The DAC inference blends overlapping longitude tiles; per-tile monocular scale
bias would leave range jumps centered on the tile seams (odd multiples of half
the stride). This probe compares median range on both sides of every seam
azimuth against the same statistic at tile centers (control), over a row band.

Run:
    PYTHONPATH=src python src/dwm_pano/debug_pano_depth_seams.py \
        --range-npz <dir>/<timestamp>_range_m.npz
"""
import argparse

import numpy as np

from dwm_pano.datasets.pano_projection import PANO_W


def create_parser():
    parser = argparse.ArgumentParser(
        description="Seam-discontinuity probe for a pano range npz.")
    parser.add_argument(
        "--range-npz", type=str, required=True,
        help="npz with key 'range_m' (600x9600) from infer_pano_depth_dac.")
    parser.add_argument(
        "--tile-stride-deg", type=float, default=60.0,
        help="Tile stride used at inference; seams sit at half-stride odds.")
    parser.add_argument(
        "--window-deg", type=float, default=3.0,
        help="Width of the band sampled on each side of an azimuth.")
    parser.add_argument(
        "--gap-deg", type=float, default=1.5,
        help="Half-gap skipped around the azimuth (the Hann blend core).")
    parser.add_argument(
        "--row-start", type=int, default=350,
        help="First row of the evaluated band (below-horizon ground).")
    parser.add_argument(
        "--row-stop", type=int, default=600,
        help="One past the last row of the evaluated band.")
    return parser


def azimuth_to_column(azimuth_deg):
    return (azimuth_deg / 360.0 * PANO_W + PANO_W / 2.0) % PANO_W


def band_median(range_rows, center_deg, near_deg, far_deg):
    """Median range over columns [center+near, center+far) of the row band."""
    column_first = int(round(azimuth_to_column(center_deg + near_deg)))
    width = int(round((far_deg - near_deg) / 360.0 * PANO_W))
    columns = (column_first + np.arange(width)) % PANO_W
    return float(np.median(range_rows[:, columns]))


def jump_percent(range_rows, center_deg, gap_deg, window_deg):
    left = band_median(
        range_rows, center_deg, -gap_deg - window_deg, -gap_deg)
    right = band_median(
        range_rows, center_deg, gap_deg, gap_deg + window_deg)
    mean = (left + right) / 2.0
    return 100.0 * (right - left) / mean, left, right


def main():
    args = create_parser().parse_args()
    range_map = np.load(args.range_npz)["range_m"]
    range_rows = range_map[args.row_start:args.row_stop]

    tile_count = int(round(360.0 / args.tile_stride_deg))
    seam_azimuths = []
    control_azimuths = []
    for tile_index in range(tile_count):
        center = -180.0 + tile_index * args.tile_stride_deg
        seam_azimuths.append(center + args.tile_stride_deg / 2.0)
        control_azimuths.append(center)

    print(f"[seams] rows [{args.row_start},{args.row_stop}) "
          f"window={args.window_deg}deg gap={args.gap_deg}deg")
    seam_jumps = []
    for azimuth in seam_azimuths:
        jump, left, right = jump_percent(
            range_rows, azimuth, args.gap_deg, args.window_deg)
        seam_jumps.append(abs(jump))
        print(f"[seams] seam@{azimuth:+7.1f}deg  left={left:6.2f}m "
              f"right={right:6.2f}m  jump={jump:+6.1f}%")
    control_jumps = []
    for azimuth in control_azimuths:
        jump, left, right = jump_percent(
            range_rows, azimuth, args.gap_deg, args.window_deg)
        control_jumps.append(abs(jump))
        print(f"[seams] ctrl@{azimuth:+7.1f}deg  left={left:6.2f}m "
              f"right={right:6.2f}m  jump={jump:+6.1f}%")

    seam_mean = float(np.mean(seam_jumps))
    control_mean = float(np.mean(control_jumps))
    print(f"[seams] mean|jump| seams={seam_mean:.1f}% "
          f"controls={control_mean:.1f}% ratio={seam_mean / control_mean:.2f}")


if __name__ == "__main__":
    main()
