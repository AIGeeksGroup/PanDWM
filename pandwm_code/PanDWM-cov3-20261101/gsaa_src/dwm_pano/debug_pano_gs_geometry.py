"""Zero-training geometry sanity for the pano -> 3D Gaussian -> perspective pipeline.

Check A: pano pixel -> ego direction (inverse map, defined HERE) -> the authoritative
         forward map `project_ego_points_to_pano` must return the source pixel.
Check B: perspective ground truth built by direct equirect resampling of the pano
         through the authoritative forward map (exact, zero parallax at pano center).
Check C: (--render, needs gsplat + CUDA) back-project every strided pano pixel to one
         Gaussian at its ray range, rasterize the same perspective cameras with gsplat,
         compare against Check B by PSNR.

From the pano center the rendered image is range-invariant, so Check C validates the
ray and camera conventions with a constant range and NO depth data. Real range maps
plug in later via --depth-npz (key "range_m", meters along the unit ray, 600x9600).

Run checks A+B on CPU:
    PYTHONPATH=src python src/dwm_pano/debug_pano_gs_geometry.py \
        --pano-path data/val_rgb/<timestamp>.jpg --out-dir /data/<...>/pano_gs_sanity
Run all checks on a GPU node (after `pip install gsplat`): append --render
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from dwm_pano.datasets.pano_projection import (
    PANO_H, PANO_W, project_ego_points_to_pano,
)
from dwm_pano.gs_geometry import (
    build_virtual_camera, pano_pixel_centers_to_ego_directions,
    sample_pano_as_ground_truth, virtual_pixel_directions_ego,
)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Pano geometry sanity: ray round-trip, resampled ground "
        "truth, and optional gsplat re-rendering of back-projected Gaussians.")
    parser.add_argument(
        "--pano-path", type=str, required=True,
        help="Path to one full 9600x600 RGB pano jpg.")
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Directory for result images and the printed report.")
    parser.add_argument(
        "--depth-npz", type=str, default="",
        help="Optional npz with key 'range_m' (600x9600); empty uses "
        "--constant-range-m.")
    parser.add_argument(
        "--constant-range-m", type=float, default=30.0,
        help="Range along each unit ray when no depth npz is given.")
    parser.add_argument(
        "--virtual-yaws-deg", type=float, nargs="+",
        default=[0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
        help="Azimuths of the virtual perspective cameras (0 = CAM_FRONT).")
    parser.add_argument(
        "--virtual-fov-deg", type=float, default=60.0,
        help="Horizontal field of view of each virtual camera.")
    parser.add_argument(
        "--virtual-height", type=int, default=300,
        help="Virtual camera image height in pixels.")
    parser.add_argument(
        "--virtual-width", type=int, default=800,
        help="Virtual camera image width in pixels.")
    parser.add_argument(
        "--lateral-shifts-m", type=float, nargs="+", default=[1.0],
        help="Extra render-only camera shifts along ego +Y (left), meters.")
    parser.add_argument(
        "--pixel-stride", type=int, default=4,
        help="Pano pixel stride for round-trip grid and Gaussian creation.")
    parser.add_argument(
        "--projection-sample-radius-m", type=float, default=10.0,
        help="Radius used to turn directions into points for the forward map.")
    parser.add_argument(
        "--round-trip-tolerance-px", type=float, default=1e-3,
        help="Maximum allowed round-trip error before hard failure.")
    parser.add_argument(
        "--gaussian-opacity", type=float, default=1.0,
        help="Opacity assigned to every back-projected Gaussian.")
    parser.add_argument(
        "--gaussian-scale-multiplier", type=float, default=1.0,
        help="Multiplier on the angular-footprint-derived Gaussian scale.")
    parser.add_argument(
        "--near-plane", type=float, default=0.1,
        help="gsplat near plane in meters.")
    parser.add_argument(
        "--far-plane", type=float, default=400.0,
        help="gsplat far plane in meters.")
    parser.add_argument(
        "--alpha-threshold", type=float, default=0.5,
        help="Alpha above which a rendered pixel counts as covered.")
    parser.add_argument(
        "--render", action="store_true",
        help="Also run Check C (requires gsplat and a CUDA device).")
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device for the gsplat render check.")
    return parser


def run_round_trip_check(stride, sample_radius_m, tolerance_px):
    u_axis = np.arange(0, PANO_W, stride)
    v_axis = np.arange(0, PANO_H, stride)
    u_grid, v_grid = np.meshgrid(u_axis, v_axis, indexing="xy")
    directions = pano_pixel_centers_to_ego_directions(
        u_grid.reshape(-1), v_grid.reshape(-1))
    points = directions * sample_radius_m
    tx, ty, valid = project_ego_points_to_pano(points)
    if not bool(valid.all()):
        raise RuntimeError(
            f"round-trip: {int((~valid).sum())} directions failed to project")
    expected_tx = u_grid.reshape(-1) + 0.5
    expected_ty = v_grid.reshape(-1) + 0.5
    wrapped_dx = (tx - expected_tx + PANO_W / 2.0) % PANO_W
    dx = wrapped_dx - PANO_W / 2.0
    dy = ty - expected_ty
    max_dx = float(np.abs(dx).max())
    max_dy = float(np.abs(dy).max())
    print(f"[check A] round-trip max |dx|={max_dx:.2e}px |dy|={max_dy:.2e}px "
          f"over {directions.shape[0]} pixels")
    if max_dx > tolerance_px or max_dy > tolerance_px:
        raise RuntimeError("round-trip error exceeds tolerance")


def load_range_map(depth_npz_path, constant_range_m):
    if depth_npz_path == "":
        return np.full((PANO_H, PANO_W), constant_range_m, dtype=np.float64)
    return np.load(depth_npz_path)["range_m"].astype(np.float64)


def back_project_to_gaussians(pano_float, range_map, stride, opacity,
                              scale_multiplier):
    """One Gaussian per strided pano pixel: mean on the ray, color from RGB."""
    u_axis = np.arange(0, PANO_W, stride)
    v_axis = np.arange(0, PANO_H, stride)
    u_grid, v_grid = np.meshgrid(u_axis, v_axis, indexing="xy")
    u_flat = u_grid.reshape(-1)
    v_flat = v_grid.reshape(-1)
    directions = pano_pixel_centers_to_ego_directions(u_flat, v_flat)
    ranges = range_map[v_flat, u_flat]
    means = directions * ranges[:, None]
    colors = pano_float[v_flat, u_flat]
    pixel_angle = 2.0 * np.pi / PANO_W
    scales = ranges * pixel_angle * stride * scale_multiplier
    scales = np.repeat(scales[:, None], 3, axis=1)
    quats = np.zeros((means.shape[0], 4))
    quats[:, 0] = 1.0
    opacities = np.full(means.shape[0], opacity)
    return means, colors, scales, quats, opacities


def render_with_gsplat(means, colors, scales, quats, opacities, c2w_list,
                       intrinsics_list, height, width, near_plane, far_plane,
                       device_name):
    import gsplat
    device = torch.device(device_name)
    viewmats = torch.linalg.inv(
        torch.from_numpy(np.stack(c2w_list)).float()).to(device)
    intrinsics = torch.from_numpy(np.stack(intrinsics_list)).float().to(device)
    rendered, alphas, _ = gsplat.rasterization(
        means=torch.from_numpy(means).float().to(device),
        quats=torch.from_numpy(quats).float().to(device),
        scales=torch.from_numpy(scales).float().to(device),
        opacities=torch.from_numpy(opacities).float().to(device),
        colors=torch.from_numpy(colors).float().to(device),
        viewmats=viewmats,
        Ks=intrinsics,
        width=width,
        height=height,
        near_plane=near_plane,
        far_plane=far_plane,
        render_mode="RGB+D")
    rgb = rendered[..., :3].clamp(0.0, 1.0).cpu().numpy()
    alpha = alphas[..., 0].cpu().numpy()
    return rgb, alpha


def peak_signal_to_noise_ratio(prediction, target, mask):
    squared_error = (prediction - target) ** 2
    mean_squared_error = squared_error[mask].mean()
    return float(10.0 * np.log10(1.0 / mean_squared_error))


def save_image(path, image_float):
    array = (np.clip(image_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(array).save(path)


def main():
    args = create_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pano_image = Image.open(args.pano_path)
    pano_float = np.asarray(pano_image, dtype=np.float64) / 255.0
    if pano_float.shape != (PANO_H, PANO_W, 3):
        raise RuntimeError(
            f"expected full {PANO_W}x{PANO_H} pano, got {pano_float.shape}")

    run_round_trip_check(
        args.pixel_stride, args.projection_sample_radius_m,
        args.round_trip_tolerance_px)

    ground_truths = []
    valid_masks = []
    center_cameras = []
    for yaw_deg in args.virtual_yaws_deg:
        c2w, intrinsics = build_virtual_camera(
            yaw_deg, args.virtual_height, args.virtual_width,
            args.virtual_fov_deg, 0.0)
        directions = virtual_pixel_directions_ego(
            c2w, intrinsics, args.virtual_height, args.virtual_width)
        ground_truth, valid_mask = sample_pano_as_ground_truth(
            pano_float, directions, yaw_deg, args.projection_sample_radius_m)
        valid_fraction = float(valid_mask.mean())
        print(f"[check B] yaw={yaw_deg:6.1f} gt-valid={valid_fraction:.3f}")
        save_image(
            os.path.join(args.out_dir, f"gt_yaw{yaw_deg:.0f}.png"),
            ground_truth)
        ground_truths.append(ground_truth)
        valid_masks.append(valid_mask)
        center_cameras.append((c2w, intrinsics))

    if not args.render:
        print("[check C] skipped (pass --render on a GPU node with gsplat)")
        return

    range_map = load_range_map(args.depth_npz, args.constant_range_m)
    means, colors, scales, quats, opacities = back_project_to_gaussians(
        pano_float, range_map, args.pixel_stride, args.gaussian_opacity,
        args.gaussian_scale_multiplier)
    print(f"[check C] gaussians={means.shape[0]} "
          f"range=[{range_map.min():.1f},{range_map.max():.1f}]m")

    c2w_list = [camera[0] for camera in center_cameras]
    intrinsics_list = [camera[1] for camera in center_cameras]
    for yaw_deg in args.virtual_yaws_deg:
        for shift_m in args.lateral_shifts_m:
            c2w, intrinsics = build_virtual_camera(
                yaw_deg, args.virtual_height, args.virtual_width,
                args.virtual_fov_deg, shift_m)
            c2w_list.append(c2w)
            intrinsics_list.append(intrinsics)

    rgb, alpha = render_with_gsplat(
        means, colors, scales, quats, opacities, c2w_list, intrinsics_list,
        args.virtual_height, args.virtual_width, args.near_plane,
        args.far_plane, args.device)

    view_count = len(args.virtual_yaws_deg)
    for view_index, yaw_deg in enumerate(args.virtual_yaws_deg):
        covered = alpha[view_index] > args.alpha_threshold
        comparison_mask = np.logical_and(covered, valid_masks[view_index])
        comparison_mask = np.repeat(comparison_mask[:, :, None], 3, axis=2)
        psnr = peak_signal_to_noise_ratio(
            rgb[view_index], ground_truths[view_index], comparison_mask)
        coverage = float(covered.mean())
        print(f"[check C] yaw={yaw_deg:6.1f} covered-psnr={psnr:6.2f}dB "
              f"coverage={coverage:.3f}")
        save_image(
            os.path.join(args.out_dir, f"render_yaw{yaw_deg:.0f}.png"),
            rgb[view_index])
        side_by_side = np.concatenate(
            [ground_truths[view_index], rgb[view_index]], axis=0)
        save_image(
            os.path.join(args.out_dir, f"compare_yaw{yaw_deg:.0f}.png"),
            side_by_side)

    shifted_index = view_count
    for yaw_deg in args.virtual_yaws_deg:
        for shift_m in args.lateral_shifts_m:
            name = f"render_yaw{yaw_deg:.0f}_shift{shift_m:+.1f}m.png"
            save_image(os.path.join(args.out_dir, name), rgb[shifted_index])
            shifted_index += 1


if __name__ == "__main__":
    main()
