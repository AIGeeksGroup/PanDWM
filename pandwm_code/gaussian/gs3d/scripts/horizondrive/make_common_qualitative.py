"""Render fixed, non-cherry-picked qualitative panels for the paper."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def parse_item(value: str):
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("item must be HORIZON_INDEX:ANCHOR_US")
    return int(parts[0]), int(parts[1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-real", required=True)
    parser.add_argument("--horizon-fake", required=True)
    parser.add_argument("--pandwm-real", required=True)
    parser.add_argument("--pandwm-fake", required=True)
    parser.add_argument("--epona-fake")
    parser.add_argument("--gem-fake")
    parser.add_argument("--vista-fake")
    parser.add_argument("--omni-fake")
    parser.add_argument("--item", action="append", type=parse_item,
                        required=True)
    parser.add_argument("--future-indices", default="7,15,23,31")
    parser.add_argument("--condition-frames", type=int, default=3)
    parser.add_argument("--target-start", type=int)
    parser.add_argument("--target-frames", type=int)
    parser.add_argument("--epona-context-frames", type=int, default=10)
    parser.add_argument("--gem-context-frames", type=int, default=1)
    parser.add_argument("--vista-context-frames", type=int, default=1)
    parser.add_argument("--omni-context-frames", type=int, default=0)
    parser.add_argument("--time-label-offset-frames", type=int, default=1)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-gt-abs-diff", type=int, default=1)
    parser.add_argument("--max-gt-diff-fraction", type=float, default=1e-5)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_prefix(directory: Path, prefix: str):
    matches = sorted(directory.glob(prefix + "*.pt"))
    if len(matches) != 1:
        raise ValueError("{} matched {} files in {}".format(
            prefix, len(matches), directory))
    return matches[0]


def load_uint8(path: Path):
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError("{} is not [T,3,H,W]: {}".format(
            path, tuple(tensor.shape)))
    if tensor.dtype != torch.uint8:
        raise TypeError("{} is {}, expected uint8".format(path, tensor.dtype))
    return tensor


def image_array(video, frame):
    return video[frame].permute(1, 2, 0).numpy()


def resize_for_panel(video, height, width):
    """Normalize display resolution without changing metric source tensors."""
    if tuple(video.shape[-2:]) == (height, width):
        return video
    resized = F.interpolate(
        video.float(), size=(height, width), mode="bilinear",
        align_corners=False)
    return resized.round().clamp(0, 255).to(torch.uint8)


def font(size, bold=False):
    names = ([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ] if bold else [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ])
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def render_panel(
        rows, future_indices, fps, output_stem: Path, scene_labels,
        scene_group_size=3, time_label_offset_frames=1):
    row_count = len(rows)
    column_count = len(future_indices)
    frame_height, frame_width = rows[0]["video"].shape[-2:]
    left_margin = 220
    header = 62
    column_gap = 8
    row_gap = 8
    scene_gap = 28
    scene_starts = {start for start, _label in scene_labels if start > 0}
    extra_scene_space = scene_gap * len(scene_starts)
    canvas_width = (
        left_margin + column_count * frame_width
        + (column_count - 1) * column_gap)
    canvas_height = (
        header + row_count * frame_height + (row_count - 1) * row_gap
        + extra_scene_space)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(28)
    label_font = font(27)
    scene_font = font(27, bold=True)

    for column_index, frame_index in enumerate(future_indices):
        title = "+{:.1f} s".format(
            (frame_index + time_label_offset_frames) / fps)
        x = left_margin + column_index * (frame_width + column_gap)
        box = draw.textbbox((0, 0), title, font=title_font)
        draw.text(
            (x + (frame_width - (box[2] - box[0])) / 2, 16),
            title, fill="black", font=title_font)

    row_tops = []
    accumulated_scene_gap = 0
    for row_index, row in enumerate(rows):
        if row_index in scene_starts:
            accumulated_scene_gap += scene_gap
        top = header + row_index * (frame_height + row_gap) \
            + accumulated_scene_gap
        row_tops.append(top)
        video = row["video"]
        for column_index, frame_index in enumerate(future_indices):
            x = left_margin + column_index * (frame_width + column_gap)
            canvas.paste(Image.fromarray(image_array(video, frame_index)),
                         (x, top))
        label = row["method"]
        box = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            (left_margin - 14 - (box[2] - box[0]),
             top + (frame_height - (box[3] - box[1])) / 2),
            label, fill="black", font=label_font)

    # A thin separator and scene label for each method group make the large panel
    # readable without adding text over any evaluated pixels.
    for row_start, label in scene_labels:
        group_end = min(row_start + scene_group_size, row_count) - 1
        group_mid = (
            row_tops[row_start] + row_tops[group_end] + frame_height) / 2
        box = draw.textbbox((0, 0), label, font=scene_font)
        draw.text(
            (8, group_mid - (box[3] - box[1]) / 2),
            label, fill="black", font=scene_font)
        if row_start > 0:
            y = row_tops[row_start] - scene_gap // 2 - row_gap // 2
            draw.line((4, y, canvas_width - 4, y), fill=(180, 180, 180),
                      width=2)
    canvas.save(output_stem.with_suffix(".png"), dpi=(300, 300))
    canvas.save(output_stem.with_suffix(".pdf"), "PDF", resolution=300)


def main():
    args = parse_args()
    horizon_real_dir = Path(args.horizon_real)
    horizon_fake_dir = Path(args.horizon_fake)
    pandwm_real_dir = Path(args.pandwm_real)
    pandwm_fake_dir = Path(args.pandwm_fake)
    epona_fake_dir = Path(args.epona_fake) if args.epona_fake else None
    gem_fake_dir = Path(args.gem_fake) if args.gem_fake else None
    vista_fake_dir = Path(args.vista_fake) if args.vista_fake else None
    omni_fake_dir = Path(args.omni_fake) if args.omni_fake else None
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    future_indices = [int(value) for value in args.future_indices.split(",")]

    manifest = {
        "selection_rule": (
            "HorizonDrive common-dataset chronological indices at the "
            "first/middle/last quantiles, fixed before visual inspection"),
        "future_indices": future_indices,
        "column_seconds": [
            (index + args.time_label_offset_frames) / args.fps
            for index in future_indices],
        "column_time_origin": (
            "first scored target frame" if args.time_label_offset_frames == 0
            else "last conditioning frame"),
        "condition_frames": args.condition_frames,
        "target_start": (
            args.condition_frames
            if args.target_start is None else args.target_start),
        "target_frames": args.target_frames,
        "epona_context_frames": (
            args.epona_context_frames if epona_fake_dir else None),
        "gem_context_frames": (
            args.gem_context_frames if gem_fake_dir else None),
        "vista_context_frames": (
            args.vista_context_frames if vista_fake_dir else None),
        "omni_context_frames": (
            args.omni_context_frames if omni_fake_dir else None),
        "display_resize": (
            "bilinear resize to canonical ground-truth resolution; raw metric "
            "tensors remain unchanged"),
        "fps": args.fps,
        "items": [],
    }
    combined_rows = []
    combined_labels = []
    for item_number, (horizon_index, anchor) in enumerate(args.item):
        label = "Scene {}".format(chr(ord("A") + item_number))
        paths = {
            "horizon_real": unique_prefix(
                horizon_real_dir, "{}_".format(horizon_index)),
            "horizon_fake": unique_prefix(
                horizon_fake_dir, "{}_".format(horizon_index)),
            "pandwm_real": pandwm_real_dir / "{}.pt".format(anchor),
            "pandwm_fake": pandwm_fake_dir / "{}.pt".format(anchor),
        }
        if epona_fake_dir is not None:
            paths["epona_fake"] = unique_prefix(
                epona_fake_dir, "{}_".format(horizon_index))
        if gem_fake_dir is not None:
            paths["gem_fake"] = (
                gem_fake_dir / "clip_{:04d}.pt".format(horizon_index))
        if vista_fake_dir is not None:
            paths["vista_fake"] = (
                vista_fake_dir / "clip_{:04d}.pt".format(horizon_index))
        if omni_fake_dir is not None:
            paths["omni_fake"] = (
                omni_fake_dir / "clip_{:04d}.pt".format(horizon_index))
        for role, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError("{}: {}".format(role, path))

        target_start = (
            args.condition_frames
            if args.target_start is None else args.target_start)
        pandwm_start = target_start - args.condition_frames
        if pandwm_start < 0:
            raise ValueError("target-start precedes PanDWM predictions")
        target_stop = (
            None if args.target_frames is None
            else target_start + args.target_frames)
        pandwm_stop = (
            None if args.target_frames is None
            else pandwm_start + args.target_frames)
        horizon_real = load_uint8(paths["horizon_real"])[
            target_start:target_stop]
        horizon_fake = load_uint8(paths["horizon_fake"])[
            target_start:target_stop]
        pandwm_real = load_uint8(paths["pandwm_real"])[
            pandwm_start:pandwm_stop]
        pandwm_fake = load_uint8(paths["pandwm_fake"])[
            pandwm_start:pandwm_stop]
        difference = (
            horizon_real.to(torch.int16) - pandwm_real.to(torch.int16))
        different_values = int((difference != 0).sum())
        max_abs_difference = int(difference.abs().max())
        difference_fraction = different_values / difference.numel()
        if (max_abs_difference > args.max_gt_abs_diff
                or difference_fraction > args.max_gt_diff_fraction):
            raise ValueError(
                "GT mismatch for {}: max abs {}, {} / {} differing values"
                .format(label, max_abs_difference, different_values,
                        difference.numel()))
        videos = {
                "Ground truth": horizon_real,
                "HorizonDrive": horizon_fake,
                "PanDWM": pandwm_fake,
        }
        if epona_fake_dir is not None:
            epona_stop = (
                None if args.target_frames is None
                else args.epona_context_frames + args.target_frames)
            videos["Epona"] = load_uint8(paths["epona_fake"])[
                args.epona_context_frames:epona_stop]
        if gem_fake_dir is not None:
            gem_stop = (
                None if args.target_frames is None
                else args.gem_context_frames + args.target_frames)
            videos["GEM"] = load_uint8(paths["gem_fake"])[
                args.gem_context_frames:gem_stop]
        if vista_fake_dir is not None:
            vista_stop = (
                None if args.target_frames is None
                else args.vista_context_frames + args.target_frames)
            videos["Vista"] = load_uint8(paths["vista_fake"])[
                args.vista_context_frames:vista_stop]
        if omni_fake_dir is not None:
            omni_stop = (
                None if args.target_frames is None
                else args.omni_context_frames + args.target_frames)
            videos["OmniNWM"] = load_uint8(paths["omni_fake"])[
                args.omni_context_frames:omni_stop]
        original_shapes = {
            name: list(video.shape) for name, video in videos.items()
        }
        for name, video in videos.items():
            if video.shape[0] <= max(future_indices):
                raise ValueError("{} has only {} future frames".format(
                    name, video.shape[0]))

        display_height, display_width = horizon_real.shape[-2:]
        videos = {
            name: resize_for_panel(video, display_height, display_width)
            for name, video in videos.items()
        }

        rows = [
            {"method": name, "video": video}
            for name, video in videos.items()
        ]
        render_panel(
            rows, future_indices, args.fps,
            output_dir / "scene_{}".format(chr(ord("A") + item_number)),
            [(0, label)], scene_group_size=len(rows),
            time_label_offset_frames=args.time_label_offset_frames)
        combined_labels.append((len(combined_rows), label))
        combined_rows.extend(rows)
        manifest["items"].append({
            "label": label,
            "horizon_dataset_index": horizon_index,
            "anchor_timestamp_us": anchor,
            "ground_truth_comparison": {
                "maximum_absolute_uint8_difference": max_abs_difference,
                "different_channel_values": different_values,
                "total_channel_values": difference.numel(),
                "different_fraction": difference_fraction,
                "explanation": (
                    "CPU/GPU grid_sample rounding may differ by one LSB; "
                    "the displayed GT is HorizonDrive's canonical dump."),
            },
            "raw_video_shapes": original_shapes,
            "files": {
                role: {"path": str(path), "sha256": sha256(path)}
                for role, path in paths.items()
            },
        })

    render_panel(
        combined_rows, future_indices, args.fps,
        output_dir / "qualitative_comparison",
        combined_labels,
        scene_group_size=(
            3 + int(epona_fake_dir is not None)
            + int(gem_fake_dir is not None)
            + int(vista_fake_dir is not None)
            + int(omni_fake_dir is not None)),
        time_label_offset_frames=args.time_label_offset_frames)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print("wrote", output_dir)


if __name__ == "__main__":
    main()
