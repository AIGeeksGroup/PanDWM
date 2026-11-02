"""Produce a band-limited six-real-camera number through the protocol bridge.

Two modes:

``floor``
    Split the real validation clips into two disjoint halves and score one
    against the other with **no model involved**.  The true distance is zero,
    so whatever comes out is the protocol's own noise/bias floor.  This is the
    first number the bridge should ever produce: without it an arm's value has
    no scale, exactly as the pano-space floor showed (a real-vs-real pooled
    PanoPatch-FVD reads 117.6 at n=450).

``arm``
    Score a directory of generated V=1 canvases against the real clips.

Runs on CPU; the only heavy step is I3D, which takes a GPU when one is
visible.  Nothing here needs the model, the Gaussian decoder, or raw nuScenes
camera images.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from dwm_pano.datasets.pano_projection import CAM_ORDER          # noqa: E402
from dwm_pano.nuscenes6_bridge import (                          # noqa: E402
    PANO_CANVAS_OVERLAP_COLUMNS, canvas_source_axes,
    real_camera_band_table, resample_pano_to_rig,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="eval config, so the clip list matches every "
                             "other arm exactly")
    parser.add_argument("--mode", choices=("floor", "arm"), default="floor")
    parser.add_argument("--items", type=int, default=149)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--render-height", type=int, default=360)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--i3d", required=True,
                        help="TATS i3d_pretrained_400.pt; must be the same "
                             "file the pano FVD uses or the numbers are not "
                             "the same instrument")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    return parser.parse_args()


def build_dataset(config_path, items):
    """Instantiate the very dataset the other arms evaluated on."""
    import dwm.common

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("validation_dataset", "evaluation_dataset", "dataset"):
        if key in config:
            dataset = dwm.common.create_instance_from_config(config[key])
            break
    else:
        raise KeyError(
            "no dataset block in {}; looked for validation_dataset / "
            "evaluation_dataset / dataset".format(config_path))
    count = len(dataset)
    if items > count:
        raise ValueError(
            "asked for {} items but the dataset has {}".format(items, count))
    return dataset, count


# dwm.datasets.common.DatasetAdapter renames the pano RGB to this key and
# hands it over as a [T, V=2, C, H, W] float tensor already in [0, 1], so the
# bridge never touches PIL.
RGB_KEY = "vae_images"


def clip_to_canvas(sample, frames):
    """Stack a clip's pano halves into the pipeline's V=1 canvas.

    The pipeline keeps the whole left half and drops the first
    PANO_CANVAS_OVERLAP_COLUMNS columns of the right one, which is the same
    convention `canvas_source_axes` inverts.
    """
    halves = sample[RGB_KEY][:frames]
    if halves.ndim != 5 or halves.shape[1] != 2:
        raise ValueError(
            "{} must be [T, 2, C, H, W], got {}".format(
                RGB_KEY, tuple(halves.shape)))
    left = halves[:, 0]
    right = halves[:, 1][..., PANO_CANVAS_OVERLAP_COLUMNS:]
    return torch.cat([left, right], dim=-1).float()


def bridge_clip(canvas_sequence, sample, output_size, axes):
    """[T,C,Hc,Wc] canvases -> [T,6,C,H,W] real-camera views."""
    c2w = sample["pano_camera_transforms"].numpy()
    intrinsics = sample["pano_camera_intrinsics"].numpy()
    image_size = sample["pano_camera_image_size"].numpy()
    views = []
    reports = None
    for index in range(canvas_sequence.shape[0]):
        frame_views, valid, frame_reports = resample_pano_to_rig(
            canvas_sequence[index], c2w[index], intrinsics[index],
            image_size[index], output_size, band_limit=True,
            source_axes=axes)
        views.append(frame_views)
        if reports is None:
            reports = frame_reports
            first_valid = valid
    return torch.stack(views), first_valid, reports


def main():
    args = parse_args()
    from dwm_pano.metrics.nuscenes6_fvd import (
        NuScenes6BandFrechetVideoDistance,
    )
    from torchmetrics.image.fid import FrechetInceptionDistance

    dataset, total = build_dataset(args.config, args.items)
    print("dataset items: {} (using {})".format(total, args.items), flush=True)

    probe = dataset[0]
    if RGB_KEY not in probe:
        raise KeyError(
            "sample has no {}; available keys: {}".format(
                RGB_KEY, sorted(probe)))
    axes = canvas_source_axes(
        half_size=tuple(int(v) for v in probe[RGB_KEY].shape[-2:]))
    band_table = real_camera_band_table(
        probe["pano_camera_transforms"][0].numpy(),
        probe["pano_camera_intrinsics"][0].numpy(),
        probe["pano_camera_image_size"][0].numpy())
    print("\n| camera | native rows in band | fraction |")
    print("| --- | --- | ---: |")
    for row in band_table:
        print("| {} | {:.0f}..{:.0f} of {} | {:.1%} |".format(
            row["camera"], row["row_low"], row["row_high"], row["height"],
            row["covered_fraction"]))

    output_size = (args.render_height, args.render_width)
    fvd = NuScenes6BandFrechetVideoDistance(
        inception_3d_checkpoint_path=args.i3d,
        sequence_count=args.frames).to(args.device)
    fid = FrechetInceptionDistance(feature=2048, normalize=True) \
        .to(args.device)

    if args.mode != "floor":
        raise NotImplementedError(
            "arm mode needs a generated-canvas dump; run --mode floor first "
            "so the protocol has a scale")

    order = list(range(args.items))
    half = len(order) // 2
    split = {"A": order[:half], "B": order[half:]}
    print("\nfloor split: A={} B={}".format(len(split["A"]),
                                            len(split["B"])), flush=True)

    coverage = []
    started = time.time()
    for side, indices in split.items():
        is_reference = side == "A"
        for position, index in enumerate(indices):
            sample = dataset[index]
            canvases = clip_to_canvas(sample, args.frames)
            views, valid, reports = bridge_clip(
                canvases, sample, output_size, axes)
            if not coverage:
                coverage = reports
            fvd.update(views[None].to(args.device), real=is_reference,
                       valid=valid.to(args.device))
            frames = views.flatten(0, 1).to(args.device)
            fid.update(frames, real=is_reference)
            if position % 10 == 0:
                print("  {} {}/{}  {:.0f}s".format(
                    side, position, len(indices), time.time() - started),
                    flush=True)

    pooled = float(fvd.compute())
    per_camera = {k: float(v) for k, v in fvd.compute_per_camera().items()}
    fid_value = float(fid.compute())
    samples = int(fvd.real_features_num_samples)
    label = "FVD-I3D-nusc6band-{}-n{}".format(args.frames, samples)
    print("\n{} = {:.3f}".format(label, pooled))
    print("FID-nusc6band-n{} = {:.3f}".format(
        int(fid.real_features_num_samples), fid_value))
    print("per-camera:", " ".join(
        "{}={:.3f}".format(k, v) for k, v in per_camera.items()))
    print("\nparallax/band coverage of the reference rig:")
    for report in coverage:
        print("  {:16s} rows {:.1%}  in-band px {:.1%}  foreign region {:.1%}"
              .format(report["camera"], report["native_row_fraction"],
                      report["in_band_pixel_fraction"],
                      report["foreign_region_fraction"]))

    payload = {
        "mode": args.mode,
        "metric_label": label,
        "pooled_fvd": pooled,
        "fid": fid_value,
        "per_camera_fvd": per_camera,
        "fvd_samples": samples,
        "fid_samples": int(fid.real_features_num_samples),
        "items": args.items,
        "frames": args.frames,
        "render_size": list(output_size),
        "split": {key: len(value) for key, value in split.items()},
        "band_table": band_table,
        "coverage": coverage,
        "camera_order": list(CAM_ORDER),
        "elapsed_s": time.time() - started,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
