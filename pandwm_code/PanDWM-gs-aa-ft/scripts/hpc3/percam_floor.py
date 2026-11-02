"""Per-camera PanoPatch-FVD floor, measured with NO model involved.

Why this exists
---------------
The six Voronoi camera patches have different widths on the 4114-wide V=1
panorama (CAM_BACK 1032 vs CAM_BACK_RIGHT 593) and each is independently
bilinear-resized to I3D's 224x224. CAM_BACK is therefore squeezed 4.61x
horizontally while the others are squeezed 2.65-2.84x. So the per-camera FVD
values, which the change log has been ranking across cameras, may carry a
geometry-dependent offset that has nothing to do with the model.

What it measures
----------------
Split the 149 real validation clips into two disjoint halves A and B and feed
A as "real" and B as "fake". Nothing is generated. The resulting per-camera
FVD(A, B) is that patch's intrinsic floor under its own geometry and its own
content diversity. Run twice:

  native  - the production patch boundaries (unequal widths)
  eqw593  - every patch center-cropped to 593 columns, so all six are
            geometrically identical

Pre-registered reading (written before seeing numbers)
------------------------------------------------------
  native uneven + eqw593 even   -> the unevenness is PATCH GEOMETRY. Across-
                                   camera ranking in the log must be redone
                                   against these floors.
  native uneven + eqw593 uneven -> it is per-camera CONTENT DIVERSITY, not
                                   the resize. Ranking is still confounded,
                                   but by scene content, not by the metric.
  both even                     -> no floor problem; CAM_BACK's 262 is a real
                                   model defect and width was a red herring.

A/B are different scenes (every clip in this set is a distinct scene), so
this floor deliberately includes scene-sampling variability -- that is the
same variability present in the reported real-vs-fake numbers.
"""

import argparse
import json
import os
import sys

import einops
import torch

import dwm.common
from dwm_pano.pano_stitch import stitch_pano_halves


def build(config_path, equal_widths):
    config = json.load(open(config_path, encoding="utf-8"))
    dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        **dwm.common.instantiate_config(config["validation_dataloader"]))
    metric_config = dict(config["pipeline"]["metrics"]["fvd"])
    metrics = {}
    for w in equal_widths:
        cfg = dict(metric_config)
        cfg["equal_patch_width"] = w
        metrics["native" if w == 0 else "eqw{}".format(w)] = \
            dwm.common.create_instance_from_config(cfg)
    return dataset, dataloader, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--items", type=int, default=149)
    parser.add_argument("--ref-frames", type=int, default=3)
    parser.add_argument("--widths", default="0,593")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    widths = [int(x) for x in args.widths.split(",") if x != ""]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset, dataloader, metrics = build(args.config, widths)
    for m in metrics.values():
        m.to(device)
    print("dataset items: {} | arms: {} | device: {}".format(
        len(dataset), list(metrics), device), flush=True)

    counts = {"A": 0, "B": 0}
    for i, batch in enumerate(dataloader):
        if i >= args.items:
            break
        # same window the production fvd block scores: drop the reference
        # frames, keep the 16 that the model would have had to generate
        frames = batch["vae_images"][:, args.ref_frames:].to(device)
        stitched = stitch_pano_halves(frames)
        flat = einops.rearrange(stitched, "b t v c h w -> (b v) t c h w")
        side_a = (i % 2 == 0)
        for m in metrics.values():
            m.update(flat, real=side_a)
        counts["A" if side_a else "B"] += 1
        if (i + 1) % 25 == 0:
            print("  {}/{} clips (A={} B={})".format(
                i + 1, args.items, counts["A"], counts["B"]), flush=True)

    print("\nclips: A={} B={}".format(counts["A"], counts["B"]))

    # Conditioning check on the estimator itself. The Frechet distance needs a
    # d x d covariance, and the per-camera Gaussians see only one sample per
    # clip (V=1 after stitching). With d = 400 I3D features that estimate is
    # rank-deficient whenever n <= d, which inflates every per-camera value --
    # including the ones the change log has been ranking. Print it so the
    # limitation travels with the numbers.
    any_metric = next(iter(metrics.values()))
    d = int(any_metric.real_features_sum.shape[0])
    n_a = int(any_metric.real_cam_num_samples.min())
    n_b = int(any_metric.fake_cam_num_samples.min())
    print("feature dim d={}  per-camera samples: A={} B={}".format(d, n_a, n_b))
    if min(n_a, n_b) <= d:
        print("  WARNING: n <= d, so each per-camera covariance has rank <= "
              "{} of {}; per-camera values carry a large positive bias and "
              "only their RANKING is meaningful.".format(min(n_a, n_b) - 1, d))
    print("  (the pooled Gaussian pools all six patches: n={})".format(
        int(any_metric.real_features_num_samples)))

    result = {"counts": counts, "feature_dim": d,
              "per_camera_samples": {"A": n_a, "B": n_b},
              "pooled_samples": int(any_metric.real_features_num_samples),
              "arms": {}}
    for name, m in metrics.items():
        pooled = float(m.compute())
        per_cam = m.compute_per_camera()
        result["arms"][name] = {"pooled": pooled, "per_camera": per_cam}
        print("\n=== {} ===".format(name))
        print("pooled real-vs-real floor: {:.3f}".format(pooled))
        order = sorted(per_cam.items(), key=lambda kv: -kv[1])
        for cam, v in order:
            print("  {:<16} {:8.3f}".format(cam, v))
        vals = [v for _, v in per_cam.items()]
        print("  spread max/min: {:.3f}".format(max(vals) / max(min(vals), 1e-9)))

    if len(metrics) == 2 and "native" in metrics:
        other = [k for k in metrics if k != "native"][0]
        a = result["arms"]["native"]["per_camera"]
        b = result["arms"][other]["per_camera"]
        print("\n=== native -> {} (per camera) ===".format(other))
        for cam in a:
            print("  {:<16} {:8.3f} -> {:8.3f}  ({:+.3f})".format(
                cam, a[cam], b[cam], b[cam] - a[cam]))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                    exist_ok=True)
        json.dump(result, open(args.output, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("\nwrote", args.output)


if __name__ == "__main__":
    main()
