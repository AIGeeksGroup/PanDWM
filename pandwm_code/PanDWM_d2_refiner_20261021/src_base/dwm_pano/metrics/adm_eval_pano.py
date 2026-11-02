"""ADM-convention FID / sFID for panoramas, read from a manifest.

Why this file exists
--------------------
Every PanDWM number so far is `torchmetrics.image.fid.FrechetInceptionDistance`
(pytorch-fid Inception weights, PyTorch resize). Published diffusion tables are
almost always the ADM / guided-diffusion convention: OpenAI's
`classify_image_graph_def.pb` graph, `pool_3:0` for FID and `mixed_6/conv:0`
(first 7 channels) for sFID. The two are not interchangeable, so a torchmetrics
number cannot be placed next to a paper's number. This script scores the SAME
pixels under the ADM convention so a conversion can be pinned once and applied
to the whole internal history.

Two deliberate choices
----------------------
1. The upstream evaluator is imported, not vendored. `Evaluator` builds the
   TensorFlow feature graph and `FIDStatistics.frechet_distance` does the
   Frechet math, both straight from
   `references/NiT/projects/evaluate/adm_evaluator.py`. Only the part with a
   bug is replaced.
2. That bug: `read_activations` enumerates a folder with
   `run_iter = int(len(files) / self.batch_size)` (adm_evaluator.py:147), which
   silently drops the final partial batch -- up to `batch_size - 1` samples
   never scored, with no warning. We read from the JSONL manifest written by
   `dwm_pano.pipelines.ctsd_pano.dump_stitched_panos` and feed batches straight
   into `compute_activations`, which consumes every batch it is given. The
   manifest also lets us assert real/fake set parity.

The ADM graph resizes internally to 299x299 (input is injected at
`ExpandDims:0`, upstream of the graph's own ResizeBilinear), exactly like
torchmetrics does on its side. For a 4114x256 panorama that is a 16:1 squash in
both cases -- severe, but symmetric, so the comparison stays meaningful.

Usage (in the TensorFlow venv, not the pandwm env)::

    /path/to/pandwm_root/envs/admeval/bin/python adm_eval_pano.py \
        --dump-dir /path/to/pandwm_root/eval_dumps/cd_10000_ema \
        --adm-evaluator-dir /path/to/pandwm_root/code/NiT_evaluate \
        --out /path/to/pandwm_root/eval_dumps/cd_10000_ema/adm_metrics.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image


def load_manifest(dump_dir: str):
    """Return (sample names, set of global_steps) from the manifest.

    Reading the manifest rather than `os.listdir` is the whole point: it is
    the record of what the metric actually scored, it survives partial runs,
    and it makes a missing file an error instead of a silently shorter set.

    `global_step` is collected so the caller can refuse a directory holding
    more than one checkpoint's output. That case is otherwise completely
    silent here: the step is part of each filename, so no name collides, the
    duplicate guard passes, and the two checkpoints get pooled into one
    distribution and scored against the real frames counted twice.
    """
    manifest_paths = sorted(
        glob.glob(os.path.join(dump_dir, "manifest_rank*.jsonl")))
    if not manifest_paths:
        raise FileNotFoundError(
            "no manifest_rank*.jsonl under {}; was PANDWM_FID_DUMP_DIR set "
            "during the eval run?".format(dump_dir))

    names = []
    seen = set()
    steps = set()
    for path in manifest_paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "{}:{} is not valid JSON: {}".format(
                            path, line_number, error))
                name = record["name"]
                if name in seen:
                    raise ValueError(
                        "duplicate sample name {} in {}".format(name, path))
                seen.add(name)
                steps.add(record.get("global_step"))
                names.append(name)

    if not names:
        raise ValueError("manifest under {} is empty".format(dump_dir))

    names.sort()
    return names, steps


def check_pairs(dump_dir: str, names):
    """Fail loudly on any name missing from either side."""
    missing = []
    for side in ("real", "fake"):
        side_dir = os.path.join(dump_dir, side)
        for name in names:
            if not os.path.exists(os.path.join(side_dir, name)):
                missing.append(os.path.join(side, name))
    if missing:
        raise FileNotFoundError(
            "{} manifest entries have no file on disk, first few: {}".format(
                len(missing), missing[:5]))


def batch_generator(dump_dir: str, side: str, names, batch_size: int):
    """Yield NHWC uint8 batches covering every name, tail included."""
    side_dir = os.path.join(dump_dir, side)
    batch = []
    for name in names:
        with Image.open(os.path.join(side_dir, name)) as image:
            batch.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        if len(batch) == batch_size:
            yield np.stack(batch)
            batch = []
    if batch:
        # The tail batch adm_evaluator.py:147 throws away.
        yield np.stack(batch)


def frechet(evaluator, acts_a, acts_b):
    stats_a = evaluator.compute_statistics(acts_a)
    stats_b = evaluator.compute_statistics(acts_b)
    return float(stats_a.frechet_distance(stats_b))


def main():
    parser = argparse.ArgumentParser(
        description="ADM-convention FID/sFID over a manifest-backed dump")
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument(
        "--adm-evaluator-dir", required=True,
        help="directory holding adm_evaluator.py (from references/NiT)")
    parser.add_argument(
        "--inception-pb", default=None,
        help="path to classify_image_graph_def.pb; downloaded if absent")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--save-activations", default=None,
        help="optional .npz path so a re-score costs no TensorFlow time")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="score only the first N manifest entries (smoke tests only)")
    parser.add_argument(
        "--expect-samples", type=int, default=None,
        help="fail unless the manifest holds exactly this many entries "
             "(actual evaluated clips x scored frames per clip). Note that "
             "evaluation_item_count is only an upper bound: cd_eval_h20.json "
             "enumerates 149 clips, hence 149 x 32 = 4768 samples. Without "
             "this guard a dump truncated by a killed eval is scored silently "
             "at a smaller N, and FID's 1/N bias then reads as a model "
             "difference")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.adm_evaluator_dir))
    import adm_evaluator
    import tensorflow.compat.v1 as tf

    if args.inception_pb is not None:
        # Module-level constant; point it at a pre-fetched copy so the run
        # does not depend on reaching openaipublic.
        adm_evaluator.INCEPTION_V3_PATH = args.inception_pb

    names, steps = load_manifest(args.dump_dir)
    if len(steps) > 1:
        raise ValueError(
            "dump directory holds records from {} different global_steps "
            "({}); scoring them together pools several checkpoints into one "
            "distribution".format(len(steps), sorted(
                s for s in steps if s is not None)))
    if args.limit is not None:
        names = names[:args.limit]
    if not names:
        raise ValueError("no samples selected from " + args.dump_dir)
    if args.expect_samples is not None and len(names) != args.expect_samples:
        raise RuntimeError(
            "expected {} samples, manifest holds {}; the eval run was "
            "truncated and any FID from it is at the wrong N".format(
                args.expect_samples, len(names)))
    check_pairs(args.dump_dir, names)
    print("manifest entries: {}".format(len(names)), flush=True)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    evaluator = adm_evaluator.Evaluator(
        tf.Session(config=config), batch_size=args.batch_size)
    print("warming up TensorFlow ...", flush=True)
    evaluator.warmup()

    print("computing real activations ...", flush=True)
    real_pool, real_spatial = evaluator.compute_activations(
        batch_generator(args.dump_dir, "real", names, args.batch_size))
    print("computing fake activations ...", flush=True)
    fake_pool, fake_spatial = evaluator.compute_activations(
        batch_generator(args.dump_dir, "fake", names, args.batch_size))

    for label, array in (
        ("real_pool", real_pool), ("fake_pool", fake_pool),
        ("real_spatial", real_spatial), ("fake_spatial", fake_spatial)
    ):
        if array.shape[0] != len(names):
            raise RuntimeError(
                "{} has {} rows for {} manifest entries -- samples were "
                "dropped".format(label, array.shape[0], len(names)))

    # Persisted BEFORE any Frechet call. The TensorFlow pass is the expensive
    # part of this script; `frechet_distance` can raise on an ill-conditioned
    # sqrtm ("Imaginary component ..."), and losing an hour of feature
    # extraction to a linear-algebra hiccup would be gratuitous.
    if args.save_activations:
        np.savez(
            args.save_activations, real_pool=real_pool, fake_pool=fake_pool,
            real_spatial=real_spatial, fake_spatial=fake_spatial,
            names=np.array(names))
        print("activations saved to {}".format(args.save_activations))

    results = {
        "sample_count": len(names),
        "batch_size": args.batch_size,
        "global_step": sorted(steps)[0] if steps else None,
        "feature_dim_pool": int(real_pool.shape[1]),
        "feature_dim_spatial": int(real_spatial.shape[1]),
        "FID_ADM_native_pano": frechet(evaluator, fake_pool, real_pool),
        "sFID_ADM_native_pano": frechet(evaluator, fake_spatial, real_spatial),
    }

    # Instrument checks, cheap because the activations are already in memory.
    #
    # self-* is a WEAK test here and must not be oversold: the same array is
    # passed on both sides, and `compute_statistics` is a pure function, so
    # there is no accumulator being exercised at all. It only shows that the
    # matrix-square-root step is numerically well behaved. The torchmetrics
    # side (dump_metrics_check.py) at least pushes data through two separate
    # accumulators.
    #
    # split_half is a LEVEL -- what a Frechet distance costs at N/2 from
    # finite sampling alone. It is NOT a minimum detectable difference
    # between two checkpoints, which are scored against the same real set so
    # that most of this noise cancels in their difference.
    #
    # The spatial branch gets the same two probes: sFID's plumbing (the
    # `mixed_6/conv:0` slice to 7 channels, then a 17*17*7 flatten) is the
    # part most likely to break silently, and it had no check at all.
    results["self_FID_real_vs_real"] = frechet(evaluator, real_pool, real_pool)
    results["self_sFID_real_vs_real"] = frechet(
        evaluator, real_spatial, real_spatial)
    half = len(names) // 2
    if half >= 2:
        results["split_half_real_FID"] = frechet(
            evaluator, real_pool[:half], real_pool[half:2 * half])
        results["split_half_real_sFID"] = frechet(
            evaluator, real_spatial[:half], real_spatial[half:2 * half])
        results["split_half_sample_count"] = half

    text = json.dumps(results, indent=2)
    print(text, flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
