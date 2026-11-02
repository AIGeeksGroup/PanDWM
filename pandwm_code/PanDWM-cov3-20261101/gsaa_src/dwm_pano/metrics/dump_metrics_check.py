"""Re-score a metric dump with our own metrics, and probe the instruments.

Two jobs, both on the PNG dump written by
`dwm_pano.pipelines.ctsd_pano.dump_stitched_panos`:

1. **Fidelity.** Recompute `torchmetrics` FID and `PanoPatchFrechetVideoDistance`
   from the dumped files. The dump stores the uint8 tensor torchmetrics itself
   produced (`(imgs * 255).byte()`), so feeding it back with `normalize=False`
   is the identical tensor and the numbers must reproduce the in-run values. If
   they do, the ADM evaluator in `adm_eval_pano.py` is scoring exactly the
   pixels torchmetrics scored, and the two numbers can be paired into a
   conversion. If they do not, the dump is lying and nothing downstream is
   trustworthy.

2. **Sanity battery** (README section 7). These test the metric, not the model:

   - `self_FID` / `self_FVD`: a distribution against itself must score 0.
     This is a WEAK test and should not be oversold: both sides are fed the
     same tensors, so the two accumulator states are identical by
     construction and the result only reflects the conditioning of the
     matrix-square-root step. It catches a broken Frechet implementation,
     not a broken data path.
   - `split_half_FID` / `split_half_FVD`: real-vs-real across disjoint
     halves. This is a LEVEL, not a minimum detectable difference. It says
     what a Frechet distance costs at N/2 purely from finite sampling; it
     does NOT bound the smallest real gap between two checkpoints, because
     two checkpoints are scored against the *same* real set and that shared
     sampling noise largely cancels in their difference. Do not use it to
     dismiss a model-vs-model gap -- that needs a paired/bootstrap estimate
     of the difference, which this script does not compute.
   - `time_shuffle_FVD`: real vs the same frames in shuffled temporal order.
     This one matters here specifically. Our FVD pools the TATS I3D **400-dim
     classifier logits**, time-averaged, rather than the usual 1024-dim
     Mixed_5c features. Time-averaging after a 3D conv stack should still
     carry temporal structure, but "should" is not evidence. If shuffling time
     barely moves FVD, our FVD is close to an image metric and every temporal
     claim made with it needs re-reading.
   - `repeat_frame_FVD`: real vs frame 0 held static for the whole clip. A
     temporally sensitive metric must punish this hard.

Usage (pandwm env)::

    python3 -m dwm_pano.metrics.dump_metrics_check \
        --dump-dir /path/to/pandwm_root/eval_dumps/cd_10000_ema \
        --i3d /path/to/pandwm_root/code/TATS_fvd/i3d_pretrained_400.pt \
        --out /path/to/pandwm_root/eval_dumps/cd_10000_ema/torchmetrics_recheck.json
"""

import argparse
import collections
import glob
import json
import os

import torch
import torchvision


def load_manifest(dump_dir: str):
    """Return records grouped into clips, in a deterministic order.

    A clip is one (global_step, rank, batch_index, sample_index, view_index)
    group ordered by frame_index. Every component earns its place:

    - `global_step`: the manifest is opened in append mode, so pointing two
      eval runs at one dump directory concatenates their records. Without
      this key the two checkpoints' frames interleave into one sequence and
      FVD scores the seam between two models as if it were motion.
    - `sample_index`: redundant at batch_size 1, but if the eval dataloader
      ever batches more than one clip, grouping by batch alone merges them.
    - `view_index`: FVD consumes `(b v) t c h w`, i.e. one sequence per
      (sample, view). Sorting views inside a clip instead of separating them
      would interleave views frame by frame.

    Returns (clips, steps) so the caller can refuse a multi-step dump.
    """
    manifest_paths = sorted(
        glob.glob(os.path.join(dump_dir, "manifest_rank*.jsonl")))
    if not manifest_paths:
        raise FileNotFoundError(
            "no manifest_rank*.jsonl under {}".format(dump_dir))

    clips = collections.defaultdict(list)
    seen_names = set()
    steps = set()
    for path in manifest_paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "sample_index" not in record or "view_index" not in record:
                    raise ValueError(
                        "manifest predates the (sample, frame, view) fields; "
                        "re-dump with the current ctsd_pano.py or clips "
                        "cannot be separated")
                name = record["name"]
                if name in seen_names:
                    # Two runs into one directory with the same global_step
                    # overwrite each other's PNGs while both append records.
                    raise ValueError(
                        "duplicate sample name {} at {}:{}; the dump "
                        "directory was written by more than one run".format(
                            name, path, line_number))
                seen_names.add(name)
                steps.add(record.get("global_step"))
                clips[(
                    record.get("global_step"), record["rank"],
                    record["batch_index"], record["sample_index"],
                    record["view_index"])].append(record)

    ordered = []
    for key in sorted(clips, key=lambda k: tuple(
            -1 if part is None else part for part in k)):
        frames = sorted(clips[key], key=lambda r: r["frame_index"])
        ordered.append([r["name"] for r in frames])
    return ordered, steps


def read_clip(dump_dir: str, side: str, names, device):
    """Return one clip as (T, C, H, W) uint8 on `device`."""
    side_dir = os.path.join(dump_dir, side)
    frames = [
        torchvision.io.read_image(os.path.join(side_dir, name))
        for name in names
    ]
    return torch.stack(frames).to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Recompute our metrics from a dump and probe them")
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument(
        "--i3d", default=None,
        help="TATS i3d_pretrained_400.pt; FVD work is skipped without it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-count", type=int, default=16)
    parser.add_argument(
        "--limit-clips", type=int, default=None,
        help="score only the first N clips (smoke tests only)")
    parser.add_argument(
        "--expect-frames-per-clip", type=int, default=None,
        help="cardinality half of the reference-exclusion check: scored "
             "frames per clip = dataset sequence_length minus "
             "reference_frame_count. For cd_eval_h20.json that is 35 - 3 = "
             "32. NOT sequence_length_per_iteration (19), which is the "
             "autoregressive chunk size, not the clip length")
    parser.add_argument(
        "--expect-clips", type=int, default=None,
        help="fail if the dump does not hold exactly this many clips; set it "
             "to the actual validation-dataset length capped by "
             "evaluation_item_count. cd_eval_h20.json enumerates 149 clips; "
             "its evaluation_item_count=480 is only an upper bound. Without "
             "this guard a dump truncated by a killed eval scores silently at "
             "a smaller N, and FID's 1/N bias makes that look like a model "
             "difference")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torchmetrics.image.fid
    # Imported up front, not lazily before the FVD section: a missing
    # pytorch_i3d on PYTHONPATH or a bad --i3d path would otherwise surface
    # only after the FID pass has already spent its time.
    fvd_module = None
    if args.i3d:
        from dwm_pano.metrics import fvd_pano as fvd_module
        if not os.path.exists(args.i3d):
            raise FileNotFoundError("I3D checkpoint missing: " + args.i3d)

    device = torch.device(args.device)
    clips, steps = load_manifest(args.dump_dir)
    if len(steps) > 1:
        raise ValueError(
            "dump directory holds records from {} different global_steps "
            "({}); scoring them together would pool several checkpoints into "
            "one distribution".format(len(steps), sorted(
                s for s in steps if s is not None)))
    if args.limit_clips is not None:
        clips = clips[:args.limit_clips]
    if not clips:
        raise ValueError("no clips to score in " + args.dump_dir)
    if args.expect_clips is not None and len(clips) != args.expect_clips:
        raise RuntimeError(
            "expected {} clips, dump holds {}; the eval run was truncated "
            "and any FID from it is at the wrong N".format(
                args.expect_clips, len(clips)))
    frame_total = sum(len(c) for c in clips)
    print(
        "clips: {}  frames: {}".format(len(clips), frame_total), flush=True)

    results = {
        "clip_count": len(clips), "frame_count": frame_total,
        "global_step": sorted(steps)[0] if steps else None,
    }

    # Reference exclusion (README section 7.4). The validation clip is 35
    # frames, the first 3 are context, and generate_frames_for_reference is
    # false -- so every clip must contribute exactly 32 scored frames
    # (sequence_length_per_iteration=19 is the AR chunk size, not the clip
    # length). A clip carrying 35 would mean context frames leaked into the
    # scored window and every FID/FVD number is flattered by 3 real frames.
    frames_per_clip = sorted({len(c) for c in clips})
    results["frames_per_clip"] = frames_per_clip
    if args.expect_frames_per_clip is not None:
        if frames_per_clip != [args.expect_frames_per_clip]:
            raise RuntimeError(
                "reference-exclusion check failed: expected every clip to "
                "hold {} frames, found {}".format(
                    args.expect_frames_per_clip, frames_per_clip))
        results["reference_exclusion"] = "pass"

    # ---- FID, fed the dumped uint8 directly -------------------------------
    # normalize=False because the PNG already holds torchmetrics' own
    # `(x * 255).byte()` output. Re-normalizing would round-trip through float
    # and stop being the identical tensor.
    fid = torchmetrics.image.fid.FrechetInceptionDistance(
        normalize=False).to(device)
    fid_half_a = torchmetrics.image.fid.FrechetInceptionDistance(
        normalize=False).to(device)
    fid_self = torchmetrics.image.fid.FrechetInceptionDistance(
        normalize=False).to(device)
    half = len(clips) // 2
    with torch.no_grad():
        for index, names in enumerate(clips):
            real = read_clip(args.dump_dir, "real", names, device)
            fake = read_clip(args.dump_dir, "fake", names, device)
            fid.update(real, real=True)
            fid.update(fake, real=False)
            # Same distribution on both sides: must be 0.
            fid_self.update(real, real=True)
            fid_self.update(real, real=False)
            # Disjoint real halves: the finite-sample floor.
            if index < half:
                fid_half_a.update(real, real=True)
            elif index < 2 * half:
                fid_half_a.update(real, real=False)
            if (index + 1) % 25 == 0:
                print("  fid {}/{}".format(index + 1, len(clips)), flush=True)

    results["FID_torchmetrics_from_dump"] = float(fid.compute())
    results["self_FID_real_vs_real"] = float(fid_self.compute())
    if half >= 1:
        results["split_half_FID_torchmetrics"] = float(fid_half_a.compute())
        results["split_half_clip_count"] = half
    del fid, fid_half_a, fid_self
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- FVD and the temporal-sensitivity probes --------------------------
    if fvd_module is not None:
        shortest = min(len(c) for c in clips)
        if shortest < args.sequence_count:
            raise ValueError(
                "--sequence-count {} exceeds the shortest clip ({} frames); "
                "the time-shuffle permutation would index past its "
                "end".format(args.sequence_count, shortest))
        # Every FVD number below scores only the FIRST --sequence-count
        # frames of each clip; the metric truncates internally. With 32
        # scored frames and sequence_count 16 that is the first half of the
        # generated horizon, so none of these probes says anything about
        # frames 16-31.
        results["fvd_frames_scored_per_clip"] = args.sequence_count

        def make_fvd():
            return fvd_module.PanoPatchFrechetVideoDistance(
                inception_3d_checkpoint_path=args.i3d,
                sequence_count=args.sequence_count).to(device)

        fvd = make_fvd()
        fvd_self = make_fvd()
        fvd_half = make_fvd()
        fvd_shuffle = make_fvd()
        fvd_static = make_fvd()

        # One fixed permutation, applied to every clip, so the probe is
        # reproducible and is not itself a source of run-to-run noise.
        generator = torch.Generator().manual_seed(0)
        permutation = torch.randperm(args.sequence_count, generator=generator)
        print("time-shuffle permutation: {}".format(
            permutation.tolist()), flush=True)

        with torch.no_grad():
            for index, names in enumerate(clips):
                real = read_clip(
                    args.dump_dir, "real", names, device).float() / 255.0
                fake = read_clip(
                    args.dump_dir, "fake", names, device).float() / 255.0
                real = real.unsqueeze(0)
                fake = fake.unsqueeze(0)

                fvd.update(real, real=True)
                fvd.update(fake, real=False)

                # Identical distribution on both sides -> must be 0.
                fvd_self.update(real, real=True)
                fvd_self.update(real, real=False)

                if index < half:
                    fvd_half.update(real, real=True)
                elif index < 2 * half:
                    fvd_half.update(real, real=False)

                fvd_shuffle.update(real, real=True)
                fvd_shuffle.update(
                    real[:, permutation[:real.shape[1]]], real=False)

                fvd_static.update(real, real=True)
                fvd_static.update(
                    real[:, :1].expand(-1, real.shape[1], -1, -1, -1),
                    real=False)

                if (index + 1) % 25 == 0:
                    print(
                        "  fvd {}/{}".format(index + 1, len(clips)),
                        flush=True)

        results["FVD_panopatch_from_dump"] = float(fvd.compute())
        results["self_FVD_real_vs_real"] = float(fvd_self.compute())
        if half >= 1:
            results["split_half_FVD"] = float(fvd_half.compute())
        results["time_shuffle_FVD"] = float(fvd_shuffle.compute())
        results["repeat_frame_FVD"] = float(fvd_static.compute())
        results["time_shuffle_permutation"] = permutation.tolist()
    else:
        print("--i3d not given; FVD section skipped", flush=True)

    text = json.dumps(results, indent=2)
    print(text, flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
