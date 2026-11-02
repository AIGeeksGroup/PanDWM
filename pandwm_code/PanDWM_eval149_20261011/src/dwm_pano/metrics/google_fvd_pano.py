"""Google-reference I3D FVD for PanDWM panorama metric dumps.

This evaluator intentionally runs in the isolated ``admeval`` TensorFlow
environment.  It follows the preprocessing and TF-Hub I3D endpoint from the
Google Research minimal FVD reference implementation, while retaining the
PanDWM six camera-owner panorama partition.  It reports both:

* pooled FVD over all six camera regions, matching the existing PanDWM pooling;
* one FVD per camera region and their macro average, so a single wide/easy
  region cannot dominate the result silently.

Reference implementation:
https://github.com/google-research/google-research/blob/master/
frechet_video_distance/frechet_video_distance.py
"""

import argparse
import collections
import glob
import json
import os

import numpy as np
from PIL import Image
import scipy.linalg


CAMERA_NAMES = (
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
)
PANO_WIDTH = 4114
PATCH_COLUMNS = (527, 1125, 1744, 2381, 3016, 3609)
TARGET_SIZE = (224, 224)
TFHUB_I3D = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"


def load_manifest(dump_dir):
    paths = sorted(glob.glob(os.path.join(dump_dir, "manifest_rank*.jsonl")))
    if not paths:
        raise FileNotFoundError("no manifest_rank*.jsonl under " + dump_dir)

    clips = collections.defaultdict(list)
    steps = set()
    seen_names = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                for key in ("rank", "batch_index", "sample_index",
                            "view_index", "frame_index", "name"):
                    if key not in record:
                        raise ValueError(
                            "manifest record {}:{} lacks {}".format(
                                path, line_number, key))
                if record["name"] in seen_names:
                    raise ValueError("duplicate frame name " + record["name"])
                seen_names.add(record["name"])
                steps.add(record.get("global_step"))
                key = (
                    record.get("global_step"), record["rank"],
                    record["batch_index"], record["sample_index"],
                    record["view_index"],
                )
                clips[key].append(record)

    if len(steps) != 1:
        raise ValueError("dump must contain exactly one global step: " + str(steps))

    ordered = []
    for key in sorted(clips, key=lambda item: tuple(
            -1 if value is None else value for value in item)):
        frames = sorted(clips[key], key=lambda item: item["frame_index"])
        ordered.append([item["name"] for item in frames])
    return ordered, next(iter(steps))


def split_camera_regions(frame):
    if frame.shape[1] != PANO_WIDTH:
        raise ValueError(
            "expected panorama width {}, got {}".format(
                PANO_WIDTH, frame.shape[1]))
    b0, b1, b2, b3, b4, b5 = PATCH_COLUMNS
    return (
        np.concatenate((frame[:, b5:], frame[:, :b0]), axis=1),
        frame[:, b0:b1],
        frame[:, b1:b2],
        frame[:, b2:b3],
        frame[:, b3:b4],
        frame[:, b4:b5],
    )


def load_clip_regions(dump_dir, side, names, sequence_length):
    if len(names) < sequence_length:
        raise ValueError(
            "clip has {} frames, needs {}".format(len(names), sequence_length))
    per_camera = [[] for _ in CAMERA_NAMES]
    for name in names[:sequence_length]:
        path = os.path.join(dump_dir, side, name)
        with Image.open(path) as image:
            frame = np.asarray(image.convert("RGB"))
        for camera_index, patch in enumerate(split_camera_regions(frame)):
            resized = Image.fromarray(patch).resize(
                TARGET_SIZE, resample=Image.Resampling.BILINEAR)
            per_camera[camera_index].append(np.asarray(resized))
    return [np.stack(frames) for frames in per_camera]


class GoogleI3DEmbedder:
    """Fixed-batch TF1 wrapper matching Google's minimal FVD reference."""

    def __init__(self, sequence_length, batch_size=16):
        import tensorflow.compat.v1 as tf
        import tensorflow_hub as hub

        tf.disable_eager_execution()
        self.tf = tf
        self.batch_size = batch_size
        graph = tf.Graph()
        with graph.as_default():
            self.input = tf.placeholder(
                tf.uint8,
                shape=[batch_size, sequence_length, 224, 224, 3],
                name="videos_uint8",
            )
            scaled = 2.0 * tf.cast(self.input, tf.float32) / 255.0 - 1.0
            module = hub.Module(TFHUB_I3D, name="fvd_i3d")
            module(scaled)
            self.output = graph.get_tensor_by_name(
                "fvd_i3d_apply_default/RGB/inception_i3d/Mean:0")
            self.initializer = tf.group(
                tf.global_variables_initializer(),
                tf.tables_initializer(),
            )
        self.graph = graph
        self.session = tf.Session(graph=graph)
        self.session.run(self.initializer)

    def run(self, batch):
        count = len(batch)
        if count == 0 or count > self.batch_size:
            raise ValueError("invalid batch length {}".format(count))
        if count < self.batch_size:
            padding = np.repeat(batch[-1:], self.batch_size - count, axis=0)
            batch = np.concatenate((batch, padding), axis=0)
        features = self.session.run(self.output, feed_dict={self.input: batch})
        return features[:count]

    def close(self):
        self.session.close()


def frechet_distance(real, fake):
    real = np.asarray(real, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)
    if len(real) < 2 or len(fake) < 2:
        raise ValueError("FVD requires at least two samples on each side")
    mean_real = real.mean(axis=0)
    mean_fake = fake.mean(axis=0)
    covariance_real = np.cov(real, rowvar=False)
    covariance_fake = np.cov(fake, rowvar=False)
    covariance_mean = scipy.linalg.sqrtm(
        covariance_real.dot(covariance_fake))
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(covariance_mean.imag, 0, atol=1e-3):
            raise ValueError("covariance square root has a large imaginary part")
        covariance_mean = covariance_mean.real
    difference = mean_real - mean_fake
    value = float(
        difference.dot(difference)
        + np.trace(covariance_real)
        + np.trace(covariance_fake)
        - 2.0 * np.trace(covariance_mean)
    )
    # Fréchet distance is non-negative; round-off in sqrtm can produce a
    # tiny negative value for identical distributions.
    return max(value, 0.0)


def main():
    parser = argparse.ArgumentParser(
        description="Google-reference TF-Hub I3D FVD for PanDWM dumps")
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--expect-clips", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    clips, global_step = load_manifest(args.dump_dir)
    if args.limit_clips is not None:
        clips = clips[:args.limit_clips]
    if args.expect_clips is not None and len(clips) != args.expect_clips:
        raise RuntimeError(
            "expected {} clips, found {}".format(args.expect_clips, len(clips)))
    if not clips:
        raise ValueError("no clips selected")

    embedder = GoogleI3DEmbedder(
        sequence_length=args.sequence_length, batch_size=args.batch_size)
    real_features = [[] for _ in CAMERA_NAMES]
    fake_features = [[] for _ in CAMERA_NAMES]
    try:
        pending_real = [[] for _ in CAMERA_NAMES]
        pending_fake = [[] for _ in CAMERA_NAMES]
        for clip_index, names in enumerate(clips):
            real = load_clip_regions(
                args.dump_dir, "real", names, args.sequence_length)
            fake = load_clip_regions(
                args.dump_dir, "fake", names, args.sequence_length)
            for camera_index in range(len(CAMERA_NAMES)):
                pending_real[camera_index].append(real[camera_index])
                pending_fake[camera_index].append(fake[camera_index])

            should_flush = (
                len(pending_real[0]) == args.batch_size
                or clip_index + 1 == len(clips)
            )
            if should_flush:
                for camera_index in range(len(CAMERA_NAMES)):
                    real_batch = np.stack(pending_real[camera_index])
                    fake_batch = np.stack(pending_fake[camera_index])
                    real_features[camera_index].append(embedder.run(real_batch))
                    fake_features[camera_index].append(embedder.run(fake_batch))
                    pending_real[camera_index].clear()
                    pending_fake[camera_index].clear()
            if (clip_index + 1) % 10 == 0 or clip_index + 1 == len(clips):
                print(
                    "embedded {}/{} clips".format(clip_index + 1, len(clips)),
                    flush=True,
                )
    finally:
        embedder.close()

    real_features = [np.concatenate(parts) for parts in real_features]
    fake_features = [np.concatenate(parts) for parts in fake_features]
    per_camera = {
        name: frechet_distance(real, fake)
        for name, real, fake in zip(
            CAMERA_NAMES, real_features, fake_features)
    }
    pooled_real = np.concatenate(real_features)
    pooled_fake = np.concatenate(fake_features)
    results = {
        "protocol": "google-tfhub-i3d-kinetics400-panorama-owner-regions-v1",
        "reference_implementation": (
            "https://github.com/google-research/google-research/blob/master/"
            "frechet_video_distance/frechet_video_distance.py"
        ),
        "tfhub_module": TFHUB_I3D,
        "dump_dir": os.path.abspath(args.dump_dir),
        "global_step": global_step,
        "clip_count": len(clips),
        "sequence_length": args.sequence_length,
        "pooled_fvd": frechet_distance(pooled_real, pooled_fake),
        "per_camera_fvd": per_camera,
        "camera_macro_fvd": float(np.mean(list(per_camera.values()))),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
