#!/usr/bin/env python3
"""Register Vista's official OpenCLIP dependency in an offline HF cache."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
FILENAME = "open_clip_pytorch_model.bin"
REVISION = "1c2b8495b28150b8a4922ee1c8edee224c284c0c"
SHA256 = "9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4"
SIZE = 3_944_692_325


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--hf-home", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    if source.stat().st_size != SIZE:
        raise ValueError("OpenCLIP size mismatch: {}".format(source.stat().st_size))
    actual = sha256(source)
    if actual != SHA256:
        raise ValueError("OpenCLIP SHA256 mismatch: {}".format(actual))

    model_dir = args.hf_home / "hub" / (
        "models--" + MODEL_ID.replace("/", "--"))
    blob = model_dir / "blobs" / SHA256
    snapshot = model_dir / "snapshots" / REVISION / FILENAME
    reference = model_dir / "refs" / "main"
    blob.parent.mkdir(parents=True, exist_ok=True)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    reference.parent.mkdir(parents=True, exist_ok=True)

    if blob.exists() or blob.is_symlink():
        if blob.resolve() != source:
            raise FileExistsError("cache blob points elsewhere: {}".format(blob))
    else:
        blob.symlink_to(source)
    if snapshot.exists() or snapshot.is_symlink():
        if snapshot.resolve() != source:
            raise FileExistsError(
                "snapshot points elsewhere: {}".format(snapshot))
    else:
        snapshot.symlink_to(Path("../../blobs") / SHA256)
    if reference.exists() and reference.read_text(encoding="utf-8").strip() != REVISION:
        raise FileExistsError("refs/main contains a different revision")
    # huggingface_hub treats the whole refs file as the commit identifier and
    # does not strip a trailing newline in its cache scanner.
    reference.write_text(REVISION, encoding="utf-8")
    print("registered", MODEL_ID, REVISION)
    print("snapshot", snapshot)
    print("source", source)


if __name__ == "__main__":
    main()
