"""CPU streaming checkpoints and committed-manifest validation."""
import json
import os
from pathlib import Path

import torch


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v) for v in value)
    return value


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(cpu_state(value), stream,
                   _use_new_zipfile_serialization=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_manifest(output_path, step, world_size):
    root = Path(output_path)
    manifest = json.loads((root / "checkpoints" /
                           f"{step}.complete.json").read_text())
    if manifest["step"] != step or manifest["world_size"] != world_size:
        raise ValueError("Resume requires the saved step and the same GPU count")
    for relative, size in manifest["files"].items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Incomplete checkpoint file: {path}")
    return manifest


def latest_step(output_path, world_size=8):
    manifests = (Path(output_path) / "checkpoints").glob("*.complete.json")
    steps = sorted((int(p.name.split(".")[0]) for p in manifests), reverse=True)
    for step in steps:
        try:
            validate_manifest(output_path, step, world_size)
            return step
        except (ValueError, OSError, KeyError):
            continue
    return None
