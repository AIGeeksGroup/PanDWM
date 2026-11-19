"""Hard-link one committed checkpoint of a run into a new output directory, so a branch can resume from it with
another config:  python3 scripts/seed_branch.py <source gan dir> <branch gan dir> <step>

Links every file the checkpoint manifest lists (model, EMA, critic, discriminator heads, rank-local optimizer and
RNG state) plus the manifest itself; nothing is copied or modified, and the source run is unaffected (its later
pruning only removes its own directory entries).  Writes BRANCH_OF.json next to the branch's checkpoints.
"""
import json
import os
import sys
import time
from pathlib import Path


def main():
    src, dst, step = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    manifest = src / "checkpoints" / f"{step}.complete.json"
    meta = json.loads(manifest.read_text())
    if meta["step"] != step:
        raise SystemExit(f"manifest step {meta['step']} != {step}")
    for rel, size in meta["files"].items():
        a, b = src / rel, dst / rel
        if a.stat().st_size != size:
            raise SystemExit(f"size mismatch: {a}")
        b.parent.mkdir(parents=True, exist_ok=True)
        if b.exists():
            if os.path.samefile(a, b):
                continue
            raise SystemExit(f"exists and is a different file: {b}")
        os.link(a, b)
    target = dst / "checkpoints" / manifest.name
    if not target.exists():
        os.link(manifest, target)
    (dst / "log").mkdir(parents=True, exist_ok=True)
    (dst / "BRANCH_OF.json").write_text(json.dumps(
        {"source": str(src), "step": step, "files": len(meta["files"]),
         "seeded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n")
    print(f"SEEDED {dst} from {src} step {step}: {len(meta['files'])} files hard-linked", flush=True)


if __name__ == "__main__":
    main()
