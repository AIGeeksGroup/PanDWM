"""Check campaign decisions before importing any training dependencies."""
import argparse
import datetime
import json
import os
from pathlib import Path


def skip_reason(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-o", "--output-path")
    args, _ = parser.parse_known_args(argv)
    campaign = os.environ.get("CAMPAIGN_DIR")
    if not campaign or not args.output_path:
        return None
    root = Path(campaign).resolve()
    output = Path(args.output_path).resolve()
    if output != root / "nogan" or not (root / "SKIP_NOGAN").exists():
        return None
    return output


def record_skip(output):
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) != 0:
        return
    log = output / "log"
    log.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "skipped",
        "step": 0,
        "reason": "User chose GAN only; compare against the original B0 student.",
        "updated_at": datetime.datetime.now().astimezone().isoformat(),
    }
    temporary = log / "progress.json.skip.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(str(temporary), str(log / "progress.json"))
