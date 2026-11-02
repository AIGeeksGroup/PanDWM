"""Persist campaign status on the login host without depending on this chat."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--job", required=True)
args = parser.parse_args()
root = Path("/path/to/pandwm_root/pandwm_runs/cfdmd_20261008_A")
destination = root / "monitor"
destination.mkdir(exist_ok=True)
while True:
    status = {"job": args.job, "checked_at": datetime.datetime.now().astimezone().isoformat(),
              "paused": (root / "PAUSE_REQUESTED").exists()}
    try:
        status["accounting"] = subprocess.check_output(
            ["sacct", "-X", "-j", args.job, "-n", "-o", "JobID,State%20,Elapsed,ExitCode"],
            universal_newlines=True, timeout=20).strip()
        for arm in ["gan", "nogan"]:
            status[arm] = {}
            for name in ["progress", "latest_checkpoint"]:
                path = root / arm / "log" / (name + ".json")
                if path.exists():
                    status[arm][name] = json.loads(path.read_text())
        log = Path("/path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008") / (
            "pandwm_cfdmd_A_" + args.job + ".log")
        if log.exists():
            status["log_tail"] = log.read_text(errors="replace")[-5000:]
    except Exception as error:
        status["check_error"] = str(error)
    temporary = destination / "current.json.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(str(temporary), str(destination / "current.json"))
    with (destination / "timeline.jsonl").open("a") as stream:
        stream.write(json.dumps(status) + "\n")
    if any(state in status.get("accounting", "") for state in
           ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"]):
        break
    time.sleep(60)
