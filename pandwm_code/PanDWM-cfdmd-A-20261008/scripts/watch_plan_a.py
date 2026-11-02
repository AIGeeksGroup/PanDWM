"""Persist Slurm state and both arm progress locally, independently of chat."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--job", required=True)
parser.add_argument("--interval", type=int, default=60)
args = parser.parse_args()
destination = Path(__file__).resolve().parents[1] / "progress"
destination.mkdir(exist_ok=True)
remote = '''import json, subprocess
from pathlib import Path
root=Path("/path/to/pandwm_root/pandwm_runs/cfdmd_20261008_A")
result={"accounting":subprocess.check_output(["sacct","-X","-j","JOB_ID","-n","-o","JobID,State%20,Elapsed,ExitCode"],text=True).strip(),"paused":(root/"PAUSE_REQUESTED").exists()}
for arm in ["gan","nogan"]:
 result[arm]={}
 for name in ["progress","latest_checkpoint"]:
  path=root/arm/"log"/(name+".json")
  if path.exists():
   result[arm][name]=json.loads(path.read_text())
log=Path("/path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008/pandwm_cfdmd_A_JOB_ID.log")
if log.exists():result["log_tail"]=log.read_text(errors="replace")[-5000:]
print(json.dumps(result))
'''.replace("JOB_ID", args.job)
while True:
    status = {"job": args.job, "checked_at": datetime.datetime.now().astimezone().isoformat()}
    try:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                                 "-o", "StrictHostKeyChecking=yes", "hpc3",
                                 "python3 - <<'PY'\n" + remote + "\nPY"],
                                capture_output=True, text=True, timeout=35)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        status.update(json.loads(result.stdout))
    except Exception as error:
        status["check_error"] = str(error)
    temporary = destination / "current.json.tmp"
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, destination / "current.json")
    with (destination / "timeline.jsonl").open("a") as stream:
        stream.write(json.dumps(status, ensure_ascii=False) + "\n")
    accounting = status.get("accounting", "")
    if any(state in accounting for state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"]):
        break
    time.sleep(args.interval)
