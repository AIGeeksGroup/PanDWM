#!/usr/bin/env bash
# Download only JSON summaries/logs from the two evaluation phases.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REMOTE=/path/to/pandwm_root/pandwm_runs/long_horizon_20260928
LOCAL_BIND_ADDR=${LOCAL_BIND_ADDR:-10.13.138.69}
mkdir -p "$HERE/results/hpc_baseline_dev195_v1" \
         "$HERE/results/hpc_diagnostic_dev195_v1"
for phase in baseline_dev195_v1 diagnostic_dev195_v1; do
  rsync -az \
    --include='*/' --include='summary.json' --include='config.json' \
    --exclude='*' \
    -e "ssh -b $LOCAL_BIND_ADDR -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes" \
    "hpc:$REMOTE/$phase/" "$HERE/results/hpc_$phase/"
done
for phase in baseline_dev195_v1 diagnostic_dev195_v1; do
  taskfile=tasks_baseline_dev195.json
  if [ "$phase" = diagnostic_dev195_v1 ]; then
    taskfile=tasks_diagnostic_dev195.json
  fi
  PYTHONPATH="$HERE/src" python3 "$HERE/summarize_eval_tasks.py" \
    --tasks "$HERE/$taskfile" --results-root "$HERE/results/hpc_$phase" \
    --output "$HERE/results/hpc_${phase}_summary.json"
done
