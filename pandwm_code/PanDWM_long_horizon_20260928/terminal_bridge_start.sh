#!/usr/bin/env bash
# Run from the user's normal terminal when Codex's tool network is restricted.
# Submits only read-only baseline/diagnostic evaluation; training is separate.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$HERE/results"
LOG=$HERE/results/hpc_submissions_$(date +%Y%m%d_%H%M%S).log
{
  date '+%Y-%m-%d %H:%M:%S %Z'
  bash "$HERE/deploy_and_submit.sh" baseline
  bash "$HERE/deploy_and_submit.sh" diagnostic
  ssh -o BatchMode=yes -o ConnectTimeout=10 hpc \
    'squeue -u "$USER" -o "%i %j %T %M %D %R" | head -60'
} 2>&1 | tee "$LOG"
echo "submission_log=$LOG"
