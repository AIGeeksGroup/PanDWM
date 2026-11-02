#!/usr/bin/env bash
# 2026-10-15: HPC3 login-node loop, so the I-4500 and K-4500 evaluations advance while the laptop session is idle
# (the session heartbeat did not fire overnight 10-14 -> 10-15 and the pipeline stalled for ~8 h).
#   setsid nohup bash watch_loop_ik.sh > /dev/null 2>&1 < /dev/null &
# Calls watch_v85.sh for I 4500 (prev tag v84i4250) every 10 min until it prints DONE v84i4500, saves that output to
# $E/done_v84i4500.txt and touches $E/.done_v84i4500; only then calls it for K 4500 (prev tag v84i4500), because the
# K call renames the shared Dev-149 metrics files; stops after DONE v84k4500 ($E/done_v84k4500.txt).  Log:
# $E/watch_loop_ik.log.  Stop it early with: touch $E/.stop_watch_loop_ik
set -uo pipefail
E=/path/to/pandwm_root/pandwm_runs/eval149_hpc3_v83_20261012
CF=/path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008
LOG=$E/watch_loop_ik.log
cd "$E" || exit 1
echo "[$(date -Is)] LOOP_START pid=$$" >> "$LOG"
while [ ! -e "$E/.stop_watch_loop_ik" ]; do
    if [ ! -e "$E/.done_v84i4500" ]; then
        out=$(BR=I I_LOG=$CF/pandwm_cfdmd_I_716489.log bash "$E/watch_v85.sh" 4500 v84i4250 2>&1)
        echo "[$(date -Is)] I: $(echo "$out" | grep -E 'STATUS|SUBMITTED|LAUNCHED|FAILED|RETRIED|WAIT|^DONE' | tr '\n' ' ')" >> "$LOG"
        if echo "$out" | grep -q "^DONE v84i4500"; then
            echo "$out" > "$E/done_v84i4500.txt"
            touch "$E/.done_v84i4500"
            echo "[$(date -Is)] I_DONE" >> "$LOG"
            continue
        fi
    else
        out=$(BR=K K_LOG=$CF/pandwm_cfdmd_K_717787.log bash "$E/watch_v85.sh" 4500 v84i4500 2>&1)
        echo "[$(date -Is)] K: $(echo "$out" | grep -E 'STATUS|SUBMITTED|LAUNCHED|FAILED|RETRIED|WAIT|^DONE' | tr '\n' ' ')" >> "$LOG"
        if echo "$out" | grep -q "^DONE v84k4500"; then
            echo "$out" > "$E/done_v84k4500.txt"
            touch "$E/.done_v84k4500"
            echo "[$(date -Is)] K_DONE LOOP_END" >> "$LOG"
            exit 0
        fi
    fi
    sleep 600
done
echo "[$(date -Is)] LOOP_STOPPED" >> "$LOG"
