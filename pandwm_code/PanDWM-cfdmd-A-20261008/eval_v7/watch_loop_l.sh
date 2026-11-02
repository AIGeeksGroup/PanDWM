#!/usr/bin/env bash
# 2026-10-16: HPC3 login-node loop for the branch-L (v8.6) step-4500 evaluation (report section 30, "下一步建议" A),
# so it advances while the laptop session is idle (session crons are unreliable).
#   cd $E; setsid nohup bash watch_loop_l.sh > /dev/null 2>&1 < /dev/null &
# Every 10 min calls watch_v86.sh (BR=L, previous tag v84k4500 = the owner of the shared Dev-149 metrics files) until it
# prints DONE v86l4500, then saves that output to $E/done_v86l4500.txt and touches $E/.done_v86l4500.  If the training
# job has left the queue without a 4500 milestone it logs TRAINING_GONE_NO_MILESTONE and stops.  Log:
# $E/watch_loop_l.log.  Stop early: touch $E/.stop_watch_loop_l
set -uo pipefail
E=/path/to/pandwm_root/pandwm_runs/eval149_hpc3_v83_20261012
CF=/path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008
RUN=/path/to/pandwm_root/pandwm_runs/cfdmd_20261016_L
LLOG=${LLOG:?set LLOG to the branch-L training log}
LOG=$E/watch_loop_l.log
cd "$E" || exit 1
echo "[$(date -Is)] LOOP_START pid=$$ training_log=$LLOG" >> "$LOG"
while [ ! -e "$E/.stop_watch_loop_l" ]; do
    if [ -e "$E/.done_v86l4500" ]; then echo "[$(date -Is)] ALREADY_DONE" >> "$LOG"; exit 0; fi
    if [ ! -f "$RUN/gan/milestones/4500_ema.pth" ] && ! squeue -u "$USER" -h -n pandwm_cfdmd_L | grep -q .; then
        echo "[$(date -Is)] TRAINING_GONE_NO_MILESTONE" >> "$LOG"; exit 1
    fi
    out=$(BR=L L_LOG=$LLOG bash "$E/watch_v86.sh" 4500 v84k4500 2>&1)
    echo "[$(date -Is)] L: $(echo "$out" | grep -E 'STATUS|SUBMITTED|LAUNCHED|FAILED|RETRIED|WAIT|MISSING|^DONE' | tr '\n' ' ')" >> "$LOG"
    if echo "$out" | grep -q "^DONE v86l4500"; then
        echo "$out" > "$E/done_v86l4500.txt"
        touch "$E/.done_v86l4500"
        echo "[$(date -Is)] L_DONE LOOP_END" >> "$LOG"
        exit 0
    fi
    sleep 600
done
echo "[$(date -Is)] LOOP_STOPPED" >> "$LOG"
