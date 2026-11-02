#!/usr/bin/env bash
# One idempotent step of the evaluation of a branch-H milestone (2026-10-12).  Runs on the HPC3 login node:
#   bash watch_v83.sh <step> <best previous step>          e.g. bash watch_v83.sh 2500 2250
# Prints one status line and exits; call it repeatedly.  Marker files in EVAL_OUT stop duplicate submissions and
# launches: .submitted_<tag> holds "<gen array> <fin1>", .flicker_<tag> marks the flicker_diag launch.
#   1) milestone linked -> move the previous arm's metrics state/log aside (fin1 refuses a state of another arm set
#      and would report the old report.json), submit the 8-task generation array (%4) and fin1 after it;
#   2) generation done -> launch flicker_diag.py once (setsid, all fds redirected);
#   3) fin1 wrote a report for this tag and flicker_diag finished -> save report_<tag>.json, print the summary, DONE.
# 2026-10-13: a failed sbatch (Slurm controller timeouts happen) no longer leaves a broken marker: a failed generation
# submit removes the marker so the next call retries; a failed fin1 submit records "<gen> pending_fin1" and the next
# call submits fin1 only.  No new generation array is submitted while one of ours (pandwm_e149g) is queued or running.
set -uo pipefail
S=${1:?step}; P=${2:?previous step}
R=/path/to/pandwm_root/pandwm_runs
E=$R/eval149_hpc3_v83_20261012
CF=/path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008
C=/path/to/pandwm_root/pandwm_code/PanDWM_eval149_20261011
PY=/path/to/pandwm_root/pandwm_venv/bin/python
M=$R/cfdmd_20261010_H/gan/milestones/${S}_ema.pth
TAG=v83h$S; PTAG=v83h$P
LOG=${HX_LOG:-$CF/pandwm_cfdmd_Hx_710735.log}   # steps after 3000: HX_LOG=<the Hx2 job log>

if [ ! -e "$E/.submitted_$TAG" ]; then
    if [ -f "$M" ] && grep -q "MILESTONE_LINKED step=$S" "$LOG"; then
        echo "pending pending" > "$E/.submitted_$TAG"
        [ -e "$E/metrics/state_$PTAG.pt" ] || { [ -e "$E/metrics/state_0_149.pt" ] && mv "$E/metrics/state_0_149.pt" "$E/metrics/state_$PTAG.pt"; }
        [ -e "$E/metrics/metrics_$PTAG.log" ] || { [ -e "$E/metrics/metrics.log" ] && mv "$E/metrics/metrics.log" "$E/metrics/metrics_$PTAG.log"; }
        [ -e "$E/metrics/report_$PTAG.json" ] || cp -p "$E/metrics/report.json" "$E/metrics/report_$PTAG.json"
        [ -e "$E/realism_arms_$PTAG.json" ] || cp -p "$E/realism_arms.json" "$E/realism_arms_$PTAG.json"
        cd "$CF" || exit 1
        if [ -n "$(squeue -u "$USER" -h -n pandwm_e149g -o %i 2>/dev/null)" ]; then
            rm -f "$E/.submitted_$TAG"; echo "WAIT_GEN_BUSY $TAG (a pandwm_e149g job is queued or running)"; exit 0
        fi
        G=$(ARMS="$TAG=$M" EVAL_OUT=$E sbatch --parsable --array=0-7%4 scripts/eval149_hpc3_gen.sbatch)
        G=${G%%;*}
        case "$G" in ''|*[!0-9]*) rm -f "$E/.submitted_$TAG"; echo "SUBMIT_FAILED $TAG gen (will retry)"; exit 0;; esac
        F=$(TAGS="$TAG" EVAL_OUT=$E sbatch --parsable --dependency=afterany:$G scripts/eval149_hpc3_fin1.sbatch)
        F=${F%%;*}
        case "$F" in ''|*[!0-9]*) echo "$G pending_fin1" > "$E/.submitted_$TAG"; echo "SUBMIT_FAILED $TAG fin1 gen=$G (will retry fin1)"; exit 0;; esac
        echo "$G $F" > "$E/.submitted_$TAG"
        echo "SUBMITTED $TAG gen=$G fin1=$F"
    else
        echo "WAIT_MILESTONE $TAG at $(grep 'Step ' "$LOG" | tail -1 | cut -c1-10)"
    fi
    exit 0
fi
read -r G F < "$E/.submitted_$TAG"
if [ "$G" = pending ]; then echo "SUBMIT_IN_PROGRESS $TAG"; exit 0; fi
if [ "$F" = pending_fin1 ]; then
    cd "$CF" || exit 1
    F=$(TAGS="$TAG" EVAL_OUT=$E sbatch --parsable --dependency=afterany:$G scripts/eval149_hpc3_fin1.sbatch)
    F=${F%%;*}
    case "$F" in ''|*[!0-9]*) echo "SUBMIT_FAILED $TAG fin1 gen=$G (will retry fin1)"; exit 0;; esac
    echo "$G $F" > "$E/.submitted_$TAG"
    echo "SUBMITTED $TAG fin1=$F (retry, gen=$G)"
    exit 0
fi
g=$(cat "$CF"/logs_eval149/pandwm_e149g_"${G}"_*.log 2>/dev/null | grep -c E149G_DONE)
if [ "$g" = 8 ] && [ ! -e "$E/.flicker_$TAG" ]; then
    touch "$E/.flicker_$TAG"
    ARMS_LIST="b0 v82g2000 v83h2000"
    for s in 2250 2500 2750 3000 3250 3500 3750 4000; do
        [ -d "$E/v83h$s" ] && [ "$(ls "$E/v83h$s" | wc -l)" -ge 149 ] && ARMS_LIST="$ARMS_LIST v83h$s"
    done
    cd "$C" || exit 1
    setsid nohup nice -n 10 "$PY" flicker_diag.py "$E" "$E/flicker_diag_$TAG.json" --pair "$PTAG" "$TAG" $ARMS_LIST \
        > "$E/flicker_diag_$TAG.log" 2>&1 < /dev/null &
    echo "FLICKER_LAUNCHED $TAG ($ARMS_LIST)"
    exit 0
fi
st=$(sacct -j "$F" -n -X -o State | head -1 | tr -d ' ')
fl=0; [ -s "$E/flicker_diag_$TAG.json" ] && fl=1
ok=0; grep -q METRICS_DONE "$E/metrics/metrics.log" 2>/dev/null && grep -q "\"$TAG\"" "$E/metrics/report.json" && ok=1
echo "STATUS $TAG gen_done=$g/8 fin1=$st metrics_ok=$ok flicker=$fl"
if [ "$ok" = 1 ] && [ "$fl" = 1 ]; then
    cp -p "$E/metrics/report.json" "$E/metrics/report_$TAG.json"
    "$PY" "$E/summary_v83.py" "$TAG" "$PTAG"
    echo "DONE $TAG"
fi
