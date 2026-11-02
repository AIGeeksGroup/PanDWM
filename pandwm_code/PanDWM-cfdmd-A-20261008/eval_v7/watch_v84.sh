#!/usr/bin/env bash
# One idempotent step of the evaluation of a branch-I (v8.4) milestone (2026-10-13).  Runs on the HPC3 login node:
#   I_LOG=<branch-I training log> bash watch_v84.sh <step> <previous tag in EVAL_OUT>   e.g. ... 4250 v83h4000
# Prints status lines and exits; call it repeatedly.  Same EVAL_OUT and steps as watch_v83.sh (8-task generation
# array %4, fin1 after it, flicker_diag on the login node), plus
#   - Dev-78 metrics (scripts/eval149_hpc3_dev78.sbatch, LABEL i<step>) after the generation array;
#   - 24 long videos (12 non-test scenes x seeds 42 / 43; scripts/long_video_hpc3_v2.sbatch, 6 one-GPU jobs);
#   - once everything is in: the pre-registered gate of V82_EVAL_HPC3_20261011.md section 25 (long4_arms.py vs
#     v83h2000 for L1-L3, Dev-78 FVD sum vs v83h2000 for S1) and the training-monitor summary; then DONE.
# Markers in EVAL_OUT: .submitted_<tag> "<gen> <fin1> <dev78>", .lv_<tag> (long-video job ids), .flicker_<tag>,
# .gate_<tag> (long4 launched).  No generation array is submitted while one of ours (pandwm_e149g) is queued.
set -uo pipefail
S=${1:?step}; PTAG=${2:?previous tag}
R=/path/to/pandwm_root/pandwm_runs
E=$R/eval149_hpc3_v83_20261012
CF=/path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008
C=/path/to/pandwm_root/pandwm_code/PanDWM_eval149_20261011
LV=$R/long_videos_hpc3_20261011
PY=/path/to/pandwm_root/pandwm_venv/bin/python
M=$R/cfdmd_20261013_I/gan/milestones/${S}_ema.pth
TAG=v84i$S
LOG=${I_LOG:?set I_LOG to the branch-I training log}
LVGROUPS=("42:0 14 21 32" "42:41 49 62 82" "42:10 31 52 72" "43:0 14 21 32" "43:41 49 62 82" "43:10 31 52 72")
D78=$E/metrics_dev78

if [ ! -e "$E/.submitted_$TAG" ]; then
    if [ -f "$M" ] && grep -q "MILESTONE_LINKED step=$S" "$LOG"; then
        if [ -n "$(squeue -u "$USER" -h -n pandwm_e149g -o %i 2>/dev/null)" ]; then
            echo "WAIT_GEN_BUSY $TAG (a pandwm_e149g job is queued or running)"; exit 0
        fi
        echo "pending pending pending" > "$E/.submitted_$TAG"
        [ -e "$E/metrics/state_$PTAG.pt" ] || { [ -e "$E/metrics/state_0_149.pt" ] && mv "$E/metrics/state_0_149.pt" "$E/metrics/state_$PTAG.pt"; }
        [ -e "$E/metrics/metrics_$PTAG.log" ] || { [ -e "$E/metrics/metrics.log" ] && mv "$E/metrics/metrics.log" "$E/metrics/metrics_$PTAG.log"; }
        [ -e "$E/metrics/report_$PTAG.json" ] || cp -p "$E/metrics/report.json" "$E/metrics/report_$PTAG.json"
        [ -e "$E/realism_arms_$PTAG.json" ] || cp -p "$E/realism_arms.json" "$E/realism_arms_$PTAG.json"
        cd "$CF" || exit 1
        G=$(ARMS="$TAG=$M" EVAL_OUT=$E sbatch --parsable --array=0-7%4 scripts/eval149_hpc3_gen.sbatch)
        G=${G%%;*}
        case "$G" in ''|*[!0-9]*) rm -f "$E/.submitted_$TAG"; echo "SUBMIT_FAILED $TAG gen (will retry)"; exit 0;; esac
        F=$(TAGS="$TAG" EVAL_OUT=$E sbatch --parsable --dependency=afterany:$G scripts/eval149_hpc3_fin1.sbatch)
        F=${F%%;*}
        case "$F" in ''|*[!0-9]*) F=pending_fin1;; esac
        D=$(TAGS="$TAG" LABEL="i$S" EVAL_OUT=$E sbatch --parsable --dependency=afterany:$G scripts/eval149_hpc3_dev78.sbatch)
        D=${D%%;*}
        case "$D" in ''|*[!0-9]*) D=pending_dev78;; esac
        echo "$G $F $D" > "$E/.submitted_$TAG"
        echo "SUBMITTED $TAG gen=$G fin1=$F dev78=$D"
    else
        echo "WAIT_MILESTONE $TAG at $(grep 'Step ' "$LOG" | tail -1 | cut -c1-12)"
    fi
    exit 0
fi
read -r G F D < "$E/.submitted_$TAG"
if [ "$G" = pending ]; then echo "SUBMIT_IN_PROGRESS $TAG"; exit 0; fi
cd "$CF" || exit 1
if [ "$F" = pending_fin1 ]; then
    F=$(TAGS="$TAG" EVAL_OUT=$E sbatch --parsable --dependency=afterany:$G scripts/eval149_hpc3_fin1.sbatch)
    F=${F%%;*}; case "$F" in ''|*[!0-9]*) F=pending_fin1;; esac
    echo "$G $F $D" > "$E/.submitted_$TAG"; echo "RETRIED fin1=$F"
fi
if [ "$D" = pending_dev78 ]; then
    D=$(TAGS="$TAG" LABEL="i$S" EVAL_OUT=$E sbatch --parsable --dependency=afterany:$G scripts/eval149_hpc3_dev78.sbatch)
    D=${D%%;*}; case "$D" in ''|*[!0-9]*) D=pending_dev78;; esac
    echo "$G $F $D" > "$E/.submitted_$TAG"; echo "RETRIED dev78=$D"
fi
if [ ! -e "$E/.lv_$TAG" ]; then
    ids=""
    for spec in "${LVGROUPS[@]}"; do
        SEED=${spec%%:*}; SC=${spec#*:}
        J=$(TAG=$TAG CKPT=$M SCENES="$SC" SEED=$SEED sbatch --parsable scripts/long_video_hpc3_v2.sbatch)
        J=${J%%;*}; case "$J" in ''|*[!0-9]*) echo "SUBMIT_FAILED $TAG long videos seed$SEED [$SC]"; J=failed;; esac
        ids="$ids $J"
    done
    echo "$ids" > "$E/.lv_$TAG"
    echo "LONG_VIDEOS_SUBMITTED $TAG:$ids"
    case "$ids" in *failed*) rm -f "$E/.lv_$TAG";; esac
fi
g=$(cat "$CF"/logs_eval149/pandwm_e149g_"${G}"_*.log 2>/dev/null | grep -c E149G_DONE)
if [ "$g" = 8 ] && [ ! -e "$E/.flicker_$TAG" ]; then
    touch "$E/.flicker_$TAG"
    cd "$C" || exit 1
    setsid nohup nice -n 10 "$PY" flicker_diag.py "$E" "$E/flicker_diag_$TAG.json" --pair "$PTAG" "$TAG" \
        b0 v82g2000 v83h2000 v83h4000 "$TAG" > "$E/flicker_diag_$TAG.log" 2>&1 < /dev/null &
    echo "FLICKER_LAUNCHED $TAG"
    exit 0
fi
fst=$(sacct -j "$F" -n -X -o State 2>/dev/null | head -1 | tr -d ' ')
fl=0; [ -s "$E/flicker_diag_$TAG.json" ] && fl=1
ok=0; grep -q METRICS_DONE "$E/metrics/metrics.log" 2>/dev/null && grep -q "\"$TAG\"" "$E/metrics/report.json" && ok=1
d78=0; grep -q METRICS_DONE "$D78/metrics_i$S.log" 2>/dev/null && [ -s "$D78/report_i$S.json" ] && d78=1
ref78=0; grep -q METRICS_DONE "$D78/metrics_h2000.log" 2>/dev/null && [ -s "$D78/report_h2000.json" ] && ref78=1
nlv=0
for spec in "${LVGROUPS[@]}"; do
    SEED=${spec%%:*}
    for sc in ${spec#*:}; do [ -f "$LV/${TAG}_scene${sc}_seed${SEED}/gt_vs_generated.mp4" ] && nlv=$((nlv + 1)); done
done
nref=0
for a in v83h2000 v83h4000; do
    for spec in "${LVGROUPS[@]}"; do
        SEED=${spec%%:*}
        for sc in ${spec#*:}; do [ -f "$LV/${a}_scene${sc}_seed${SEED}/gt_vs_generated.mp4" ] && nref=$((nref + 1)); done
    done
done
echo "STATUS $TAG gen_done=$g/8 fin1=$fst metrics_ok=$ok dev78=$d78 ref_dev78=$ref78 flicker=$fl long=$nlv/24 ref_long=$nref/48"
if [ "$nlv" = 24 ] && [ "$nref" = 48 ] && [ ! -e "$E/.gate_$TAG" ]; then
    touch "$E/.gate_$TAG"
    cd "$C" || exit 1
    setsid nohup nice -n 10 "$PY" long4_arms.py "$LV" "$LV/long4_$TAG.json" --scenes 0 14 21 32 41 49 62 82 10 31 52 72 \
        --seeds 42 43 --arms v83h2000 v83h4000 "$TAG" --ref v83h2000 > "$LV/long4_$TAG.log" 2>&1 < /dev/null &
    echo "LONG4_LAUNCHED $TAG"
    exit 0
fi
l4=0; grep -q LONG4_DONE "$LV/long4_$TAG.log" 2>/dev/null && l4=1
if [ "$ok" = 1 ] && [ "$fl" = 1 ] && [ "$d78" = 1 ] && [ "$ref78" = 1 ] && [ "$l4" = 1 ]; then
    cp -p "$E/metrics/report.json" "$E/metrics/report_$TAG.json"
    "$PY" "$E/summary_v83.py" "$TAG" "$PTAG"
    grep -E "videos|^sharp/GT|^change/GT|late sharp/GT per scene|^  v8|^GATE" "$LV/long4_$TAG.log"
    "$PY" - "$D78/report_h2000.json" "$D78/report_i$S.json" "$TAG" <<'PY'
import json, sys
ref = json.load(open(sys.argv[1]))["arms"]["v83h2000"]
cur = json.load(open(sys.argv[2]))["arms"][sys.argv[3]]
fr, fc = ref["fvd_w1"] + ref["fvd_w2"], cur["fvd_w1"] + cur["fvd_w2"]
print(f"S1 Dev-78 FVD sum {sys.argv[3]} {fc:.1f} vs v83h2000 {fr:.1f} (needs <= {fr - 1.0:.1f}): "
      f"{'ok' if fc <= fr - 1.0 else 'FAIL'}; FID {cur['fid_all']:.2f} vs {ref['fid_all']:.2f}")
PY
    "$PY" "$CF/eval_v7/v84_monitor_summary.py" "$LOG" --bins 4001-4100 4101-4250 2>&1 | tail -30
    echo "DONE $TAG"
fi
