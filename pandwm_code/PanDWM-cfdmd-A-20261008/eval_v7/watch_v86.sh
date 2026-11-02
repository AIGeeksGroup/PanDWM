#!/usr/bin/env bash
# watch_v86.sh (2026-10-16) = watch_v85.sh + BR=L (branch L, v8.6: tag v86l<step>, Dev-78 label l<step>).  For BR=L
# only the EMA weights get Dev-76-L rollouts (raw weights were clearly worse for I and K), and the Dev-76-L analysis
# arms default to "v84i4250 <tag> v84i4500" (v84i4250 first = the reference of the intervals; I 4500 = same start,
# steps and data).  J / I / K behave exactly as in watch_v85.sh.  2026-10-16 23:50: BR=M (branch M, v8.6 with the
# rollout cap 1.0: tag v86m<step>, label m<step>) is handled exactly like BR=L.
# One idempotent step of the evaluation of a branch-J (v8.5) milestone (2026-10-14).  Runs on the HPC3 login node:
#   J_LOG=<branch-J training log> bash watch_v85.sh <step> <previous tag in EVAL_OUT>   e.g. ... 4500 v84i4250
#   BR=I I_LOG=<branch-I log> bash watch_v85.sh 4500 v84i4250    the same protocol for a branch-I (v8.4) milestone
#   (BR=I: tag v84i<step>, Dev-78 label i<step>; BR=K: tag v84k<step>, label k<step>; default BR=J)
#   AN_ARMS="..." overrides the Dev-76-L analysis arms (default: v84i4250 <tag> <tag>raw v83h4000; for BR=K
#   v84i4500 v84i4500raw <tag> <tag>raw v84i4250, i.e. K is compared with I 4500, the same start and steps).
# watch_v84.sh with the long-video part moved to Dev-76-L (the user's decision of 2026-10-14, report section 27.9):
#   - the same 8-task generation array, fin1, Dev-78 metrics (LABEL j<step>, i<step> or k<step>) and login-node flicker_diag;
#   - Dev-76-L rollouts of the EMA weights (tag v85j<step>) and of the raw weights (v85j<step>raw), 4 array tasks
#     each (scripts/long_video_dev76.sbatch, CHUNK 19), then scripts/long_dev76_analyze.sbatch for
#     v84i4250 v85j<step> v85j<step>raw v83h4000 (features of the first and last are reused);
#   - DONE prints the 149-clip summary, the Dev-78 FVD sums vs v83h2000 and v84i4250, the Dev-76-L buckets,
#     intervals and split-half realism, and the CFG schedule seen in the log (V85 lines).
# Markers in EVAL_OUT: .submitted_<tag> "<gen> <fin1> <dev78>", .lv76_<tag>, .l76an_<tag>, .flicker_<tag>.
set -uo pipefail
S=${1:?step}; PTAG=${2:?previous tag}
R=/path/to/pandwm_root/pandwm_runs
E=$R/eval149_hpc3_v83_20261012
CF=/path/to/pandwm_root/pandwm_code/PanDWM-cfdmd-A-20261008
C=/path/to/pandwm_root/pandwm_code/PanDWM_eval149_20261011
LV=$R/long_videos_hpc3_20261011
PY=/path/to/pandwm_root/pandwm_venv/bin/python
BR=${BR:-J}
case "$BR" in
    J) RUN=$R/cfdmd_20261014_J; TAG=v85j$S; LAB=j$S; LOG=${J_LOG:?set J_LOG to the branch-J training log};;
    I) RUN=$R/cfdmd_20261013_I; TAG=v84i$S; LAB=i$S; LOG=${I_LOG:?set I_LOG to the branch-I training log};;
    K) RUN=$R/cfdmd_20261014_K; TAG=v84k$S; LAB=k$S; LOG=${K_LOG:?set K_LOG to the branch-K training log};;
    L) RUN=$R/cfdmd_20261016_L; TAG=v86l$S; LAB=l$S; LOG=${L_LOG:?set L_LOG to the branch-L training log};;
    M) RUN=$R/cfdmd_20261016_M; TAG=v86m$S; LAB=m$S; LOG=${M_LOG:?set M_LOG to the branch-M training log};;
    *) echo "UNKNOWN_BRANCH $BR"; exit 1;;
esac
if [ "$BR" = K ]; then AN_ARMS=${AN_ARMS:-"v84i4500 v84i4500raw $TAG ${TAG}raw v84i4250"}
elif [ "$BR" = L ] || [ "$BR" = M ]; then AN_ARMS=${AN_ARMS:-"v84i4250 $TAG v84i4500"}
else AN_ARMS=${AN_ARMS:-"v84i4250 $TAG ${TAG}raw v83h4000"}; fi
FL_ARMS=$(echo b0 v82g2000 v83h2000 v83h4000 v84i4250 "$PTAG" "$TAG" | tr " " "\n" | awk '!seen[$0]++' | tr "\n" " ")
NO_RAW=0; { [ "$BR" = L ] || [ "$BR" = M ]; } && NO_RAW=1      # watch_v86: EMA only on Dev-76-L for branches L / M
M=$RUN/gan/milestones/${S}_ema.pth
MRAW=$RUN/gan/milestones/${S}.pth
LV76=$R/long_videos_dev76_20261014
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
        D=$(TAGS="$TAG" LABEL="$LAB" EVAL_OUT=$E sbatch --parsable --dependency=afterany:$G scripts/eval149_hpc3_dev78.sbatch)
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
    D=$(TAGS="$TAG" LABEL="$LAB" EVAL_OUT=$E sbatch --parsable --dependency=afterany:$G scripts/eval149_hpc3_dev78.sbatch)
    D=${D%%;*}; case "$D" in ''|*[!0-9]*) D=pending_dev78;; esac
    echo "$G $F $D" > "$E/.submitted_$TAG"; echo "RETRIED dev78=$D"
fi
if [ ! -e "$E/.lv76_$TAG" ]; then
    [ "$NO_RAW" = 1 ] || [ -f "$MRAW" ] || { echo "RAW_MISSING $MRAW"; exit 0; }
    A=$(TAG=$TAG CKPT=$M CHUNK=19 sbatch --parsable --array=0-3%2 scripts/long_video_dev76.sbatch); A=${A%%;*}
    if [ "$NO_RAW" = 1 ]; then B=0
    else B=$(TAG=${TAG}raw CKPT=$MRAW CHUNK=19 sbatch --parsable --array=0-3%2 scripts/long_video_dev76.sbatch); B=${B%%;*}; fi
    case "$A$B" in ''|*[!0-9]*) echo "SUBMIT_FAILED Dev-76-L $TAG ($A / $B)"; exit 0;; esac
    echo "$A $B" > "$E/.lv76_$TAG"
    echo "DEV76_SUBMITTED $TAG ema=$A raw=$B"
fi
g=$(cat "$CF"/logs_eval149/pandwm_e149g_"${G}"_*.log 2>/dev/null | grep -c E149G_DONE)
if [ "$g" = 8 ] && [ ! -e "$E/.flicker_$TAG" ]; then
    touch "$E/.flicker_$TAG"
    cd "$C" || exit 1
    setsid nohup nice -n 10 "$PY" flicker_diag.py "$E" "$E/flicker_diag_$TAG.json" --pair "$PTAG" "$TAG" \
        $FL_ARMS > "$E/flicker_diag_$TAG.log" 2>&1 < /dev/null &
    echo "FLICKER_LAUNCHED $TAG"
    exit 0
fi
fst=$(sacct -j "$F" -n -X -o State 2>/dev/null | head -1 | tr -d ' ')
fl=0; [ -s "$E/flicker_diag_$TAG.json" ] && fl=1
ok=0; grep -q METRICS_DONE "$E/metrics/metrics.log" 2>/dev/null && grep -q "\"$TAG\"" "$E/metrics/report.json" && ok=1
d78=0; grep -q METRICS_DONE "$D78/metrics_$LAB.log" 2>/dev/null && [ -s "$D78/report_$LAB.json" ] && d78=1
n76=0; n76r=0
for i in $(seq 0 75); do
    [ -f "$LV76/${TAG}_scene${i}_seed42/gt_vs_generated.mp4" ] && n76=$((n76 + 1))
    [ -f "$LV76/${TAG}raw_scene${i}_seed42/gt_vs_generated.mp4" ] && n76r=$((n76r + 1))
done
[ "$NO_RAW" = 1 ] && n76r=76
echo "STATUS $TAG gen_done=$g/8 fin1=$fst metrics_ok=$ok dev78=$d78 flicker=$fl dev76=$n76/76 dev76raw=$n76r/76"
if [ "$n76" = 76 ] && [ "$n76r" = 76 ] && [ ! -e "$E/.l76an_$TAG" ]; then
    J=$(ARMS="$AN_ARMS" OUTNAME=dev76_analyze_$TAG sbatch --parsable scripts/long_dev76_analyze.sbatch)
    J=${J%%;*}; case "$J" in ''|*[!0-9]*) echo "SUBMIT_FAILED dev76 analysis"; exit 0;; esac
    echo "$J" > "$E/.l76an_$TAG"; echo "DEV76_ANALYSIS_SUBMITTED $TAG job=$J"
    exit 0
fi
an=0; [ -e "$E/.l76an_$TAG" ] && grep -q L76AN_END "$CF/logs_eval149/pandwm_l76an_$(cat "$E/.l76an_$TAG").log" 2>/dev/null \
    && [ -s "$LV76/dev76_analyze_$TAG.json" ] && an=1
if [ "$ok" = 1 ] && [ "$fl" = 1 ] && [ "$d78" = 1 ] && [ "$an" = 1 ]; then
    cp -p "$E/metrics/report.json" "$E/metrics/report_$TAG.json"
    "$PY" "$E/summary_v83.py" "$TAG" "$PTAG"
    "$PY" - "$D78" "$TAG" "$LAB" <<'PY'
import json, os, sys
d, tag, lab = sys.argv[1], sys.argv[2], sys.argv[3]
cur = json.load(open(f"{d}/report_{lab}.json"))["arms"][tag]
for ref_lab, ref_tag in (("h2000", "v83h2000"), ("i4250", "v84i4250"), ("i4500", "v84i4500")):
    if ref_tag == tag or not os.path.exists(f"{d}/report_{ref_lab}.json"):
        continue
    ref = json.load(open(f"{d}/report_{ref_lab}.json"))["arms"][ref_tag]
    fr, fc = ref["fvd_w1"] + ref["fvd_w2"], cur["fvd_w1"] + cur["fvd_w2"]
    print(f"Dev-78 FVD sum {tag} {fc:.1f} vs {ref_tag} {fr:.1f} ({fc - fr:+.1f}); FID {cur['fid_all']:.2f} vs {ref['fid_all']:.2f}")
PY
    grep -E "^  (FID|FVD) |late - early|minus |== 4|excess|w9-10|late w6-10" "$CF/logs_eval149/pandwm_l76an_$(cat "$E/.l76an_$TAG").log" | head -60
    grep -E "^V85 " "$LOG" | awk 'NR % 10 == 1' | tail -12
    "$PY" "$CF/eval_v7/v84_monitor_summary.py" "$LOG" --bins 4251-4350 4351-$S 2>&1 | tail -30
    echo "DONE $TAG"
fi
