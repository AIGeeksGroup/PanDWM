#!/usr/bin/env bash
# 2026-10-18 (report section 38): login-node watcher for the formal F1 run.  For each of steps 1000..5000, once
# $RUNS/<TAG>/checkpoints/<step>.pth exists and has not changed for 3 minutes, submit
#   eval_depth_pilot.sbatch  (25 val clips, evaluate_4d)        -> $RUNS/eval/F1_<step>.json
#   diag_depth_metric.sbatch (holdout5 depth error, 25 clips)   -> $RUNS/label_depth/diag_depth_metric_F1_<step>.json
# Markers in $RUNS/<TAG>/eval_submitted/ prevent resubmission.  Exits after 5000 or when the training job is gone and
# no checkpoint is pending.  Usage: TAG=formal_F1 TRAIN_JOB=<id> setsid nohup bash formal_eval_watch.sh &
set -u
export PATH=$PATH:/opt/slurm/bin
R=/path/to/pandwm_root/pandwm_runs/gs4d_hpc3_20261017
G=/path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002
TAG=${TAG:-formal_F1}
M=$R/$TAG/eval_submitted; mkdir -p $M
LOG=$R/formal_eval_watch_$TAG.log
echo "$(date -Is) watch start tag=$TAG train_job=${TRAIN_JOB:-?}" >> $LOG
while true; do
  for s in 1000 2000 3000 4000 5000; do
    c=$R/$TAG/checkpoints/$s.pth
    [ -f $M/$s ] && continue
    [ -f $c ] || continue
    age=$(( $(date +%s) - $(stat -c %Y $c) ))
    [ $age -ge 180 ] || continue
    cd $G
    e=$(NAME=F1_$s CHECKPOINT=$c sbatch --parsable scripts_hpc3/eval_depth_pilot.sbatch)
    d=$(TAG=F1_$s CLIPS=25 PKL=$R/data/nuscenes_interp_12Hz_infos_train_holdout5.pkl CKPTS="$c" \
        sbatch --parsable scripts_hpc3/diag_depth_metric.sbatch)
    echo "$(date -Is) step $s eval=$e diag=$d" | tee -a $LOG > $M/$s
    echo "F1 step $s evals: eval4d=$e holdout_diag=$d" >> $R/jobs.txt
  done
  [ -f $M/5000 ] && { echo "$(date -Is) all submitted" >> $LOG; exit 0; }
  if [ -n "${TRAIN_JOB:-}" ] && ! squeue -h -j $TRAIN_JOB 2>/dev/null | grep -q .; then
    pending=0
    for s in 1000 2000 3000 4000 5000; do [ -f $R/$TAG/checkpoints/$s.pth ] && [ ! -f $M/$s ] && pending=1; done
    [ $pending -eq 0 ] && { echo "$(date -Is) training job gone, nothing pending" >> $LOG; exit 0; }
  fi
  sleep 300
done
