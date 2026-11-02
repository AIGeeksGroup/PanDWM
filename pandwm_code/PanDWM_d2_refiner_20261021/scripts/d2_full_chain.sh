#!/usr/bin/env bash
# 2026-10-21 D2 continuation chain (report section 43.3; user 21:40 "改进后续训"), HPC3 login node.
#   cd $R && setsid nohup bash $D2/scripts/d2_full_chain.sh > $R/logs/chain_full.log 2>&1 < /dev/null &
# The pilot run (runs/pilot, step 1000) is resumed in place with configs/d2_full_hpc3_8gpu.json (render-init
# samples), 1000 steps per segment.  After each segment (2000, 3000, 4000): refine S27 at +/-2 m with B (pure noise,
# CFG 2 "all") and D (render noised to sigma 0.8, CFG 2), then eval in / B / D / Bf / Df (f = low frequencies from
# the GS render, d2_eval_refined.py ":fuse").  Training and refinement alternate on the same 8 GPUs (8 + 8 split
# with abl_A).  At the end the (checkpoint, arm) with coarse_psnr >= best - 1 dB and the lowest FID + FVD / 10 over
# all evaluated ones (incl. the pilot's step 1000) is refined on G69 at +/-2 and +/-1 m and evaluated.
# Markers in $M make every stage idempotent; the chain can be restarted.
# 2026-10-22 16:40 (user: "挂上去吧，等他们跑完就自动跑"): before any new GPU job the chain waits until no
# pandwm_abl* job is left on HPC3 (jobs pending with DependencyNeverSatisfied do not count); a segment whose job
# was submitted and held earlier ($M/held_job_<step>) is released instead of being submitted again.
set -uo pipefail
DR=/path/to/pandwm_root
D2=$DR/pandwm_code/PanDWM_d2_refiner_20261021
GSAA=$DR/pandwm_code/PanDWM-gs-aa-ft
R=$DR/pandwm_runs/d2_refiner_20261021
PY=$DR/pandwm_venv/bin/python
RUN=$R/runs/pilot
INPUTS=$R/review_inputs/c2b_drop2
M=$R/chain_full; mkdir -p $M
S27=$($PY -c "s=[c for c in range(149) if (c//20)%2==0]; print(','.join(map(str,s[::3])))")
G69=$($PY -c "print(','.join(str(c) for c in range(149) if (c//20)%2==1))")
VIDEO=$($PY -c "g=[c for c in range(149) if (c//20)%2==1]; print(','.join(map(str,g[::7][:10])))")
log() { echo "$(date -Is) $*"; }
wait_jobs() { while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done; }
need_files() {
  $PY - "$@" <<'PY'
import sys, pathlib
root, prefix, clips, shifts = sys.argv[1:5]
missing = [f"{c}:{s}" for c in clips.split(",") for s in shifts.split(",")
           if not (pathlib.Path(root) / f"clip{int(c):03d}" / f"{prefix}_dy{float(s):g}.pt").is_file()]
print(len(missing), " ".join(missing[:8]))
sys.exit(1 if missing else 0)
PY
}
submit_refine() {  # <ck> <out tag> <clips> <shifts> <gmode> <gscale> <init sigma> <tasks> [video]
  (cd $D2 && CK=$1 OUT=$2 CLIPS=$3 SHIFTS=$4 STEPS=20 GMODE=$5 GSCALE=$6 INIT_SIGMA=$7 VIDEO=${9:-} \
    sbatch --parsable --array=0-$(( $8 - 1 )) --job-name=pandwm_d2ref_$2 scripts/d2_refine.sbatch)
}
refine_until_complete() {  # same args as submit_refine; resubmits once if outputs are missing (bad node)
  local j; j=$(submit_refine "$@"); log "REFINE $2 $j"; wait_jobs $j
  if ! need_files $R/refined/$2 d2 "$3" "$4" > /dev/null; then
    j=$(submit_refine "$1" "$2" "$3" "$4" "$5" "$6" "$7" 1 "${9:-}"); log "REFINE_RETRY $2 $j"; wait_jobs $j
  fi
  need_files $R/refined/$2 d2 "$3" "$4" || { log "REFINE_INCOMPLETE $2"; exit 1; }
}
wait_for_ablations() {
  local n
  while true; do
    n=$(squeue -u $USER -h -o "%j %T %r" | awk '$1 ~ /^pandwm_abl/ && $3 != "DependencyNeverSatisfied" {c++} END {print c+0}')
    [[ $n -eq 0 ]] && break
    log "waiting for $n ablation jobs"; sleep 600
  done
  log "ABLATIONS_DONE"
}
log "CHAIN_FULL_START"
[[ ${SKIP_ABL_WAIT:-0} == 1 ]] || wait_for_ablations
for STEP in 2000 3000 4000; do
  PREV=$(( STEP - 1000 )); CK=$RUN/checkpoints/$STEP.pth
  if [[ ! -f $M/trained_$STEP ]]; then
    if [[ ! -f $CK ]]; then
      if [[ -f $M/held_job_$STEP ]] && squeue -h -j "$(cat $M/held_job_$STEP)" 2>/dev/null | grep -q .; then
        jt=$(cat $M/held_job_$STEP); scontrol release $jt; log "RELEASED $jt for $PREV->$STEP"
      else
        jt=$(cd $D2 && TAG=pilot RESUME_FROM=$PREV STEPS=$STEP CKPT_STEPS=1000 CONFIG_FILE=$D2/configs/d2_full_hpc3_8gpu.json \
             sbatch --parsable --job-name=pandwm_d2_cont scripts/train_d2.sbatch)
        log "TRAIN $PREV->$STEP job $jt"
      fi
      wait_jobs $jt
      grep -q "D2_TRAIN_END rc=0" $R/logs/pandwm_d2_cont_$jt.log || { log "TRAIN_FAILED $STEP $(grep D2_TRAIN_END $R/logs/pandwm_d2_cont_$jt.log)"; exit 1; }
      grep -E "^Step [0-9]+ " $R/logs/pandwm_d2_cont_$jt.log | tail -1
    fi
    [[ -f $CK ]] || { log "NO_CHECKPOINT $CK"; exit 1; }
    touch $M/trained_$STEP
  fi
  if [[ ! -f $M/refined_$STEP ]]; then
    jb=$(submit_refine $CK full${STEP}_B "$S27" "2,-2" all 2 0 4); jd=$(submit_refine $CK full${STEP}_D "$S27" "2,-2" all 2 0.8 4)
    log "REFINE_S27 $STEP B=$jb D=$jd"; wait_jobs $jb; wait_jobs $jd
    for s in B D; do
      if ! need_files $R/refined/full${STEP}_$s d2 "$S27" "2,-2" > /dev/null; then
        if [[ $s == B ]]; then refine_until_complete $CK full${STEP}_B "$S27" "2,-2" all 2 0 1; else refine_until_complete $CK full${STEP}_D "$S27" "2,-2" all 2 0.8 1; fi
      fi
    done
    touch $M/refined_$STEP
  fi
  if [[ ! -f $M/eval_S27_$STEP.json ]]; then
    ARMS="--arm in=$INPUTS:render --arm B=$R/refined/full${STEP}_B:latents --arm D=$R/refined/full${STEP}_D:latents"
    ARMS="$ARMS --arm Bf=$R/refined/full${STEP}_B:latents:fuse --arm Df=$R/refined/full${STEP}_D:latents:fuse"
    je=$(cd $GSAA && MODE=eval ARGS="$ARMS --shifts 2 --clips $S27 --state $M/eval_S27_$STEP.pt --output $M/eval_S27_$STEP.json" \
         sbatch --parsable --job-name=pandwm_d2eval_S27_$STEP --time=03:00:00 d2_render_hpc3.sbatch)
    log "EVAL_S27 $STEP $je"; wait_jobs $je
    [[ -f $M/eval_S27_$STEP.json ]] || { log "EVAL_S27_FAILED $STEP"; exit 1; }
  fi
  log "S27 $STEP $($PY -c 'import json,sys; r=json.load(open(sys.argv[1]))["arms"]; print({k: (round(v["fid"],2), round(v["fvd_w1"],1), round(v.get("coarse_psnr", 0),2)) for k, v in r.items()})' $M/eval_S27_$STEP.json)"
done
PICK=$($PY - $M $R <<'PY'
import json, sys, pathlib
m, r = pathlib.Path(sys.argv[1]), sys.argv[2]
cands = {}
for step, path in [(1000, pathlib.Path(r) / "chain_pilot_1000" / "eval_S27_fuse.json")] + \
        [(s, m / f"eval_S27_{s}.json") for s in (2000, 3000, 4000)]:
    if not path.is_file():
        continue
    for arm, v in json.load(open(path))["arms"].items():
        name = arm.rsplit("_", 1)[0]
        if name == "in" or "coarse_psnr" not in v:
            continue
        cands[(step, name)] = v
best = max(v["coarse_psnr"] for v in cands.values())
ok = {k: v for k, v in cands.items() if v["coarse_psnr"] >= best - 1.0}
step, name = min(ok, key=lambda k: ok[k]["fid"] + ok[k]["fvd_w1"] / 10)
print(step, name)
PY
)
read PSTEP PARM <<< "$PICK"; log "PICK step=$PSTEP arm=$PARM"
echo "$PSTEP $PARM" > $M/pick.txt
BASE=${PARM%f}; FUSE=""; [[ $PARM == *f ]] && FUSE=":fuse"
if [[ $BASE == B ]]; then INIT=0; else INIT=0.8; fi
PCK=$RUN/checkpoints/$PSTEP.pth; [[ $PSTEP == 1000 ]] && PCK=$RUN/checkpoints/1000.pth
OUTG=g69_${PSTEP}_${BASE}
if [[ ! -f $M/refined_G69 ]]; then
  refine_until_complete $PCK $OUTG "$G69" "2,-2,1,-1" all 2 $INIT 8 "$VIDEO"
  touch $M/refined_G69
fi
if [[ ! -f $M/eval_G69.json ]]; then
  je=$(cd $GSAA && MODE=eval ARGS="--arm in=$INPUTS:render --arm d2=$R/refined/$OUTG:latents$FUSE --shifts 2,1 --clips $G69 --state $M/eval_G69.pt --output $M/eval_G69.json" \
       sbatch --parsable --job-name=pandwm_d2eval_G69 --time=04:00:00 d2_render_hpc3.sbatch)
  log "EVAL_G69 $je"; wait_jobs $je
  [[ -f $M/eval_G69.json ]] || { log "EVAL_G69_FAILED"; exit 1; }
fi
log "G69 $($PY -c 'import json,sys; r=json.load(open(sys.argv[1]))["arms"]; print({k: (round(v["fid"],2), round(v["fvd_w1"],1), round(v.get("coarse_psnr", 0),2)) for k, v in r.items()})' $M/eval_G69.json)"
log "CHAIN_DONE"
