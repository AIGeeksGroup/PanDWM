#!/usr/bin/env bash
# 2026-10-21 D2 pilot evaluation chain (D2_PLAN_20261021.md Task 10; report section 43), HPC3 login node.
#   cd $R && setsid nohup bash $D2/scripts/d2_pilot_chain.sh > $R/logs/chain_pilot_1000.log 2>&1 < /dev/null &
# Idempotent: every stage leaves a marker in $M and is skipped when the marker exists, so the chain can be restarted.
#   1. wait for the training job's D2_TRAIN_END (rc=0) and $CK
#   2. wait for the review inputs of all clips used below
#   3. refine sampler A (20 steps, no CFG) and B (20 steps, CFG 2 on render + text) on S27 (every 3rd clip of the
#      selection half), shifts +/-2 (A also 0 m for the fidelity check)
#   4. eval S27: in_2 / A_2 / B_2  ->  pick the sampler with the lower FID + FVD / 10
#   5. refine the chosen sampler on G69 (the whole gate half), shifts +/-2 and +/-1, videos for 10 clips
#   6. eval G69: in_2, in_1, <chosen>_2, <chosen>_1   ->  $M/eval_G69.json, then CHAIN_DONE
set -uo pipefail
DR=/path/to/pandwm_root
D2=$DR/pandwm_code/PanDWM_d2_refiner_20261021
GSAA=$DR/pandwm_code/PanDWM-gs-aa-ft
R=$DR/pandwm_runs/d2_refiner_20261021
PY=$DR/pandwm_venv/bin/python
TAG=${RUN_TAG:-pilot}; STEP=${STEP:-1000}
RUN=$R/runs/$TAG; CK=$RUN/checkpoints/$STEP.pth
TRAIN_LOG=${TRAIN_LOG:?TRAIN_LOG=<training job log>}
INPUTS=$R/review_inputs/c2b_drop2
M=$R/chain_${TAG}_${STEP}; mkdir -p $M
S27=$($PY -c "s=[c for c in range(149) if (c//20)%2==0]; print(','.join(map(str,s[::3])))")
G69=$($PY -c "print(','.join(str(c) for c in range(149) if (c//20)%2==1))")
VIDEO=$($PY -c "g=[c for c in range(149) if (c//20)%2==1]; print(','.join(map(str,g[::7][:10])))")
log() { echo "$(date -Is) $*"; }
wait_jobs() { while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done; }
need_files() {   # need_files <root> <prefix> <clips csv> <shifts csv>: all files present?
  $PY - "$@" <<'PY'
import sys, pathlib
root, prefix, clips, shifts = sys.argv[1:5]
missing = [f"{c}:{s}" for c in clips.split(",") for s in shifts.split(",")
           if not (pathlib.Path(root) / f"clip{int(c):03d}" / f"{prefix}_dy{float(s):g}.pt").is_file()]
print(len(missing), " ".join(missing[:8]))
sys.exit(1 if missing else 0)
PY
}
log "CHAIN_START tag=$TAG step=$STEP ck=$CK S27=$S27"
log "G69=$G69 video=$VIDEO"

if [[ ! -f $M/trained ]]; then
  until grep -q "D2_TRAIN_END" "$TRAIN_LOG" 2>/dev/null; do sleep 120; done
  grep -q "D2_TRAIN_END rc=0" "$TRAIN_LOG" || { log "TRAIN_FAILED $(grep D2_TRAIN_END $TRAIN_LOG)"; exit 1; }
  [[ -f $CK ]] || { log "NO_CHECKPOINT $CK"; exit 1; }
  touch $M/trained; log "TRAINED $(ls -la $CK)"
fi

until need_files $INPUTS render "$S27,$G69" "2,-2,1,-1,0" > $M/inputs_missing.txt; do
  log "waiting for review inputs: $(cat $M/inputs_missing.txt)"; sleep 300
done

submit_refine() {  # submit_refine <out tag> <clips> <shifts> <gmode> <gscale> <tasks> [video]
  (cd $D2 && CK=$CK OUT=$1 CLIPS=$2 SHIFTS=$3 STEPS=20 GMODE=$4 GSCALE=$5 VIDEO=${7:-} \
    sbatch --parsable --array=0-$(( $6 - 1 )) --job-name=pandwm_d2ref_$1 scripts/d2_refine.sbatch)
}
A=${TAG}${STEP}_A; B=${TAG}${STEP}_B
if [[ ! -f $M/refined_S27 ]]; then
  ja=$(submit_refine $A "$S27" "2,-2,0" none 1 4); jb=$(submit_refine $B "$S27" "2,-2" all 2 4)
  log "REFINE_S27 A=$ja B=$jb"; wait_jobs $ja; wait_jobs $jb
  need_files $R/refined/$A d2 "$S27" "2,-2,0" && need_files $R/refined/$B d2 "$S27" "2,-2" || { log "REFINE_S27_INCOMPLETE"; exit 1; }
  touch $M/refined_S27
fi

if [[ ! -f $M/eval_S27.json ]]; then
  je=$(cd $GSAA && MODE=eval ARGS="--arm in=$INPUTS:render --arm A=$R/refined/$A:latents --arm B=$R/refined/$B:latents --shifts 2 --clips $S27 --state $M/eval_S27.pt --output $M/eval_S27.json" \
       sbatch --parsable --job-name=pandwm_d2eval_S27 --time=02:00:00 d2_render_hpc3.sbatch)
  log "EVAL_S27 $je"; wait_jobs $je
  [[ -f $M/eval_S27.json ]] || { log "EVAL_S27_FAILED"; exit 1; }
fi
CHOSEN=$($PY - $M/eval_S27.json <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))["arms"]
score = {k: r[f"{k}_2"]["fid"] + r[f"{k}_2"]["fvd_w1"] / 10 for k in ("A", "B")}
print(min(score, key=score.get))
PY
)
log "S27 $(cat $M/eval_S27.json | $PY -c 'import json,sys; r=json.load(sys.stdin)["arms"]; print({k: (round(v["fid"],2), round(v["fvd_w1"],1)) for k, v in r.items()})') CHOSEN=$CHOSEN"
echo $CHOSEN > $M/chosen.txt
if [[ $CHOSEN == A ]]; then GM=none; GS=1; else GM=all; GS=2; fi
C=${TAG}${STEP}_${CHOSEN}_G
if [[ ! -f $M/refined_G69 ]]; then
  jg=$(submit_refine $C "$G69" "2,-2,1,-1" $GM $GS 8 "$VIDEO")
  log "REFINE_G69 $jg"; wait_jobs $jg
  need_files $R/refined/$C d2 "$G69" "2,-2,1,-1" || { log "REFINE_G69_INCOMPLETE"; exit 1; }
  touch $M/refined_G69
fi
if [[ ! -f $M/eval_G69.json ]]; then
  je=$(cd $GSAA && MODE=eval ARGS="--arm in=$INPUTS:render --arm d2=$R/refined/$C:latents --shifts 2,1 --clips $G69 --state $M/eval_G69.pt --output $M/eval_G69.json" \
       sbatch --parsable --job-name=pandwm_d2eval_G69 --time=03:00:00 d2_render_hpc3.sbatch)
  log "EVAL_G69 $je"; wait_jobs $je
  [[ -f $M/eval_G69.json ]] || { log "EVAL_G69_FAILED"; exit 1; }
fi
log "G69 $(cat $M/eval_G69.json | $PY -c 'import json,sys; r=json.load(sys.stdin)["arms"]; print({k: (round(v["fid"],2), round(v["fvd_w1"],1)) for k, v in r.items()})')"
log "CHAIN_DONE"
