#!/usr/bin/env bash
# 2026-10-21 D2 pilot evaluation chain v2 (report section 43.2), HPC3 login node; replaces d2_pilot_chain.sh after
# its S27 stage (A / B refined, one A task lost to a CUDA error on ACD1-59 and rerun as 745495).
#   cd $R && setsid nohup bash $D2/scripts/d2_pilot_chain_v2.sh > $R/logs/chain_pilot_1000_v2.log 2>&1 < /dev/null &
# Stages (markers in $M, so the chain can be restarted):
#   1. refine C (no CFG, start from the render noised to sigma 0.8), D (CFG 2 "all", sigma 0.8) and E (no CFG,
#      sigma 0.6) on S27 at +/-2 m; wait for A (incl. the rerun) and B to be complete
#   2. eval S27: in / A / B / C / D / E at 2 m, with coarse_psnr against "in"
#   3. pick: among samplers with coarse_psnr >= best - 1 dB, the lowest FID + FVD / 10
#   4. refine the pick on G69 at +/-2 and +/-1 m (videos for 10 clips); 5. eval G69 (in / d2 at 2 and 1 m)
set -uo pipefail
DR=/path/to/pandwm_root
D2=$DR/pandwm_code/PanDWM_d2_refiner_20261021
GSAA=$DR/pandwm_code/PanDWM-gs-aa-ft
R=$DR/pandwm_runs/d2_refiner_20261021
PY=$DR/pandwm_venv/bin/python
TAG=pilot; STEP=1000
CK=$R/runs/$TAG/checkpoints/$STEP.pth
INPUTS=$R/review_inputs/c2b_drop2
M=$R/chain_${TAG}_${STEP}; mkdir -p $M
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
submit_refine() {  # submit_refine <out tag> <clips> <shifts> <gmode> <gscale> <init sigma> <tasks> [video]
  (cd $D2 && CK=$CK OUT=$1 CLIPS=$2 SHIFTS=$3 STEPS=20 GMODE=$4 GSCALE=$5 INIT_SIGMA=$6 VIDEO=${8:-} \
    sbatch --parsable --array=0-$(( $7 - 1 )) --job-name=pandwm_d2ref_$1 scripts/d2_refine.sbatch)
}
P=${TAG}${STEP}
declare -A GM=( [A]=none [B]=all [C]=none [D]=all [E]=none )
declare -A GS=( [A]=1 [B]=2 [C]=1 [D]=2 [E]=1 )
declare -A IS=( [A]=0 [B]=0 [C]=0.8 [D]=0.8 [E]=0.6 )
log "CHAIN_V2_START ck=$CK"
if [[ ! -f $M/refined_S27_v2 ]]; then
  jc=$(submit_refine ${P}_C "$S27" "2,-2" none 1 0.8 2); jd=$(submit_refine ${P}_D "$S27" "2,-2" all 2 0.8 3)
  je=$(submit_refine ${P}_E "$S27" "2,-2" none 1 0.6 2)
  log "REFINE_S27_V2 C=$jc D=$jd E=$je"; wait_jobs $jc; wait_jobs $jd; wait_jobs $je
  until need_files $R/refined/${P}_A d2 "$S27" "2,-2,0" > $M/a_missing.txt; do log "waiting for A: $(cat $M/a_missing.txt)"; sleep 120; done
  for s in B C D E; do need_files $R/refined/${P}_$s d2 "$S27" "2,-2" || { log "REFINE_S27_${s}_INCOMPLETE"; exit 1; }; done
  touch $M/refined_S27_v2
fi
if [[ ! -f $M/eval_S27_v2.json ]]; then
  ARMS="--arm in=$INPUTS:render"; for s in A B C D E; do ARMS="$ARMS --arm $s=$R/refined/${P}_$s:latents"; done
  jx=$(cd $GSAA && MODE=eval ARGS="$ARMS --shifts 2 --clips $S27 --state $M/eval_S27_v2.pt --output $M/eval_S27_v2.json" \
       sbatch --parsable --job-name=pandwm_d2eval_S27v2 --time=03:00:00 d2_render_hpc3.sbatch)
  log "EVAL_S27_V2 $jx"; wait_jobs $jx
  [[ -f $M/eval_S27_v2.json ]] || { log "EVAL_S27_V2_FAILED"; exit 1; }
fi
CHOSEN=$($PY - $M/eval_S27_v2.json <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))["arms"]
names = [k for k in "ABCDE" if f"{k}_2" in r]
best = max(r[f"{k}_2"]["coarse_psnr"] for k in names)
ok = [k for k in names if r[f"{k}_2"]["coarse_psnr"] >= best - 1.0]
print(min(ok, key=lambda k: r[f"{k}_2"]["fid"] + r[f"{k}_2"]["fvd_w1"] / 10))
PY
)
log "S27 $($PY -c 'import json,sys; r=json.load(open(sys.argv[1]))["arms"]; print({k: (round(v["fid"],2), round(v["fvd_w1"],1), round(v.get("coarse_psnr", 0),2)) for k, v in r.items()})' $M/eval_S27_v2.json) CHOSEN=$CHOSEN"
echo $CHOSEN > $M/chosen_v2.txt
C=${P}_${CHOSEN}_G
if [[ ! -f $M/refined_G69 ]]; then
  jg=$(submit_refine $C "$G69" "2,-2,1,-1" ${GM[$CHOSEN]} ${GS[$CHOSEN]} ${IS[$CHOSEN]} 8 "$VIDEO")
  log "REFINE_G69 $C $jg"; wait_jobs $jg
  need_files $R/refined/$C d2 "$G69" "2,-2,1,-1" || { log "REFINE_G69_INCOMPLETE"; exit 1; }
  touch $M/refined_G69
fi
if [[ ! -f $M/eval_G69.json ]]; then
  jx=$(cd $GSAA && MODE=eval ARGS="--arm in=$INPUTS:render --arm d2=$R/refined/$C:latents --shifts 2,1 --clips $G69 --state $M/eval_G69.pt --output $M/eval_G69.json" \
       sbatch --parsable --job-name=pandwm_d2eval_G69 --time=04:00:00 d2_render_hpc3.sbatch)
  log "EVAL_G69 $jx"; wait_jobs $jx
  [[ -f $M/eval_G69.json ]] || { log "EVAL_G69_FAILED"; exit 1; }
fi
log "G69 $($PY -c 'import json,sys; r=json.load(open(sys.argv[1]))["arms"]; print({k: (round(v["fid"],2), round(v["fvd_w1"],1), round(v.get("coarse_psnr", 0),2)) for k, v in r.items()})' $M/eval_G69.json)"
log "CHAIN_DONE"
