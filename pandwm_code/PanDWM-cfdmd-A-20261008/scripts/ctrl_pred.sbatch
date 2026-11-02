#!/usr/bin/env bash
# 2026-10-21 controllability predictions (ctrl_predict.py; protocol CONTROLLABILITY_PLAN_20261021.md, frozen).
#   ARMS="rec448=views:$B/rec448_val150 recband=recband" [REFS="recband=$FV/real.npz"] [MODELS=frcnn,m2f_cs]
#   [UNMASKED=rec448u] [GT=1: also build the GT (ctrl_gt.py, CPU) in the background] sbatch scripts/ctrl_pred.sbatch
# Outputs: $OUT/pred/<arm>__<det>+<seg>/clipNNN.npz, $OUT/gt/<timing>/clipNNN.npz (resumable).
#SBATCH --job-name=pandwm_ctrl
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-28,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
PY=$DR/pandwm_venv/bin/python
OUT=$DR/pandwm_runs/ctrl_20261021
mkdir -p $OUT/logs $OUT/pred $OUT/gt
export PYTHONPATH=$C:$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=$DR/.cache/torch OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$C" || exit 1
echo "CTRL_JOB_START $(date -Is) $(hostname) arms=$ARMS models=${MODELS:-frcnn,m2f_cs} gt=${GT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GTPID=
if [[ ${GT:-0} == 1 ]]; then
  "$PY" ctrl_gt.py --base-config cd_eval_frozen149_hpc3.json --out-dir $OUT/gt > $OUT/logs/gt_${SLURM_JOB_ID}.log 2>&1 &
  GTPID=$!
fi
args=(); for x in $ARMS; do args+=(--arm "$x"); done
for r in ${REFS:-}; do args+=(--source-ref "$r"); done
for u in ${UNMASKED:-}; do args+=(--unmasked "$u"); done
"$PY" ctrl_predict.py --base-config cd_eval_frozen149_hpc3.json --out-dir $OUT/pred --models ${MODELS:-frcnn,m2f_cs} \
  "${args[@]}" ${POSITIONS:+--positions $POSITIONS}
rc=$?
if [[ -n $GTPID ]]; then wait $GTPID; echo "GT rc=$? ($(tail -1 $OUT/logs/gt_${SLURM_JOB_ID}.log))"; fi
echo "CTRL_JOB_END rc=$rc $(date -Is)"
exit $rc
