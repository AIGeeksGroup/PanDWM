#!/usr/bin/env bash
# 2026-10-19: Val-150 features of a six-camera baseline run (baseline_feats.py), paper instrument.
#   RUNS=<run dir> NAME=<arm name> sbatch baseline_feats.sbatch
#SBATCH --job-name=pandwm_bfeat
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
PY=$DR/pandwm_venv/bin/python
export PYTHONPATH=$C:$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=$DR/.cache/torch OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
cd "$C" || exit 1
mkdir -p $DR/pandwm_runs/tests_20261018/feats_val150_baselines
echo "BFEAT_START $(date -Is) $(hostname) runs=$RUNS name=$NAME"
"$PY" baseline_feats.py --runs "$RUNS" --map val150_scene_map.json --name "$NAME" ${PARTIAL:+--allow-partial} \
  --out $DR/pandwm_runs/tests_20261018/feats_val150_baselines/feats_${NAME}.npz
echo "BFEAT_END rc=$? $(date -Is)"
