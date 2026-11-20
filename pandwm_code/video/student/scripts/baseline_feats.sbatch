#!/usr/bin/env bash
# Features of a six-camera baseline run on the Val-150 scenes (baseline_feats.py).
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
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/eval
PY=$DR/pandwm_venv/bin/python
export PYTHONPATH=$C:$DR/pandwm_code/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=$DR/.cache/torch OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
cd "$C" || exit 1
mkdir -p $DR/pandwm_runs/tests/feats_val150_baselines
echo "BFEAT_START $(date -Is) $(hostname) runs=$RUNS name=$NAME"
"$PY" baseline_feats.py --runs "$RUNS" --map val150_scene_map.json --name "$NAME" ${PARTIAL:+--allow-partial} \
  --out $DR/pandwm_runs/tests/feats_val150_baselines/feats_${NAME}.npz
echo "BFEAT_END rc=$? $(date -Is)"
