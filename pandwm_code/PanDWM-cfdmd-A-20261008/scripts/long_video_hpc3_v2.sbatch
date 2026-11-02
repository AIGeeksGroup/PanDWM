#!/usr/bin/env bash
# 2026-10-13: long_video_hpc3.sbatch with a SEED env (default 42) and the 12-scene non-test long-video dev set:
# the 8 dev scenes of split_195_scene_manifest.json plus the 4 heldout scenes excluded from Test-S / Test-L because they
# were run as paper-page demos (heldout_demo_exclusions_20261010.json: 10 / 31 / 52 / 72).  These are the only val
# scenes outside the 71 test scenes with a valid 195-frame (12-window) sequence.  Output <tag>_scene<N>_seed<SEED>;
# everything else is the HPC1 protocol of long_video_hpc3.sbatch (12 closed-loop windows, clean history, student).
#   TAG=v83h2000 CKPT=<..._ema.pth> SCENES="10 31 52 72" SEED=43 sbatch scripts/long_video_hpc3_v2.sbatch
#SBATCH --job-name=pandwm_lv3
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=05:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
CFG=$C/cd_eval_frozen149_hpc3.json
LV=${LV_OUT:-$DR/pandwm_runs/long_videos_hpc3_20261011}
PY=$DR/pandwm_venv/bin/python
T=${TAG:?TAG}; CK=${CKPT:?CKPT}; SCENES=${SCENES:?SCENES}; SEED=${SEED:-42}
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=12 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR PANDWM_HISTORY_GUIDANCE || true
declare -A TOK=([0]=2f56eb47c64f43df8902d9f88aa8a019 [14]=ed242d80ccb34b139aaf9ab89859332e [21]=9068766ee9374872a380fe75fcfb299e [32]=ca6e45c25d954dc4af6e12dd7b23454d [41]=380ff00ec86447e3b986edc8e82ffba7 [49]=85651af9c04945c3a394cf845cb480a6 [62]=5521cd85ed0e441f8d23938ed09099dd [82]=0dae482684ce4cd69a7258f55bc98d73 [10]=3363f396bb43405fbdd17d65dc123d4e [31]=4bb9626f13184da7a67ba1b241c19558 [52]=3a2d9bf6115f40898005d1c1df2b7282 [72]=7052d21b95fc4bae8761b8d9524f3e42)
[ -f "$CK" ] || { echo "CKPT_MISSING $CK"; exit 1; }
mkdir -p "$LV"
cd "$C"
echo "LV3_START $(date -Is) node=$(hostname) tag=$T scenes=$SCENES seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
rc=0
for SC in $SCENES; do
  OUT=$LV/${T}_scene${SC}_seed${SEED}
  if [[ -f $OUT/gt_vs_generated.mp4 ]]; then echo "SCENE_SKIP $T $SC"; continue; fi
  RES=(); [[ -f $OUT/resume_state.pt ]] && RES=(--resume)
  export MASTER_PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  "$PY" lowmem_launch.py run_long_rollout.py --base-config "$CFG" --checkpoint "$CK" --output "$OUT" \
    --scene-token ${TOK[$SC]} --seed $SEED --windows 12 --history-mode clean --engine student "${RES[@]}"
  echo "SCENE_DONE $T $SC rc=$? $(date +%T)"
  [[ -f $OUT/gt_vs_generated.mp4 ]] || rc=1
done
echo "LV3_END $T rc=$rc $(date -Is)"
exit $rc
