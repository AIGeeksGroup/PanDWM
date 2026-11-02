#!/usr/bin/env bash
# 2026-10-13: copy of long_video_hpc3.sbatch with HISTORY_MODE (default clean; e.g. anchor_sink) passed to
# run_long_rollout.py --history-mode.  Original header follows.
# 2026-10-11: long videos on HPC3 with the HPC1 protocol (~/hpc1_video_local.sh there: 8 dev scenes x 12 closed-loop
# windows, seed 42, clean history, student engine, run_long_rollout.py through lowmem_launch.py; same code tree and
# config as the HPC3 149-clip evaluation, PanDWM_eval149_20261011).  One GPU, the listed scenes one after another
# (a rollout peaks at ~150 GB host memory); finished scenes are skipped and a started scene resumes.  The 149-clip run
# showed a negligible HPC1 -> HPC3 shift (v8.1 step 1600: PSNR 21.42 vs 21.41, same frame change and sharpness).
#   TAG=v82g2000 CKPT=<..._ema.pth> SCENES="0 14 21 32" sbatch scripts/long_video_hpc3.sbatch
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
T=${TAG:?TAG}; CK=${CKPT:?CKPT}; SCENES=${SCENES:?SCENES}; echo "HISTORY_MODE=${HISTORY_MODE:-clean}"
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=12 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR PANDWM_HISTORY_GUIDANCE || true
declare -A TOK=([0]=2f56eb47c64f43df8902d9f88aa8a019 [14]=ed242d80ccb34b139aaf9ab89859332e [21]=9068766ee9374872a380fe75fcfb299e [32]=ca6e45c25d954dc4af6e12dd7b23454d [41]=380ff00ec86447e3b986edc8e82ffba7 [49]=85651af9c04945c3a394cf845cb480a6 [62]=5521cd85ed0e441f8d23938ed09099dd [82]=0dae482684ce4cd69a7258f55bc98d73)
[ -f "$CK" ] || { echo "CKPT_MISSING $CK"; exit 1; }
mkdir -p "$LV"
cd "$C"
echo "LV3_START $(date -Is) node=$(hostname) tag=$T scenes=$SCENES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
rc=0
for SC in $SCENES; do
  OUT=$LV/${T}_scene${SC}_seed42
  if [[ -f $OUT/gt_vs_generated.mp4 ]]; then echo "SCENE_SKIP $T $SC"; continue; fi
  RES=(); [[ -f $OUT/resume_state.pt ]] && RES=(--resume)
  export MASTER_PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  "$PY" lowmem_launch.py run_long_rollout.py --base-config "$CFG" --checkpoint "$CK" --output "$OUT" \
    --scene-token ${TOK[$SC]} --seed 42 --windows 12 --history-mode ${HISTORY_MODE:-clean} --engine student "${RES[@]}"
  echo "SCENE_DONE $T $SC rc=$? $(date +%T)"
  [[ -f $OUT/gt_vs_generated.mp4 ]] || rc=1
done
echo "LV3_END $T rc=$rc $(date -Is)"
exit $rc
