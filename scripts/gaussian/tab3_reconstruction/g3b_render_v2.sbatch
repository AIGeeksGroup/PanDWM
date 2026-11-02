#!/usr/bin/env bash
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs_recon_g3b_20261101/logs/%x_%A_%a.log
# 2026-11-01 tab:gs_recon with the FT3 G3b decoder (PanDWM-d2-refiner-20261021/CONSIST_PLAN_20261101.md section 5).
# Settings of gs_recon_ext249_20261024/scripts/ext_ours_render.sbatch (config g2.json, pkl, DA3 weights, render_ours.py
# modes 3d / 4d, the three resolutions) with the PanDWM-ft3-20261030 code and checkpoint G3b 2000 (md5 18e655d8...):
#   3d: scene overrides {"plane_offset_px": 1.0}            (single frame, per-frame DA3 as before)
#   4d: scene overrides {"da3_window": "clip", "da3_fuse": true, "plane_offset_px": 1.0}   (G3b's V2 scene, frames 0..20)
# Task k of N renders positions k::N of the 213-clip targets, then of the 36-clip targets, for both modes.
#   [TAG=g3b CODE=PanDWM-ft3-20261030 OVR3=<json> OVR4=<json>] sbatch --array=0-2 g3b_render.sbatch
#   11-01: TAG=g3bs CODE=PanDWM-cov-20261101 with seam_bridge overrides (CONSIST_PLAN_20261101.md 5.6)
set -uo pipefail
source $HOME/pandwm_env.sh
H=$HOME; X2=$H/pandwm_runs/gs_recon_g3b_20261101
F=$H/pandwm_code/${CODE:-PanDWM-ft3-20261030}
CFG=$H/pandwm_runs/long_horizon_20260928/gs_long_demo_20261001/configs/g2.json
CK=${CKPT:-$H/pandwm_runs/gs_ft3_20261030/G3b/checkpoints/2000.pth}   # 11-01: CKPT override (FT4)
PKL=$H/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl
DA3H=$H/pandwm_code/third_party/depth_anything_3
T213=$H/pandwm_runs/gs_recon_20261022/targets.json
T36=$H/pandwm_runs/gs_recon_ext249_20261024/inputs/targets_ext36.json
EXT=/tmp/pandwm_g3b_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}; rm -rf $EXT; mkdir -p $EXT
cp -a $H/.cache/pandwm_ext_golden_20261009/gsplat_cuda $EXT/
trap 'rm -rf $EXT' EXIT
export PYTHONPATH=$F/gsaa_src:$F/gs4d_src:$H/pandwm_code/TATS_fvd:$H/pandwm_da3_pkgs/site TORCH_EXTENSIONS_DIR=$EXT
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID} * 7 + ${SLURM_ARRAY_TASK_ID:-0} * 131 + RANDOM) % 20000 ))
cd $F/gsaa_src
md5sum dwm_pano/recon/render_ours.py dwm_pano/gs_scene_v3.py $CK
nvidia-smi --query-gpu=name --format=csv,noheader
N=${SLURM_ARRAY_TASK_COUNT:-1}; K=${SLURM_ARRAY_TASK_ID:-0}
OVR3=${OVR3:-'{"plane_offset_px": 1.0}'}
OVR4=${OVR4:-'{"da3_window": "clip", "da3_fuse": true, "plane_offset_px": 1.0}'}
for T in $T213 $T36; do
  S=$(python -c "import json; n=json.load(open('$T'))['count']; print(','.join(map(str, range(n)[$K::$N])))")
  for M in 3d 4d; do
    [ $M = 3d ] && OVR=$OVR3 || OVR=$OVR4
    echo "G3B_RENDER_START task=$K/$N targets=$T mode=$M ovr=$OVR $(date -Is)"
    python -m dwm_pano.recon.render_ours -c $CFG --data-pkl $PKL --checkpoint $CK --targets $T \
      --out-root $X2/renders/${TAG:-g3b} --resolutions 224x400,352x640,280x518 --modes $M --samples "$S" \
      --da3-src $DA3H/repo/src --da3-anyview $DA3H/weights/DA3-LARGE-1.1 --da3-metric $DA3H/weights/DA3METRIC-LARGE \
      --scene-overrides "$OVR" || { echo "G3B_RENDER_FAIL $T $M"; exit 1; }
  done
done
echo "G3B_RENDER_END $(date -Is)"
