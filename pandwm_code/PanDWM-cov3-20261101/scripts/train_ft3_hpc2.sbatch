#!/usr/bin/env bash
#SBATCH --job-name=ft3_train
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs_ft3_20261030/logs/%x_%j.log
# 2026-10-30 round-3 fine-tune (GS_FT3_PLAN_20261030.md): ARM=G0|G1|G2 STEPS (default 2000) steps from R1 step 2000
# into gs_ft3_20261030/<TAG or ARM>, checkpoints every 500 (all kept).  Code: the isolated PanDWM-ft3-20261030 copy
# first on PYTHONPATH (pipeline_relabel.py and the pipeline_depth hook live only there).
# Smoke: ARM=G1 STEPS=20 CKPT_STEPS=20 LOG_STEPS=1 TAG=smoke_G1 sbatch --time=01:00:00 train_ft3_hpc2.sbatch
set -uo pipefail
ARM=${ARM:?ARM=G0|G1|G2}; STEPS=${STEPS:-2000}
source $HOME/pandwm_code/PanDWM-gs4d-20261002/scripts/common_env.sh
F=$HOME/pandwm_code/PanDWM-ft3-20261030
export PYTHONPATH=$F/gs4d_src:$F/gsaa_src:$HOME/pandwm_code/TATS_fvd:$HOME/pandwm_da3_pkgs/site
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXT=/tmp/pandwm_ext_${SLURM_JOB_ID}; rm -rf $EXT; mkdir -p $EXT
cp -a $HOME/.cache/pandwm_ext_golden_20261009/gsplat_cuda $EXT/
trap 'rm -rf $EXT' EXIT
export TORCH_EXTENSIONS_DIR=$EXT
CFG=$F/configs/gs4dd_hpc2_ft3_${ARM}.json
OUT=$HOME/pandwm_runs/gs_ft3_20261030/${TAG:-$ARM}
echo "FT3_START arm=$ARM tag=${TAG:-$ARM} steps=$STEPS node=$(hostname) $(date -Is)"
python -c "import dwm_pano_4d.pipeline_relabel as m, dwm_pano.gs_scene_v3 as s; print('code', m.__file__, s.__file__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python $F/gsaa_src/dwm_pano/train.py -c $CFG -o $OUT --max-steps $STEPS \
  --log-steps ${LOG_STEPS:-20} --preview-steps 100000000 --evaluation-steps 100000000 \
  --checkpointing-steps ${CKPT_STEPS:-500} --keep-last-ckpts 20
rc=$?
echo "FT3_END rc=$rc $(date -Is)"
exit $rc
