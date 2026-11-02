#!/usr/bin/env bash
#SBATCH --job-name=ft3_dev
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/gs_dev_hpc3/logs/%x_%A_%a.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58,ACD1-59
# 2026-10-30 round-3 (GS_FT3_PLAN_20261030.md): dev_eval_hpc3.sbatch with the PanDWM-ft3-20261030 code copy first on
# PYTHONPATH.  Copied from the 2026-10-24 round-2 GS-Dev eval (GS_FT2_PLAN_20261024.md section 3), tab:gs_recon
# protocol at 224x400.
#   ARM=<name> [CKPT=<decoder .pth>] [OVR='<SceneOptions JSON>'] MODE=render sbatch --array=0-3 dev_eval_hpc3.sbatch
#   ARM=<name> MODE=score sbatch dev_eval_hpc3.sbatch        (scores 3d and 4d)
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/da3_env_hpc3.sh
export MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID} + ${SLURM_ARRAY_TASK_ID:-0}) % 20000 ))
export PYTHONPATH=$PYTHONPATH:/path/to/pandwm_root/pandwm_pylibs
OV=$DR/pandwm_code/PanDWM-ft3-20261030
export PYTHONPATH=$OV/gs4d_src:$OV/gsaa_src:$PYTHONPATH
R=$DR/pandwm_runs/gs_dev_hpc3
CK=${CKPT:-$RUNS/c2_hpc2/checkpoints/3000.pth}
VAL_PKL=$DR/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl
ARM=${ARM:?ARM}; MODE=${MODE:?MODE=render|score}
mkdir -p $R/logs $R/scores
cd $GSAA
echo "DEV_START code=$OV arm=$ARM mode=$MODE task=${SLURM_ARRAY_TASK_ID:-0} ckpt=$CK ovr=${OVR:-} node=$(hostname) $(date -Is)"
case $MODE in
  render)
    N=${SLURM_ARRAY_TASK_COUNT:-1}; K=${SLURM_ARRAY_TASK_ID:-0}
    SAMPLES=$($PY -c "print(','.join(map(str, range(78)[$K::$N])))")
    $PY -m dwm_pano.recon.render_ours -c $GS4D/configs/gs_review_hpc3.json --data-pkl $VAL_PKL --checkpoint $CK \
      --targets $R/targets.json --out-root $R/renders/$ARM --resolutions 224x400 --modes ${MODES:-3d,4d} \
      --samples "$SAMPLES" --da3-src $DA3_SRC --da3-anyview $DA3_ANYVIEW --da3-metric $DA3_METRIC \
      ${OVR:+--scene-overrides "$OVR"} ;;
  score)
    for m in ${MODES:-3d 4d}; do
      $PY -m dwm_pano.recon.score --targets $R/targets.json --masks $R/masks --renders $R/renders/$ARM/$m \
        --res 224x400 --out $R/scores/${ARM}_${m}.json || exit 1
    done ;;
esac
rc=$?
echo "DEV_END rc=$rc $(date -Is)"
exit $rc
