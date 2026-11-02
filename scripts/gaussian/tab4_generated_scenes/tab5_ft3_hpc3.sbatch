#!/usr/bin/env bash
#SBATCH --job-name=ft3_tab5
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/tab5_gen_gs_ft3_20261030/logs/%x_%A_%a.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58,ACD1-59
# 2026-10-30 round 3 (GS_FT3_PLAN_20261030.md section 4): tab5_shift_hpc3.sbatch on the 71-clip Test-S subset with
# the PanDWM-ft3-20261030 code copy (sharpness, no-offset), per-ARM states / report / samples, scene flags from
# DA3FLAGS / EXTRA, and the s71 clip list built in.  Copied from:
# 2026-10-23 Table 5 (Gaussians on generated Test-S scenes), our row: dwm_pano.eval_gs_shift on the v84i4250 s42
# Test-S generations, fixed 4D scene (section 44: camera-time poses, mmdet3d v0.x boxes) with the v4 settings
# (C2 step 3000, DA3 scale relabel, 2 bottom rows dropped), shifts 0 / +-1 / +-2 / +-4 m, plus LiDAR depth of the 0 m
# render (recon/gen_depth_pack.py).  Task i evaluates clips [i*CHUNK, (i+1)*CHUNK); MERGE=1 merges.
#   [CHUNK=18] sbatch --array=0-11 tab5_shift_hpc3.sbatch ;  MERGE=1 sbatch tab5_shift_hpc3.sbatch
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/common_env_hpc3.sh
source /path/to/pandwm_root/pandwm_code/PanDWM-gs4d-20261002/scripts_hpc3/da3_env_hpc3.sh
export MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID} + ${SLURM_ARRAY_TASK_ID:-0}) % 20000 ))
export PANDWM_PANO_BLEND_W=128
OV=$DR/pandwm_code/PanDWM-ft3-20261030
export PYTHONPATH=$OV/gs4d_src:$OV/gsaa_src:$PYTHONPATH
ARM=${ARM:?ARM}
CLIP_LIST=$(python3 -c "print(','.join(map(str, range(1, 213, 3))))"); SUBSET=$ARM
OUT=$DR/pandwm_runs/tab5_gen_gs_ft3_20261030
PACK=$DR/pandwm_runs/tab5_gen_gs_20261023/depth_pack
ST=$OUT/states${SUBSET:+_$SUBSET}
CK=${CKPT:-$RUNS/c2_hpc2/checkpoints/3000.pth}
VAL_PKL=$DR/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl
CLIPS=${CLIPS_ROOT:-$DR/pandwm_runs/tests_20261018/v84i4250_s42}
CHUNK=${CHUNK:-18}
TASK=${SLURM_ARRAY_TASK_ID:-0}
S=$(( TASK * CHUNK )); E=$(( S + CHUNK )); if (( E > 213 )); then E=213; fi
mkdir -p $ST $OUT/logs
# SUBSET=s71 CLIP_LIST=<71 test indices, one per scene> [TASKS=6]: task i takes CLIP_LIST[i::TASKS]
if [ -n "${CLIP_LIST:-}" ]; then
  SEL=$(python3 -c "import sys; l=sys.argv[1].split(','); n,k=int(sys.argv[2]),int(sys.argv[3]); print(','.join(l[k::n]))" "$CLIP_LIST" ${TASKS:-6} $TASK)
  SPAN="--clip-list $SEL"; NEXP=$(echo "$CLIP_LIST" | tr ',' '\n' | grep -c .)
else
  SPAN="--start $S --end $E"; NEXP=213
fi
cd $GSAA
COMMON="-c $GS4D/configs/gs_review_hpc3.json --data-pkl $VAL_PKL --checkpoint $CK --clips-root $CLIPS \
  --i3d $DR/pandwm_code/TATS_fvd/i3d_pretrained_400.pt --shifts 0,1,2,4 --modes 4d ${DA3FLAGS---da3-relabel --da3-mode scale} \
  --drop-bottom-rows 2 --fps-stride 10,5 --expected-clips $NEXP --box-reading mmdet3d_v0x --camera-time-poses \
  --depth-pack $PACK --depth-res 224x400 --da3-src $DA3_SRC --da3-anyview $DA3_ANYVIEW --da3-metric $DA3_METRIC ${EXTRA:-}"
echo "TAB5_START code=$OV arm=$ARM ckpt=$CK extra=${EXTRA:-} task=$TASK clips=[$S,$E) node=$(hostname) $(date -Is)"
if [ -n "${MERGE:-}" ]; then
  $PY -m dwm_pano.eval_gs_shift $COMMON --state $ST/merged.pt --merge $ST/state_*.pt \
    --output $OUT/report_ours4d${SUBSET:+_$SUBSET}.json --device cpu
else
  $PY -m dwm_pano.eval_gs_shift $COMMON $SPAN --state $ST/state_$(printf %02d $TASK).pt \
    --sample-clips "${SAMPLE_CLIPS:-1,58,118,178}" --sample-dir $OUT/samples_$ARM
fi
rc=$?
echo "TAB5_END task=$TASK rc=$rc $(date -Is)"
exit $rc
