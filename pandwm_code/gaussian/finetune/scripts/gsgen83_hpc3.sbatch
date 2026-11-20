#!/usr/bin/env bash
#SBATCH --job-name=gsgen83
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/d2v2/logs/%x_%j.log
# Evaluation of the Gaussian row on 83 generated test clips.
#   MODE=g3b   eval_gs_shift (with depth) on the 12 extension clips, then merges of the existing 6 states and of
#              existing + new states
#   MODE=merge TAG=<refiner tag>   d2v2_eval merge of eval_test/<TAG> (71) + eval_testext/<TAG> (12) into eval_test83/<TAG>
set -uo pipefail
source /path/to/pandwm_root/pandwm_code/gaussian/gs4d/scripts_hpc3/common_env_hpc3.sh
source /path/to/pandwm_root/pandwm_code/gaussian/gs4d/scripts_hpc3/da3_env_hpc3.sh
export MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID} * 7 + RANDOM) % 20000 ))
export PANDWM_PANO_BLEND_W=128
OV=$DR/pandwm_code/gaussian/finetune
export PYTHONPATH=$DR/pandwm_code/src:$PYTHONPATH
VAL_PKL=$DR/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl
I3D=$DR/pandwm_code/TATS_fvd/i3d_pretrained_400.pt
NEW=214,217,220,223,226,229,232,235,238,241,244,247
cd $GSAA
echo "GSGEN83_START mode=$MODE tag=${TAG:-} node=$(hostname) $(date -Is)"
case $MODE in
  g3b)
    O=$DR/pandwm_runs/tab5_gen_gs_ft3; X=$DR/pandwm_runs/tab5_ext83
    CK=$RUNS/ft3/G3b/2000.pth
    common() {   # $1 clips root, $2 depth pack, $3 expected clips
      echo "-c $GS4D/configs/gs_review_hpc3.json --data-pkl $VAL_PKL --checkpoint $CK --clips-root $1 --i3d $I3D \
      --shifts 0,1,2,4 --modes 4d --da3-relabel --da3-mode scale --drop-bottom-rows 2 --fps-stride 10,5 \
      --expected-clips $3 --box-reading mmdet3d_v0x --camera-time-poses --depth-pack $2 --depth-res 224x400 \
      --da3-src $DA3_SRC --da3-anyview $DA3_ANYVIEW --da3-metric $DA3_METRIC --da3-window clip --da3-fuse --plane-offset-px 1"
    }
    mkdir -p $O/states_FT3_G3b_ext12 $O/states_FT3_G3b_merge
    $PY -m dwm_pano.eval_gs_shift $(common $DR/pandwm_runs/tests_ext/v84i4250_s42 $X/depth_pack_ext36 12) \
      --clip-list $NEW --state $O/states_FT3_G3b_ext12/state_00.pt --sample-clips "" --sample-dir $O/samples_FT3_G3b_ext12
    echo "STEP1_EXT12 rc=$?"
    OLDST=$(ls $O/states_FT3_G3b/state_0[0-5].pt | tr '\n' ' ')
    $PY -m dwm_pano.eval_gs_shift $(common $DR/pandwm_runs/tests/v84i4250_s42 $DR/pandwm_runs/tab5_gen_gs/depth_pack 71) \
      --state $O/states_FT3_G3b_merge/merged_repro71.pt --merge $OLDST --output $O/report_ours4d_FT3_G3b_repro71.json --device cpu
    echo "STEP2_REPRO71 rc=$?"
    $PY $X/scripts/compare_ours.py report $O/report_ours4d_FT3_G3b.json $O/report_ours4d_FT3_G3b_repro71.json
    $PY -m dwm_pano.eval_gs_shift $(common $DR/pandwm_runs/tests_ext/v84i4250_s42 $X/depth_pack_ext36 83) \
      --state $O/states_FT3_G3b_merge/merged_ext83.pt --merge $OLDST $O/states_FT3_G3b_ext12/state_00.pt \
      --output $O/report_ours4d_FT3_G3b_ext83.json --device cpu
    echo "STEP3_MERGE83 rc=$?" ;;
  merge)
    R=$DR/pandwm_runs/d2v2; TAG=${TAG:?TAG}; ST=$R/eval_test83/$TAG; mkdir -p $ST
    $PY -m dwm_pano.d2v2_eval -c $GS4D/configs/gs_review_hpc3.json --data-pkl $VAL_PKL --inputs $R/inputs_test \
      --refined $R/refined_test/$TAG --fps-stride 10,5 --i3d $I3D --expected-clips 83 --state $ST/merged.pt \
      --merge $R/eval_test/$TAG/state_*.pt $R/eval_testext/$TAG/state_*.pt --output $ST/report.json --device cpu
    echo "MERGE83 rc=$?" ;;
esac
echo "GSGEN83_END $(date -Is)"
