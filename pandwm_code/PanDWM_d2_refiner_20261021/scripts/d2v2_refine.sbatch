#!/usr/bin/env bash
#SBATCH --job-name=d2v2_ref
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/d2v2_20261031/logs/%x_%A_%a.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58,ACD1-59
# 2026-10-31 D2 v2 (D2V2_PLAN_20261031.md section 3): d2_refine.sbatch with inputs / outputs under
# pandwm_runs/d2v2_20261031 (SET=dev: inputs_dev, the GS-Dev 78 clips of the 149-clip protocol; SET=test with
# MANIFEST=tests213_manifest_20261018.json FS=10,5: Test-S).  Copied from:
# 2026-10-21 D2 refiner inference on the review clips (d2_refine.py; D2_PLAN_20261021.md Tasks 8 / 10 / 11).
#   CK=<run>/checkpoints/1000.pth OUT=<tag> CLIPS=<list> [SHIFTS=2,-2 STEPS=20 GMODE=none GSCALE=1 INIT_SIGMA=0 INPUTS=c2b_drop2]
#   [NOSTRICT=1 VIDEO=<list>] sbatch --array=0-3 d2_refine.sbatch      (array tasks split the clip list)
set -uo pipefail
DR=/path/to/pandwm_root
D2=$DR/pandwm_code/PanDWM_d2_refiner_20261021
C=$DR/pandwm_code/PanDWM_eval149_20261011
R=$DR/pandwm_runs/d2v2_20261031
PY=$DR/pandwm_venv/bin/python
export PYTHONPATH=$D2/src:$D2/src_base:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID} * 7 + ${SLURM_ARRAY_TASK_ID:-0} * 131 + RANDOM) % 20000 ))   # 11-01: id-based ports collided (785507)
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR || true
N=${SLURM_ARRAY_TASK_COUNT:-1}; K=${SLURM_ARRAY_TASK_ID:-0}
CL=$($PY -c "import sys; t=sys.argv[1]; s=(list(range(*map(int,t.split(':')))) if ':' in t else [int(x) for x in t.split(',') if x]); print(','.join(map(str,s[int(sys.argv[3])::int(sys.argv[2])])))" "${CLIPS:?CLIPS}" $N $K)
EXTRA=""; [[ ${NOSTRICT:-0} == 1 ]] && EXTRA="--no-strict"
echo "D2REF_START ck=${CK:?CK} out=${OUT:?OUT} clips=$CL shifts=${SHIFTS:-0,1,-1,2,-2,4,-4} steps=${STEPS:-20} g=${GMODE:-none}/${GSCALE:-1} init=${INIT_SIGMA:-0} seed=${SEED:-42} node=$(hostname) $(date -Is)"
cd $D2 || exit 1
$PY d2_refine.py --config ${CONFIG:-$D2/configs/d2v2_hpc3_8gpu.json} --checkpoint $CK $EXTRA \
  --eval-config $C/cd_eval_frozen149_hpc3.json --manifest $C/${MANIFEST:-dev149_manifest_20261018.json} --fps-stride ${FS:-10,20} \
  --inputs $R/inputs_${SET:-dev} --out-root $R/refined_${SET:-dev}/$OUT --clips "$CL" --shifts ${SHIFTS:-0,1,-1,2,-2,4,-4} \
  --steps ${STEPS:-20} --guidance-mode ${GMODE:-none} --guidance-scale ${GSCALE:-1} --init-sigma ${INIT_SIGMA:-0} \
  --video-clips "${VIDEO:-}" --seed ${SEED:-42}
rc=$?
echo "D2REF_END rc=$rc $(date -Is)"
exit $rc
