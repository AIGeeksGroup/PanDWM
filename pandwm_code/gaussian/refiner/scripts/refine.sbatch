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
#SBATCH --output=/path/to/pandwm_root/pandwm_runs/d2v2/logs/%x_%A_%a.log
# Render refiner inference (refine.py) on exported lateral-shift renders.
#   CK=<ckpt> OUT=<tag> CLIPS=<list> [SHIFTS=2,-2 STEPS=20 GMODE=none GSCALE=1 INIT_SIGMA=0 INPUTS=...]
#   [NOSTRICT=1 VIDEO=<list> SET=dev MANIFEST=... FS=10,20] sbatch --array=0-3 refine.sbatch   (array tasks split the clip list)
set -uo pipefail
DR=/path/to/pandwm_root
D2=$DR/pandwm_code/gaussian/refiner
C=$DR/pandwm_code/eval
R=$DR/pandwm_runs/d2v2
PY=$DR/pandwm_venv/bin/python
export PYTHONPATH=$DR/pandwm_code/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID} * 7 + ${SLURM_ARRAY_TASK_ID:-0} * 131 + RANDOM) % 20000 ))   # random term avoids port clashes between array tasks
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR || true
N=${SLURM_ARRAY_TASK_COUNT:-1}; K=${SLURM_ARRAY_TASK_ID:-0}
CL=$($PY -c "import sys; t=sys.argv[1]; s=(list(range(*map(int,t.split(':')))) if ':' in t else [int(x) for x in t.split(',') if x]); print(','.join(map(str,s[int(sys.argv[3])::int(sys.argv[2])])))" "${CLIPS:?CLIPS}" $N $K)
EXTRA=""; [[ ${NOSTRICT:-0} == 1 ]] && EXTRA="--no-strict"
echo "D2REF_START ck=${CK:?CK} out=${OUT:?OUT} clips=$CL shifts=${SHIFTS:-0,1,-1,2,-2,4,-4} steps=${STEPS:-20} g=${GMODE:-none}/${GSCALE:-1} init=${INIT_SIGMA:-0} seed=${SEED:-42} node=$(hostname) $(date -Is)"
cd $D2 || exit 1
$PY refine.py --config ${CONFIG:-$D2/configs/infer_hpc3_8gpu.json} --checkpoint $CK $EXTRA \
  --eval-config $C/manifests/cd_eval_frozen149_hpc3.json --manifest $C/${MANIFEST:-manifests/dev149_manifest.json} --fps-stride ${FS:-10,20} \
  --inputs $R/inputs_${SET:-dev} --out-root $R/refined_${SET:-dev}/$OUT --clips "$CL" --shifts ${SHIFTS:-0,1,-1,2,-2,4,-4} \
  --steps ${STEPS:-20} --guidance-mode ${GMODE:-none} --guidance-scale ${GSCALE:-1} --init-sigma ${INIT_SIGMA:-0} \
  --video-clips "${VIDEO:-}" --seed ${SEED:-42}
rc=$?
echo "D2REF_END rc=$rc $(date -Is)"
exit $rc
