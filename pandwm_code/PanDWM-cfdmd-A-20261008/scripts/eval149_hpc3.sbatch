#!/usr/bin/env bash
# 2026-10-11 (user's choice: evaluate on HPC3 once v8.2 training ends): the frozen 149-clip evaluation of several EMA
# checkpoints on one HPC3 node (8 H100), with the HPC1 protocol's code and config (tree built by
# scripts/setup_eval149_hpc3.sh).  H100 numbers are compared only with each other: the anchors (v8.1 step 1600 =
# v8.2's starting point, v7 step 1000, B0) are rerun here next to v8.2 steps 1750 and 2000.
#   Phase 1: GPU i renders its fixed clip range for every arm in turn (run_eval149_videos.py through lowmem_launch.py,
#            seed 42, clean history; each clip's sampler seed depends only on (seed, clip index), so the split does not
#            change the output).  A range that comes back incomplete is rerun once (finished clips are skipped).
#   Phase 2: FID / FVD (compute_eval149_metrics.py, all arms per clip range, merged into one report) on the GPUs, look
#            stats and frame-to-frame change (realism_arms.py, b0 first as in realism3.py) on the CPU meanwhile.
#   sbatch --dependency=afterany:<training job> scripts/eval149_hpc3.sbatch
# ARMS="tag=ckpt ..." overrides the arm list; missing checkpoints are skipped with ARM_MISSING.  Ends EVAL149_HPC3_DONE.
#SBATCH --job-name=pandwm_eval149
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
CFG=$C/cd_eval_frozen149_hpc3.json
OUT=${EVAL_OUT:-$DR/pandwm_runs/eval149_hpc3_20261011}
PY=$DR/pandwm_venv/bin/python
R=$DR/pandwm_runs
ARMS=${ARMS:-"v82g2000=$R/cfdmd_20261010_G/gan/milestones/2000_ema.pth v81f1600=$R/seeds/F_1600/gan/checkpoints/1600_ema.pth v7d1000=$R/cfdmd_20261010_D/gan/milestones/1000_ema.pth v82g1750=$R/cfdmd_20261010_G/gan/milestones/1750_ema.pth b0=$DR/pandwm_data/cd_10000_ema.pth"}
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR PANDWM_HISTORY_GUIDANCE || true
[ -f "$CFG" ] && [ -f "$C/run_eval149_videos.py" ] || { echo "EVAL_TREE_MISSING $C (run scripts/setup_eval149_hpc3.sh)"; exit 1; }
mkdir -p "$OUT/logs" "$OUT/metrics"
cd "$C"
echo "EVAL149_HPC3_START $(date -Is) node=$(hostname) job=${SLURM_JOB_ID:-} out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
read -ra SPECS <<< "$ARMS"
TAGS=()
for spec in "${SPECS[@]}"; do
  tag=${spec%%=*}; ck=${spec#*=}
  if [ -f "$ck" ]; then TAGS+=("$tag"); echo "ARM $tag $ck $(stat -c %s "$ck")"; else echo "ARM_MISSING $tag $ck"; fi
done
[ ${#TAGS[@]} -gt 0 ] || { echo "NO_ARMS"; exit 1; }
RANGES=(0 19 38 57 76 95 114 132 149)
clips_in() {   # <tag> <start> <end> -> finished clips in [start, end)
  local n=0 i
  for ((i = $2; i < $3; i++)); do [ -f "$(printf '%s/%s/clip%03d/summary.json' "$OUT" "$1" "$i")" ] && n=$((n + 1)); done
  echo $n
}

# Phase 1: generation
gen_worker() {   # <gpu index>
  local g=$1 s=${RANGES[$1]} e=${RANGES[$(( $1 + 1 ))]} spec tag ck a k=0
  for spec in "${SPECS[@]}"; do
    tag=${spec%%=*}; ck=${spec#*=}; k=$((k + 1))
    [ -f "$ck" ] || continue
    for a in 1 2; do
      CUDA_VISIBLE_DEVICES=$g MASTER_PORT=$((29400 + 20 * g + 2 * k + a)) \
      TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x}_$g TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}_$g \
        "$PY" lowmem_launch.py run_eval149_videos.py --base-config "$CFG" --checkpoint "$ck" \
        --out-root "$OUT/$tag" --start $s --end $e --seed 42 --history-mode clean >> "$OUT/logs/gen_${tag}_$s.log" 2>&1
      echo "$(date +%T) gen $tag [$s,$e) attempt $a rc=$? clips=$(clips_in "$tag" $s $e)/$((e - s))"
      [ "$(clips_in "$tag" $s $e)" = $((e - s)) ] && break
    done
  done
}
for g in 0 1 2 3 4 5 6 7; do gen_worker $g > "$OUT/logs/worker_$g.log" 2>&1 & done
wait
cat "$OUT"/logs/worker_*.log
DONE_TAGS=()
for tag in "${TAGS[@]}"; do
  n=$(clips_in "$tag" 0 149); echo "GEN_RESULT $tag clips=$n/149"
  [ "$n" = 149 ] && DONE_TAGS+=("$tag")
done
[ ${#DONE_TAGS[@]} -gt 0 ] || { echo "NO_COMPLETE_ARM"; exit 1; }
echo "PHASE1_DONE $(date -Is) complete arms: ${DONE_TAGS[*]}"

# Phase 2: FID / FVD on the GPUs, look stats on the CPU
MARGS=()
for tag in "${DONE_TAGS[@]}"; do MARGS+=(--arm "$tag=$OUT/$tag"); done
RTAGS=()
for tag in "${DONE_TAGS[@]}"; do [ "$tag" = b0 ] && RTAGS=(b0 "${RTAGS[@]}") || RTAGS+=("$tag"); done
OMP_NUM_THREADS=16 "$PY" realism_arms.py "$OUT" "$OUT/realism_arms.json" "${RTAGS[@]}" > "$OUT/logs/realism.log" 2>&1 &
RPID=$!
met_worker() {   # <gpu index>
  local g=$1 s=${RANGES[$1]} e=${RANGES[$(( $1 + 1 ))]} a
  for a in 1 2; do
    CUDA_VISIBLE_DEVICES=$g "$PY" compute_eval149_metrics.py --base-config "$CFG" "${MARGS[@]}" \
      --state "$OUT/metrics/state_${s}_$e.pt" --output "$OUT/metrics/report_$s.json" \
      --clip-start $s --clip-end $e >> "$OUT/logs/met_$s.log" 2>&1
    grep -q SHARD_DONE "$OUT/logs/met_$s.log" && break
    echo "$(date +%T) metrics [$s,$e) attempt $a failed"
  done
}
for g in 0 1 2 3 4 5 6 7; do met_worker $g & done
wait $(jobs -p | grep -v "^$RPID$")
n=$(grep -l SHARD_DONE "$OUT"/logs/met_*.log | wc -l)
echo "METRICS_SHARDS_DONE $n/8"
if [ "$n" = 8 ]; then
  CUDA_VISIBLE_DEVICES=0 "$PY" compute_eval149_metrics.py --base-config "$CFG" "${MARGS[@]}" \
    --state "$OUT/metrics/unused.pt" --output "$OUT/metrics/report.json" --merge "$OUT"/metrics/state_*.pt \
    > "$OUT/logs/met_merge.log" 2>&1
  grep -q METRICS_DONE "$OUT/logs/met_merge.log" && cat "$OUT/metrics/report.json" || { echo "MERGE_FAILED"; tail -20 "$OUT/logs/met_merge.log"; }
fi
wait $RPID
tail -40 "$OUT/logs/realism.log"
echo "EVAL149_HPC3_DONE $(date -Is)"
