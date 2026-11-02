#!/usr/bin/env bash
# 2026-10-12 17:00: copy of eval149_hpc3_gen2.sbatch whose base config can be swapped (EVAL_CFG), for the inference-only
# step-schedule test of the v8.3 student: cd_eval_frozen149_hpc3_s13.json is the frozen-149 config with
# pipeline.denoising_step_list [1000, 625, 250, 42] (the s13 schedule that took B0 from FVD 79.3 to 70.1 on 2026-09-03)
# instead of the training schedule [1000, 750, 500, 250].  Still 4 network evaluations.  Everything else unchanged.
# 2026-10-12: copy of eval149_hpc3_gen.sbatch with an optional launcher in front of lowmem_launch.py (LAUNCH, e.g.
# corr_noise_launch.py for the temporally correlated sampler noise test; its PANDWM_RENOISE_CORR / PANDWM_INIT_CORR come
# through --export=ALL).  Everything else unchanged.  Original header:
# 2026-10-11 19:10: the generation phase of scripts/eval149_hpc3.sbatch split into 1-GPU array tasks.  The 8-GPU job
# (709178) found no node with 8 free GPUs (cluster full, estimated start 22:57); single-GPU tasks start wherever a GPU
# frees up.  Task i renders arm ARMS[i / 8], clip range i % 8 (the same 8 ranges as the 8-GPU job), so the most
# important arms come first: v8.2 step 2000 and its starting point v8.1 step 1600.  Same code, config, seed and
# settings as eval149_hpc3.sbatch; finished clips are skipped, so a rerun resumes.
#   sbatch --array=0-39%8 scripts/eval149_hpc3_gen.sbatch      (then eval149_hpc3_met.sbatch, eval149_hpc3_fin.sbatch)
#SBATCH --job-name=pandwm_e149g
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=01:30:00
#SBATCH --output=logs_eval149/%x_%A_%a.log
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
CFG=${EVAL_CFG:-$C/cd_eval_frozen149_hpc3.json}
OUT=${EVAL_OUT:-$DR/pandwm_runs/eval149_hpc3_20261011}
PY=$DR/pandwm_venv/bin/python
R=$DR/pandwm_runs
ARMS=${ARMS:-"v82g2000=$R/cfdmd_20261010_G/gan/milestones/2000_ema.pth v81f1600=$R/seeds/F_1600/gan/checkpoints/1600_ema.pth v7d1000=$R/cfdmd_20261010_D/gan/milestones/1000_ema.pth v82g1750=$R/cfdmd_20261010_G/gan/milestones/1750_ema.pth b0=$DR/pandwm_data/cd_10000_ema.pth"}
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29400 + ${SLURM_ARRAY_TASK_ID:-0} % 1000))
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR PANDWM_HISTORY_GUIDANCE || true
read -ra SPECS <<< "$ARMS"
RANGES=(0 19 38 57 76 95 114 132 149)
i=${SLURM_ARRAY_TASK_ID:?array task}; a=$((i / 8)); r=$((i % 8))
spec=${SPECS[$a]:?no arm for task $i}; tag=${spec%%=*}; ck=${spec#*=}
s=${RANGES[$r]}; e=${RANGES[$((r + 1))]}
clips_in() { local n=0 k; for ((k = s; k < e; k++)); do [ -f "$(printf '%s/%s/clip%03d/summary.json' "$OUT" "$tag" "$k")" ] && n=$((n + 1)); done; echo $n; }
echo "E149G_START $(date -Is) node=$(hostname) task=$i arm=$tag range=[$s,$e) ckpt=$ck cfg=$(basename "$CFG") launch=${LAUNCH:-none} renoise_corr=${PANDWM_RENOISE_CORR:-0} init_corr=${PANDWM_INIT_CORR:-0}"
[ -f "$CFG" ] || { echo "EVAL_TREE_MISSING"; exit 1; }
[ -f "$ck" ] || { echo "ARM_MISSING $tag $ck"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$C"
for attempt in 1 2; do
  "$PY" ${LAUNCH:-} lowmem_launch.py run_eval149_videos.py --base-config "$CFG" --checkpoint "$ck" --out-root "$OUT/$tag" \
    --start $s --end $e --seed 42 --history-mode clean
  echo "$(date +%T) attempt $attempt rc=$? clips=$(clips_in)/$((e - s))"
  [ "$(clips_in)" = $((e - s)) ] && { echo "E149G_DONE $tag [$s,$e)"; exit 0; }
done
echo "E149G_INCOMPLETE $tag [$s,$e) clips=$(clips_in)/$((e - s))"
exit 1
