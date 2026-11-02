#!/usr/bin/env bash
# 2026-10-18: acceptance of PanDWM_eval149_20261011/run_clips_videos.py before Test-S.
#  1. EQUALITY: Dev-149 clips 0 and 1 of the locked model (v84i4250 EMA) through run_clips_videos.py with the Dev-149
#     manifest (stride 20, test_index = clip index) must give latents.pt bit-identical to the run_eval149_videos.py
#     outputs in eval149_hpc3_v83_20261012/v84i4250 (same seed 42, same sampler settings).
#  2. SMOKE (one clip each, outputs only checked for shape / finiteness): --history none (setting G) with the student;
#     the causal teacher (stage1_causal_20000.pth) with 4 guided steps; the bidirectional Stage 0 with 4 guided steps.
#   sbatch scripts/verify_clips_runner.sbatch
#SBATCH --job-name=pandwm_vclips
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=01:30:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
CFG=$C/cd_eval_frozen149_hpc3.json
PY=$DR/pandwm_venv/bin/python
R=$DR/pandwm_runs
CK=$R/cfdmd_20261013_I/gan/milestones/4250_ema.pth
REF=$R/eval149_hpc3_v83_20261012/v84i4250
V=$R/verify_clips_runner_20261018
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR PANDWM_HISTORY_GUIDANCE || true
port() { "$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
echo "VCLIPS_START $(date -Is) node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$C" || exit 1
run() { MASTER_PORT=$(port) "$PY" lowmem_launch.py run_clips_videos.py --base-config "$CFG" "$@"; echo "RUN_RC $? $*"; }
run --checkpoint "$CK" --out-root "$V/student_rec" --manifest dev149_manifest_20261018.json --dataset-stride 20 \
  --positions 0 2 --seed 42
"$PY" - "$V/student_rec" "$REF" <<'EOF'
import sys, torch
new, ref = sys.argv[1:]
ok = True
for i in (0, 1):
    a = torch.load(f"{new}/clip{i:03d}/latents.pt", map_location="cpu", weights_only=True)["latents"]
    b = torch.load(f"{ref}/clip{i:03d}/latents.pt", map_location="cpu", weights_only=True)["latents"]
    same = a.shape == b.shape and torch.equal(a, b)
    ok &= same
    print(f"clip {i}: shape {tuple(a.shape)} equal={same} maxdiff={(a.float() - b.float()).abs().max().item():.3e}")
print("VERIFY_CLIPS_EQUAL", "PASS" if ok else "FAIL")
EOF
run --checkpoint "$CK" --out-root "$V/student_none" --manifest dev78_manifest_20261018.json --dataset-stride 20 \
  --positions 0 1 --seed 42 --history none
run --checkpoint "$DR/pandwm_data/stage1_causal_20000.pth" --out-root "$V/teacher4" \
  --manifest tests213_manifest_20261018.json --dataset-stride 5 --positions 0 1 --seed 42 --engine teacher \
  --teacher-steps 4
run --checkpoint "$DR/pandwm_data/stage0_bidirectional_20000.pth" --out-root "$V/stage0_4" \
  --manifest tests213_manifest_20261018.json --dataset-stride 5 --positions 0 1 --seed 42 --engine teacher \
  --teacher-steps 4 --bidirectional
"$PY" - "$V" <<'EOF'
import glob, json, os, sys, torch
v = sys.argv[1]
dirs = [os.path.relpath(p, v) for p in sorted(glob.glob(f"{v}/student_none/clip*"))] or ["student_none/missing"]
for d in dirs + ["teacher4/clip000", "stage0_4/clip000"]:
    try:
        p = torch.load(f"{v}/{d}/latents.pt", map_location="cpu", weights_only=True)
        s = json.load(open(f"{v}/{d}/summary.json"))
        x = p["latents"].float()
        print(f"SMOKE {d} shape={tuple(x.shape)} finite={bool(torch.isfinite(x).all())} std={x.std():.3f} "
              f"psnr={s['psnr_mean_db_per_window']} t={s['runtime_seconds']}s history={p.get('history')}")
    except Exception as e:  # noqa: BLE001
        print(f"SMOKE {d} FAIL {e!r}")
EOF
echo "VCLIPS_END $(date -Is)"
