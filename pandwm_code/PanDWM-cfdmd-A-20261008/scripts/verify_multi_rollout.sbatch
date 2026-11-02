#!/usr/bin/env bash
# 2026-10-14: equality check of run_long_rollout_multi.py against single-scene run_long_rollout(.py/_v2.py) outputs.
# Two scenes in ONE process (scene 0 then 14, v84i4250 EMA, seed 42, 12-window clips, first 2 windows) must give
# bit-identical window01/02.pt latents to long_videos_hpc3_20261011/v84i4250_scene{0,14}_seed42.
#SBATCH --job-name=pandwm_vmulti
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=logs_eval149/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
C=$DR/pandwm_code/PanDWM_eval149_20261011
LV=$DR/pandwm_runs/long_videos_hpc3_20261011
OUT=$DR/pandwm_runs/verify_multi_20261014
PY=$DR/pandwm_venv/bin/python
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_4STEP_SAMPLER=stochastic PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch
export OMP_NUM_THREADS=12 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID:-x} TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID:-x}
unset HF_AUTO_UPLOAD_REPO CKPT_OUT_DIR PANDWM_HISTORY_GUIDANCE || true
mkdir -p "$OUT"
cat > "$OUT/two_scenes.json" <<'J'
{"scenes": [{"dev76_index": 0, "scene_token": "2f56eb47c64f43df8902d9f88aa8a019"},
            {"dev76_index": 14, "scene_token": "ed242d80ccb34b139aaf9ab89859332e"}]}
J
cd "$C" || exit 1
export MASTER_PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$PY" lowmem_launch.py run_long_rollout_multi.py --base-config "$C/cd_eval_frozen149_hpc3.json" \
  --checkpoint "$DR/pandwm_runs/cfdmd_20261013_I/gan/milestones/4250_ema.pth" --output-root "$OUT" --tag v84i4250 \
  --scene-list "$OUT/two_scenes.json" --indices 0 14 --seed 42 --windows 12 --debug-windows 2
"$PY" - "$OUT" "$LV" <<'PY'
import sys
import torch
out, lv = sys.argv[1], sys.argv[2]
ok = True
for sc in (0, 14):
    for w in (1, 2):
        a = torch.load(f"{out}/v84i4250_scene{sc}_seed42/window{w:02d}.pt")
        b = torch.load(f"{lv}/v84i4250_scene{sc}_seed42/window{w:02d}.pt")
        same = all(torch.equal(a[k], b[k]) for k in ("history_latent", "future_latent"))
        diff = float((a["future_latent"].float() - b["future_latent"].float()).abs().max())
        print(f"scene {sc} window {w}: {'IDENTICAL' if same else 'DIFFERENT'} (max |diff| {diff:.3g})")
        ok &= same
print("VERIFY_MULTI", "PASS" if ok else "FAIL")
PY
