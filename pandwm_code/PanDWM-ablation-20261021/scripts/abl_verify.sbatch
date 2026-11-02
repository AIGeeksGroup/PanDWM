#!/usr/bin/env bash
# 2026-10-21 ablation: check that the eval generation path (scripts/tests_gen.sbatch of PanDWM-cfdmd-A-20261008,
# unchanged, with MANIFEST=dev78_manifest_20261018.json STRIDE=20 SEED=42) reproduces the existing Dev-78 clips of two
# known checkpoints bit for bit, on Dev-78 position 0 (Dev-149 clip 0):
#   v84i4250 EMA (locked)          vs pandwm_runs/eval149_hpc3_v83_20261012/v84i4250/clip000/latents.pt
#   b0 = cd_10000_ema.pth           vs pandwm_runs/eval149_hpc3_v83_20261012/b0/clip000/latents.pt
# (these two runs are the sources of feats_dev78/feats_{v84i4250P,b0P}_s42.npz), and that clip_feats.py gives the
# stored feature row of v84i4250 clip 0.  Exit 0 only if all three hold; then the b0 seed-42 Dev-78 features are copied
# into the ablation feature dir as feats_b0_s42.npz (reused; only b0 seed 1000 is generated).
#SBATCH --job-name=pandwm_abl_verify
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --exclude=ACD1-3,ACD1-24,ACD1-50,ACD1-54,ACD1-56,ACD1-58
set -uo pipefail
DR=/path/to/pandwm_root
CF=$DR/pandwm_code/PanDWM-cfdmd-A-20261008
C=$DR/pandwm_code/PanDWM_eval149_20261011
ABL=$DR/pandwm_code/PanDWM-ablation-20261021
EV=$DR/pandwm_runs/abl_20261021_eval
REF=$DR/pandwm_runs/eval149_hpc3_v83_20261012
OLDF=$DR/pandwm_runs/tests_20261018/feats_dev78
PY=$DR/pandwm_venv/bin/python
VO=$EV/verify
mkdir -p "$VO" "$EV/feats"
echo "ABL_VERIFY_START $(date -Is) node=$(hostname) job=${SLURM_JOB_ID:-}"
fail=0
for spec in "v84i4250=$DR/pandwm_runs/cfdmd_20261013_I/gan/milestones/4250_ema.pth" "b0=$DR/pandwm_data/cd_10000_ema.pth"; do
  name=${spec%%=*}; ck=${spec#*=}
  OUT=$VO TAG=verify_$name CKPT=$ck SEED=42 MANIFEST=dev78_manifest_20261018.json STRIDE=20 CHUNK=1 BUDGET=2400 \
    bash "$CF/scripts/tests_gen.sbatch" || { echo "VERIFY_GEN_FAIL $name"; fail=1; }
done
cd "$C" || exit 1
"$PY" - "$VO" "$REF" <<'PY' || fail=1
import sys, torch
vo, ref = sys.argv[1], sys.argv[2]
bad = 0
for name in ("v84i4250", "b0"):
    a = torch.load(f"{vo}/verify_{name}_s42/clip000/latents.pt", map_location="cpu", weights_only=True)["latents"]
    b = torch.load(f"{ref}/{name}/clip000/latents.pt", map_location="cpu", weights_only=True)["latents"]
    eq = a.shape == b.shape and torch.equal(a, b)
    d = float((a.float() - b.float()).abs().max()) if a.shape == b.shape else float("nan")
    print(f"VERIFY_LATENTS {name} {'EQUAL' if eq else 'DIFFER'} shape={tuple(a.shape)} max_abs_diff={d:.3e}", flush=True)
    bad += not eq
sys.exit(1 if bad else 0)
PY
# feature row of v84i4250 clip 0 (1-clip manifest = Dev-78 header + its first clip)
"$PY" - "$C/dev78_manifest_20261018.json" "$VO/dev78_first1.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
m["clips"] = m["clips"][:1]
m["count"] = 1
json.dump(m, open(sys.argv[2], "w"), indent=1)
PY
export PYTHONPATH=$C/vendor:$C/src:$DR/pandwm_code/TATS_fvd
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PANDWM_PANO_BLEND_W=128 TORCH_HOME=$DR/.cache/torch OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
if [[ ! -e "$VO/feats_verify_v84i4250_clip0.npz" ]]; then
  "$PY" clip_feats.py --base-config "$C/cd_eval_frozen149_hpc3.json" --manifest "$VO/dev78_first1.json" \
    --dataset-stride 20 --out "$VO/feats_verify_v84i4250_clip0.npz" --arm-root "$VO/verify_v84i4250_s42" || fail=1
fi
"$PY" - "$VO/feats_verify_v84i4250_clip0.npz" "$OLDF/feats_v84i4250P_s42.npz" <<'PY' || fail=1
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
i = list(b["test_index"]).index(int(a["test_index"][0]))
di = float(np.abs(a["inc"][0] - b["inc"][i]).max()); dv = float(np.abs(a["i3d"][0] - b["i3d"][i]).max())
scale_i, scale_v = float(np.abs(b["inc"][i]).max()), float(np.abs(b["i3d"][i]).max())
ok = di <= 1e-2 and dv <= 1e-2
print(f"VERIFY_FEATS v84i4250 clip {int(a['test_index'][0])}: inc max abs diff {di:.3e} (scale {scale_i:.2f}), "
      f"i3d {dv:.3e} (scale {scale_v:.2f}) -> {'OK' if ok else 'FAIL'}", flush=True)
sys.exit(0 if ok else 1)
PY
if [[ $fail == 0 ]]; then
  if [[ ! -e "$EV/feats/feats_b0_s42.npz" ]]; then cp -p "$OLDF/feats_b0P_s42.npz" "$EV/feats/feats_b0_s42.npz"; fi
  if [[ ! -e "$EV/feats/feats_real_dev78_manifest_20261018.npz" ]]; then
    cp -p "$OLDF/feats_real_dev78_manifest_20261018.npz" "$EV/feats/feats_real_dev78_manifest_20261018.npz"
  fi
  md5sum "$EV/feats/feats_b0_s42.npz" "$OLDF/feats_b0P_s42.npz" "$EV/feats/feats_real_dev78_manifest_20261018.npz" \
    "$OLDF/feats_real_dev78_manifest_20261018.npz"
  echo "ABL_VERIFY PASS $(date -Is)"; exit 0
fi
echo "ABL_VERIFY FAIL $(date -Is)"; exit 1
