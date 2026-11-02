#!/usr/bin/env bash
#SBATCH --job-name=pandwm_g2_aa_c149
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

# Matched one-GPU common149 assessment of the final AA-compatible G2 head.
# This does not evaluate FID/FVD: it feeds the same frozen Stage-2 4-step
# latents through VAE and GS outputs and scores RGB reconstruction.  Inference
# uses 19 frames (the trained DiT context) but selects the exact 149 anchors
# from the canonical 35-frame common149 enumeration.  The validation package
# has no aligned dense-depth targets, so depth is explicitly disabled rather
# than silently substituting training-set depth.
# The original PanDWM checkout and all checkpoints are read-only.
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM-gs-aa-ft
S=$H/pandwm_gs_aa_experiment
VENDOR=$H/pandwm_gs_holdout5_aa/vendor
G2_RUN=$S/runs/g2_mix75_250_aa_eps0to0p1
G2_CFG=$P/configs/ctsd/pandwm/hpc3/gs_decoder_g2_aa_eps0to0p1_250.json
COMMON_PKL=$H/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl
STAGE2_CFG=$P/configs/ctsd/pandwm/h20/cd_eval_h20.json
RUN=$S/eval_common149_aa_${SLURM_JOB_ID}
STAGE2_RUN=$RUN/stage2_run
FINAL_G2_SHA=4929f9a0f621896d6c1becf89279f8159b1b9f30bc651e80c699a048e72763d5
STAGE2_SHA=beb842bc47996bb700d4f6055b9552660d2733922973db8b6379ea4811f11e6e

for file in \
  "$G2_RUN/checkpoints/250.pth" "$G2_CFG" "$COMMON_PKL" "$STAGE2_CFG" \
  "$H/pandwm_data/cd_10000_ema.pth" \
  "$P/src/dwm_pano/evaluate_gs_decode_compare.py"; do
  test -f "$file" || { echo "missing required file: $file" >&2; exit 2; }
done
for dir in \
  "$P" "$VENDOR" "$H/pandwm_data/rgb" "$H/pandwm_data/bbox" \
  "$H/pandwm_data/hdmap" "$H/pandwm_data/nuscenes_meta/v1.0-trainval"; do
  test -d "$dir" || { echo "missing required directory: $dir" >&2; exit 2; }
done
test ! -e "$RUN" || { echo "refusing to overwrite run: $RUN" >&2; exit 2; }
mkdir -p "$RUN/configs" "$RUN/torch_extensions" "$STAGE2_RUN/checkpoints"
ln -s "$H/pandwm_data/cd_10000_ema.pth" \
  "$STAGE2_RUN/checkpoints/10000_ema.pth"

echo "$FINAL_G2_SHA  $G2_RUN/checkpoints/250.pth" | sha256sum --check --strict
echo "$STAGE2_SHA  $H/pandwm_data/cd_10000_ema.pth" | sha256sum --check --strict

source /share/anaconda3/bin/activate pytorch271
module load gcc/12.5
export CC="$(command -v gcc)" CXX="$(command -v g++)"
export PYTHONPATH="$VENDOR:$P/src:$H/pandwm_code/TATS_fvd"
export PATH="$VENDOR/bin:$PATH"
export PANDWM_BBOX_DIR="$H/pandwm_data/bbox"
export PANDWM_HDMAP_DIR="$H/pandwm_data/hdmap"
export PANDWM_GS_RASTERIZE_MODE=antialiased
export PANDWM_G1_DECODE_BATCH=1
export TORCH_EXTENSIONS_DIR="$RUN/torch_extensions"
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 500))

"$H/pandwm_venv/bin/python" - "$G2_CFG" "$STAGE2_CFG" "$RUN" "$H" \
  "$COMMON_PKL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

g2_source, stage2_source, run, home, common_pkl = map(Path, sys.argv[1:])
g2 = json.loads(g2_source.read_text())
g2['pipeline']['pretrained_model_name_or_path'] = str(
    home / 'pandwm_data/stable-diffusion-3.5-medium')
g2['pipeline']['initial_decoder_checkpoint_path'] = None
g2['pipeline']['latent_domain_adaptation']['enabled'] = False
g2['pipeline']['training_config']['range_dir'] = str(
    home / 'pandwm_data/train/depth')
base = g2['training_dataset']['base_dataset']
base['pkl_path'] = str(common_pkl)
base['rgb_dir'] = str(home / 'pandwm_data/rgb')
base['nuscenes_meta_dir'] = str(
    home / 'pandwm_data/nuscenes_meta/v1.0-trainval')
base['split'] = 'val-common149'
(run / 'configs/g2.json').write_text(json.dumps(g2, indent=2) + '\n')

stage2 = json.loads(stage2_source.read_text())
stage2['global_state']['device_mesh']['mesh_shape'] = [1, 1]
def rewrite(value):
    if isinstance(value, dict):
        return {key: rewrite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite(item) for item in value]
    if not isinstance(value, str):
        return value
    return (value
        .replace('/path/to/pandwm_root/ckpts/stable-diffusion-3.5-medium',
                 str(home / 'pandwm_data/stable-diffusion-3.5-medium'))
        .replace('/path/to/pandwm_root/ckpts/cd_10000_ema.pth',
                 str(home / 'pandwm_data/cd_10000_ema.pth'))
        .replace('/path/to/pandwm_root/data/val/rgb', str(home / 'pandwm_data/rgb'))
        .replace('/path/to/pandwm_root/data/bbox', str(home / 'pandwm_data/bbox'))
        .replace('/path/to/pandwm_root/data/hdmap', str(home / 'pandwm_data/hdmap'))
        .replace('/path/to/pandwm_root/data/nuscenes_meta',
                 str(home / 'pandwm_data/nuscenes_meta'))
        .replace('/path/to/pandwm_root/data/pickle/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl',
                 str(common_pkl))
        .replace('/path/to/pandwm_root/code/TATS_fvd',
                 str(home / 'pandwm_code/TATS_fvd')))
stage2 = rewrite(stage2)
(run / 'configs/stage2.json').write_text(json.dumps(stage2, indent=2) + '\n')

record = {
    'purpose': 'final AA G2 common149 matched reconstruction evaluation',
    'protocol': '149 canonical 35-frame anchors at 10Hz/20s stride; each scored through a 19-frame 4-step inference window, 4 evenly-spaced RGB frames',
    'gaussian_checkpoint_sha256': hashlib.sha256(
        (run.parent / 'runs/g2_mix75_250_aa_eps0to0p1/checkpoints/250.pth').read_bytes()
    ).hexdigest(),
    'stage2_checkpoint_sha256': hashlib.sha256(
        (home / 'pandwm_data/cd_10000_ema.pth').read_bytes()).hexdigest(),
    'common_pkl_sha256': hashlib.sha256(common_pkl.read_bytes()).hexdigest(),
    'rasterize_mode': 'antialiased',
    'eps2d_schedule': '0 -> 0.1 over 200 steps during GS-only fine-tuning',
    'gaussian_rgb_blend': 0.0,
    'depth_evaluated': False,
    'depth_reason': 'common149 validation package lacks aligned dense-depth H5 targets',
}
(run / 'recipe.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
PY

"$H/pandwm_venv/bin/python" - <<'PY'
import inspect
import gsplat
assert 'eps2d' in inspect.signature(gsplat.rasterization).parameters
assert 'rasterize_mode' in inspect.signature(gsplat.rasterization).parameters
print('gsplat', gsplat.__version__, 'antialiased API verified')
PY

base=(
  "$H/pandwm_venv/bin/python" -m dwm_pano.evaluate_gs_decode_compare
  -c "$RUN/configs/g2.json" -r "$G2_RUN" -s 250
  --holdout-pkl "$COMMON_PKL"
  --stage2-config "$RUN/configs/stage2.json" --stage2-run "$STAGE2_RUN"
  --stage2-step 10000 --stage2-ema --gaussian-rgb-blend 0.0
  --sequence-length 19 --anchor-reference-sequence-length 35 \
  --expected-clip-count 149 --rgb-only
)

# The smoke uses the exact same common149-anchor/19-frame-inference path and
# fails fast before committing the full 149-clip run.  The full result remains
# the only report.
"${base[@]}" --clip-limit 1 --out "$RUN/smoke1"
"${base[@]}" --clip-limit 0 --out "$RUN/common149"
