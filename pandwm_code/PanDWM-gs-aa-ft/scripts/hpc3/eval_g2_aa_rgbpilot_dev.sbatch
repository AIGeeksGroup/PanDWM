#!/usr/bin/env bash
#SBATCH --job-name=pandwm_g2_aa_dev
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log

# One-GPU, development-only RGB assessment of the isolated AA pilot.  It does
# not read common149 and it deliberately does not fabricate unavailable depth
# ground truth.  The primary repository and all prior runs remain untouched.
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM-gs-aa-ft
S=$H/pandwm_gs_aa_experiment
VENDOR=$H/pandwm_gs_holdout5_aa/vendor
G2_RUN=$S/runs/g2_mix75_250_aa_eps0to0p1_rgbpilot
G2_CFG=$P/configs/ctsd/pandwm/hpc3/gs_decoder_g2_aa_eps0to0p1_rgbpilot_250.json
DEV_PKL=$S/data_splits/val_dev_noncommon_scene7.pkl
RUN=$S/eval_dev_aa_rgbpilot_${SLURM_JOB_ID}
STAGE2_RUN=$RUN/stage2_run
STAGE2_SOURCE_CFG=$H/pandwm_gs_holdout5_aa/code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_h20.json

for file in \
  "$G2_RUN/checkpoints/250.pth" \
  "$G2_CFG" \
  "$DEV_PKL" \
  "$STAGE2_SOURCE_CFG" \
  "$H/pandwm_data/cd_10000_ema.pth"; do
  test -f "$file" || { echo "missing required file: $file" >&2; exit 2; }
done
for dir in \
  "$P" "$VENDOR" "$H/pandwm_data/rgb" "$H/pandwm_data/bbox" \
  "$H/pandwm_data/hdmap" "$H/pandwm_data/nuscenes_meta/v1.0-trainval"; do
  test -d "$dir" || { echo "missing required directory: $dir" >&2; exit 2; }
done
test ! -e "$RUN" || { echo "refusing to overwrite run: $RUN" >&2; exit 2; }
mkdir -p "$RUN/configs" "$RUN/torch_extensions" "$STAGE2_RUN/checkpoints"
ln -s "$H/pandwm_data/cd_10000_ema.pth" "$STAGE2_RUN/checkpoints/10000_ema.pth"

source /share/anaconda3/bin/activate pytorch271
module load gcc/12.5
export CC="$(command -v gcc)" CXX="$(command -v g++)"
export PYTHONPATH="$VENDOR:$P/src:$H/pandwm_code/TATS_fvd"
export PATH="$VENDOR/bin:$PATH"
export PANDWM_BBOX_DIR="$H/pandwm_data/bbox"
export PANDWM_HDMAP_DIR="$H/pandwm_data/hdmap"
export TORCH_EXTENSIONS_DIR="$RUN/torch_extensions"
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 500))

echo "6c0c2b51314a23aa3f65d7d9a452795385c7e43b60fc03d703ee73a0a9052fef  $G2_CFG" | sha256sum --check --strict
echo "8f5c6cae9d5723e0cda35649ea1ee73f763e0a1bc902eb7ed74c238560dc801f  $DEV_PKL" | sha256sum --check --strict

"$H/pandwm_venv/bin/python" - "$G2_CFG" "$STAGE2_SOURCE_CFG" "$RUN" "$H" "$DEV_PKL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

g2_source, stage2_source, run, home, dev_pkl = map(Path, sys.argv[1:])
g2 = json.loads(g2_source.read_text())
g2['pipeline']['pretrained_model_name_or_path'] = str(home / 'pandwm_data/stable-diffusion-3.5-medium')
g2['pipeline']['initial_decoder_checkpoint_path'] = None
g2['pipeline']['latent_domain_adaptation']['enabled'] = False
g2['training_dataset']['base_dataset']['pkl_path'] = str(dev_pkl)
g2['training_dataset']['base_dataset']['rgb_dir'] = str(home / 'pandwm_data/rgb')
g2['training_dataset']['base_dataset']['nuscenes_meta_dir'] = str(home / 'pandwm_data/nuscenes_meta/v1.0-trainval')
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
        .replace('/path/to/pandwm_root/ckpts/stable-diffusion-3.5-medium', str(home / 'pandwm_data/stable-diffusion-3.5-medium'))
        .replace('/path/to/pandwm_root/ckpts/cd_10000_ema.pth', str(home / 'pandwm_data/cd_10000_ema.pth'))
        .replace('/path/to/pandwm_root/data/val/rgb', str(home / 'pandwm_data/rgb'))
        .replace('/path/to/pandwm_root/data/bbox', str(home / 'pandwm_data/bbox'))
        .replace('/path/to/pandwm_root/data/hdmap', str(home / 'pandwm_data/hdmap'))
        .replace('/path/to/pandwm_root/data/nuscenes_meta', str(home / 'pandwm_data/nuscenes_meta'))
        .replace('/path/to/pandwm_root/data/pickle/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl', str(dev_pkl))
        .replace('/path/to/pandwm_root/code/TATS_fvd', str(home / 'pandwm_code/TATS_fvd')))
stage2 = rewrite(stage2)
(run / 'configs/stage2.json').write_text(json.dumps(stage2, indent=2) + '\n')

record = {
    'purpose': 'development-only AA-pilot RGB evaluation; no common149, no depth metrics',
    'aa_g2_config_sha256': hashlib.sha256(g2_source.read_bytes()).hexdigest(),
    'dev_pkl_sha256': hashlib.sha256(dev_pkl.read_bytes()).hexdigest(),
    'rgb_only': True,
}
(run / 'recipe.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
PY

"$H/pandwm_venv/bin/python" - <<'PY'
import inspect
import gsplat
assert 'eps2d' in inspect.signature(gsplat.rasterization).parameters
assert 'rasterize_mode' in inspect.signature(gsplat.rasterization).parameters
print('gsplat', gsplat.__version__, 'AA API verified')
PY

"$H/pandwm_venv/bin/python" -m dwm_pano.evaluate_gs_decode_compare \
  -c "$RUN/configs/g2.json" -r "$G2_RUN" -s 250 \
  --holdout-pkl "$DEV_PKL" \
  --stage2-config "$RUN/configs/stage2.json" \
  --stage2-run "$STAGE2_RUN" --stage2-step 10000 --stage2-ema \
  --gaussian-rgb-blend 0.0 --clip-limit 0 --rgb-only --out "$RUN/metrics"
