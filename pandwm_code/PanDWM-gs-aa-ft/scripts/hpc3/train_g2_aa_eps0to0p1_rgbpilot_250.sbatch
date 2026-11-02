#!/usr/bin/env bash
#SBATCH --job-name=pandwm_g2_aa_rgb
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log

# This job owns no file in the primary PanDWM checkout or its prior runs.
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM-gs-aa-ft
S=$H/pandwm_gs_aa_experiment
RUN=$S/runs/g2_mix75_250_aa_eps0to0p1_rgbpilot
CFG=$P/configs/ctsd/pandwm/hpc3/gs_decoder_g2_aa_eps0to0p1_rgbpilot_250.json
VENDOR=$H/pandwm_gs_holdout5_aa/vendor

test -d "$P" || { echo "missing isolated code: $P" >&2; exit 2; }
test -d "$VENDOR" || { echo "missing gsplat vendor tree: $VENDOR" >&2; exit 2; }
test -f "$CFG" || { echo "missing config: $CFG" >&2; exit 2; }
test ! -e "$RUN" || { echo "refusing to overwrite run: $RUN" >&2; exit 2; }
for file in \
  "$S/inputs/g2_mix75_250.pth" \
  "$S/inputs/nuscenes_interp_12Hz_infos_train_p2.pkl" \
  "$S/inputs/stage2_train_g2_seed0_offset2500_500/manifest.json" \
  "$S/data_splits/val_dev_noncommon_scene7.pkl" \
  "$H/pandwm_data/stable-diffusion-3.5-medium/vae/config.json" \
  "$H/pandwm_data/nuscenes_meta/v1.0-trainval/sample.json"; do
  test -f "$file" || { echo "missing required file: $file" >&2; exit 2; }
done
test -d "$H/pandwm_data/train/rgb" || { echo "missing train RGB" >&2; exit 2; }

echo "8d201fbd83275ddd9b3aa0d20d1b3164dadfff972c9ffea685f68934f697cf7b  $S/inputs/g2_mix75_250.pth" | sha256sum --check --strict
echo "7ecb988e43440dbb1b1ea4914e2b388040daa5202f97317490f123828fe68678  $S/inputs/nuscenes_interp_12Hz_infos_train_p2.pkl" | sha256sum --check --strict

source /share/anaconda3/bin/activate pytorch271
module load gcc/12.5
export CC="$(command -v gcc)" CXX="$(command -v g++)"
export PYTHONPATH="$VENDOR:$P/src:$H/pandwm_code/TATS_fvd"
unset PANDWM_GS_RASTERIZE_MODE || true
export PATH="$VENDOR/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$S/torch_extensions"
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 500))

mkdir -p "$RUN" "$TORCH_EXTENSIONS_DIR"

"$H/pandwm_venv/bin/python" - "$CFG" "$S" "$P" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

cfg_path, root, code = map(Path, sys.argv[1:])
cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
train = cfg['pipeline']['training_config']
adapt = cfg['pipeline']['latent_domain_adaptation']
required = {
    'antialiased': train['rasterize_mode'] == 'antialiased',
    'eps schedule': (train['aa_eps2d_start'], train['aa_eps2d_end'], train['aa_eps2d_ramp_steps']) == (0.0, 0.1, 200),
    '100% cached stage2 latent': adapt['probability'] == 1.0,
    'generated geometry off': adapt['supervise_geometry'] is False,
    'RGB-only pilot losses': train['weight_seg'] == 0.0 and train['weight_render_depth'] == 0.0 and train['weight_range_l1'] == 0.0,
    'cross losses off': train['weight_cross_rgb_l1'] == 0.0 and train['weight_cross_alpha_coverage'] == 0.0,
}
failed = [name for name, valid in required.items() if not valid]
cache = json.loads((root / 'inputs/stage2_train_g2_seed0_offset2500_500/manifest.json').read_text())
if cache['completed_steps'] < 250 or cache['start_offset'] != 2500:
    failed.append('Stage-2 cache does not cover 250 steps from offset 2500')
if failed:
    raise RuntimeError('AA recipe guard failed: ' + '; '.join(failed))
record = {
    'recipe': 'Development-only RGB/LPIPS: G2 mix75-250 -> antialiased GS-only fine tune; eps2d 0..0.1 over steps 0..200',
    'config_sha256': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
    'pipeline_sha256': hashlib.sha256((code / 'src/dwm_pano/pipelines/ctsd_pano_gs_decoder.py').read_bytes()).hexdigest(),
    'gs_checkpoint_sha256': hashlib.sha256((root / 'inputs/g2_mix75_250.pth').read_bytes()).hexdigest(),
    'train_pkl_sha256': hashlib.sha256((root / 'inputs/nuscenes_interp_12Hz_infos_train_p2.pkl').read_bytes()).hexdigest(),
    'dev_split': json.loads((root / 'data_splits/val_dev_noncommon_scene7.json').read_text()),
    'caveat': 'RGB-only development pilot: depth, range and segmentation objectives are zero pending full data staging; never report as the matched full AA arm.',
}
(root / 'runs/g2_mix75_250_aa_eps0to0p1_rgbpilot/recipe.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, indent=2), flush=True)
PY

"$H/pandwm_venv/bin/python" -m dwm_pano.train \
  -c "$CFG" -o "$RUN" \
  --log-steps 10 --preview-steps 250 --checkpointing-steps 250 \
  --evaluation-steps 0 --keep-last-ckpts 0 --max-steps 250 \
  --initial-data-offset 2500
