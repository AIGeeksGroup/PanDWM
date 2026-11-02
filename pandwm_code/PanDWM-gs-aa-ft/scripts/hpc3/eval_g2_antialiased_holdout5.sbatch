#!/usr/bin/env bash
#SBATCH --job-name=pandwm_g2_aa
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

# Strict, matched G2 rasterizer A/B: same mix75 checkpoint, Stage-2 latent
# checkpoint and five fixed holdout clips as the prior classic result.
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
S=$H/pandwm_gs_holdout5_aa
RUN=$S/eval_antialiased_${SLURM_JOB_ID}
G2_RUN=$S/runs/gs_g2_ft_stage2mix75_250_offset2500
STAGE2_RUN=$S/stage2_run

for file in \
  "$G2_RUN/checkpoints/250.pth" \
  "$S/data/pickle/nuscenes_interp_12Hz_infos_holdout_p2.pkl" \
  "$H/pandwm_data/cd_10000_ema.pth"; do
  test -f "$file" || { echo "missing required asset: $file" >&2; exit 2; }
done
test ! -e "$RUN" || { echo "refusing to overwrite $RUN" >&2; exit 2; }
mkdir -p "$RUN/configs" "$RUN/torch_extensions" "$STAGE2_RUN/checkpoints"
stage2_link="$STAGE2_RUN/checkpoints/10000_ema.pth"
if [[ -e "$stage2_link" && ! -L "$stage2_link" ]]; then
  echo "refusing to replace non-link stage-2 checkpoint: $stage2_link" >&2
  exit 2
fi
ln -sfn "$H/pandwm_data/cd_10000_ema.pth" "$stage2_link"

source /share/anaconda3/bin/activate pytorch271
module load gcc/12.5
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export PYTHONPATH="$S/vendor:$P/src:$H/pandwm_code/TATS_fvd"
export PANDWM_GS_RASTERIZE_MODE=antialiased
export PANDWM_BBOX_DIR="$S/data/bbox"
export PANDWM_HDMAP_DIR="$S/data/hdmap"
export TORCH_EXTENSIONS_DIR="$RUN/torch_extensions"
export MAX_JOBS=8
export PATH="$S/vendor/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 500))

"$H/pandwm_venv/bin/python" - "$S" "$H" "$RUN" <<'PY'
import json
import sys
from pathlib import Path

stage, home, run = map(Path, sys.argv[1:])
with (stage / 'code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_g2_ft_stage2mix75_250.json').open() as f:
    g2 = json.load(f)
pipe = g2['pipeline']
pipe['pretrained_model_name_or_path'] = str(home / 'pandwm_data/stable-diffusion-3.5-medium')
pipe['initial_decoder_checkpoint_path'] = None
pipe['latent_domain_adaptation']['enabled'] = False
pipe['training_config']['range_dir'] = str(stage / 'data/train/depth')
base = g2['training_dataset']['base_dataset']
base['pkl_path'] = str(stage / 'data/pickle/nuscenes_interp_12Hz_infos_holdout_p2.pkl')
base['rgb_dir'] = str(stage / 'data/train/rgb')
base['nuscenes_meta_dir'] = str(home / 'pandwm_data/nuscenes_meta/v1.0-trainval')
with (run / 'configs/g2.json').open('w') as f:
    json.dump(g2, f, indent=2)

with (stage / 'code/PanDWM/configs/ctsd/pandwm/h20/cd_eval_h20.json').open() as f:
    stage2 = json.load(f)
stage2['global_state']['device_mesh']['mesh_shape'] = [1, 1]
def rewrite_paths(value):
    if isinstance(value, dict):
        return {key: rewrite_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_paths(item) for item in value]
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
                 str(home / 'pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl'))
        .replace('/path/to/pandwm_root/code/TATS_fvd',
                 str(home / 'pandwm_code/TATS_fvd')))
stage2 = rewrite_paths(stage2)
with (run / 'configs/stage2.json').open('w') as f:
    json.dump(stage2, f, indent=2)
PY

"$H/pandwm_venv/bin/python" - <<'PY'
import inspect
import gsplat
assert 'rasterize_mode' in inspect.signature(gsplat.rasterization).parameters
print('gsplat', gsplat.__version__, 'antialiased API verified')
PY

base=(
  "$H/pandwm_venv/bin/python" "$P/src/dwm_pano/evaluate_gs_decode_compare.py"
  -c "$RUN/configs/g2.json" -r "$G2_RUN" -s 250
  --holdout-pkl "$S/data/pickle/nuscenes_interp_12Hz_infos_holdout_p2.pkl"
  --stage2-config "$RUN/configs/stage2.json" --stage2-run "$STAGE2_RUN"
  --stage2-step 10000 --stage2-ema --gaussian-rgb-blend 0.0
)

"${base[@]}" --clip-limit 1 --out "$RUN/smoke1"
"${base[@]}" --clip-limit 0 --out "$RUN/holdout5"
