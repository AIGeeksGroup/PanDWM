#!/usr/bin/env bash
#SBATCH --job-name=pandwm_gs_aa_vid
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

# Produce a qualitative before/after GS anti-aliasing video on one GPU.  This
# reads checkpoints only and writes to the isolated AA experiment directory.
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM-gs-aa-ft
S=$H/pandwm_gs_aa_experiment
VENDOR=$H/pandwm_gs_holdout5_aa/vendor
G2_CFG=$P/configs/ctsd/pandwm/hpc3/gs_decoder_g2_aa_eps0to0p1_250.json
STAGE2_CFG=$P/configs/ctsd/pandwm/h20/cd_eval_h20.json
COMMON_PKL=$H/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl
CLASSIC=$S/inputs/g2_mix75_250.pth
AA=$S/runs/g2_mix75_250_aa_eps0to0p1/checkpoints/250.pth
OUT=$S/qualitative_gs_aa_ab_${SLURM_JOB_ID}
STAGE2_RUN=$OUT/stage2_run

for file in \
  "$G2_CFG" "$STAGE2_CFG" "$COMMON_PKL" "$CLASSIC" "$AA" \
  "$H/pandwm_data/cd_10000_ema.pth" \
  "$P/src/dwm_pano/export_gs_aa_ab_video.py"; do
  test -f "$file" || { echo "missing required file: $file" >&2; exit 2; }
done
for dir in \
  "$P" "$VENDOR" "$H/pandwm_data/rgb" "$H/pandwm_data/bbox" \
  "$H/pandwm_data/hdmap" "$H/pandwm_data/nuscenes_meta/v1.0-trainval"; do
  test -d "$dir" || { echo "missing required directory: $dir" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 2; }
mkdir -p "$OUT/configs" "$OUT/torch_extensions" "$STAGE2_RUN/checkpoints"
ln -s "$H/pandwm_data/cd_10000_ema.pth" \
  "$STAGE2_RUN/checkpoints/10000_ema.pth"

source /share/anaconda3/bin/activate pytorch271
module load gcc/12.5
export CC="$(command -v gcc)" CXX="$(command -v g++)"
export PYTHONPATH="$VENDOR:$P/src:$H/pandwm_code/TATS_fvd"
export PATH="$VENDOR/bin:$PATH"
export PANDWM_BBOX_DIR="$H/pandwm_data/bbox"
export PANDWM_HDMAP_DIR="$H/pandwm_data/hdmap"
export PANDWM_G1_DECODE_BATCH=1
export TORCH_EXTENSIONS_DIR="$OUT/torch_extensions"
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 500))

"$H/pandwm_venv/bin/python" - "$G2_CFG" "$STAGE2_CFG" "$OUT" "$H" \
  "$COMMON_PKL" <<'PY'
import json
import sys
from pathlib import Path

g2_source, stage2_source, out, home, common_pkl = map(Path, sys.argv[1:])
g2 = json.loads(g2_source.read_text())
g2['pipeline']['pretrained_model_name_or_path'] = str(
    home / 'pandwm_data/stable-diffusion-3.5-medium')
g2['pipeline']['initial_decoder_checkpoint_path'] = None
g2['pipeline']['latent_domain_adaptation']['enabled'] = False
base = g2['training_dataset']['base_dataset']
base['pkl_path'] = str(common_pkl)
base['rgb_dir'] = str(home / 'pandwm_data/rgb')
base['nuscenes_meta_dir'] = str(
    home / 'pandwm_data/nuscenes_meta/v1.0-trainval')
base['split'] = 'common149-video'
(out / 'configs/g2.json').write_text(json.dumps(g2, indent=2) + '\n')

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
(out / 'configs/stage2.json').write_text(
    json.dumps(rewrite(stage2), indent=2) + '\n')
PY

"$H/pandwm_venv/bin/python" -m dwm_pano.export_gs_aa_ab_video \
  -c "$OUT/configs/g2.json" --data-pkl "$COMMON_PKL" \
  --stage2-config "$OUT/configs/stage2.json" --stage2-run "$STAGE2_RUN" \
  --stage2-step 10000 --stage2-ema --classic-checkpoint "$CLASSIC" \
  --aa-checkpoint "$AA" --clip-index 0 --out "$OUT/video"

