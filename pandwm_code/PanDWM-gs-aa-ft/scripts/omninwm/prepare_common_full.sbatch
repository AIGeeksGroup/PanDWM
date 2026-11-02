#!/bin/bash
#SBATCH --job-name=omni_data_full
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
OUT=$H/omninwm_runs/common149_adapter

test ! -e "$OUT"
export PYTHONPATH=$P/src
export PYTHONUNBUFFERED=1
"$H/pandwm_venv/bin/python" \
  "$P/scripts/omninwm/prepare_common12hz_sixview.py" \
  --common-pkl "$H/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl" \
  --pano-rgb-dir "$H/pandwm_data/rgb" \
  --out "$OUT" \
  --device cuda

test "$(find "$OUT/images" -type f -name '*.jpg' | wc -l)" -eq 29502
"$H/pandwm_venv/bin/python" - "$OUT" <<'PY'
import json,pickle,sys
from pathlib import Path
root=Path(sys.argv[1])
manifest=json.loads((root/'manifest.json').read_text())
payload=pickle.load((root/'omninwm_common149_12hz.pkl').open('rb'))
assert manifest['clips']==149
assert len(payload['infos'])==149*33
assert len(payload['scene_tokens'])==149
assert all(len(scene)==33 for scene in payload['scene_tokens'])
for record in manifest['records']:
    indices=record['generated_indices_scored']
    assert len(indices)==16 and indices==sorted(indices) and indices[0]>0
print('adapter_full_ok')
PY
du -sh "$OUT"
