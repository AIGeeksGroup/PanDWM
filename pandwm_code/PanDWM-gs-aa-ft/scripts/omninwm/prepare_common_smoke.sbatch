#!/bin/bash
#SBATCH --job-name=omni_data_smoke
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --exclude=ACD1-28
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
P=$H/pandwm_code/PanDWM
OUT=$H/omninwm_runs/common149_adapter_smoke

test ! -e "$OUT"
export PYTHONPATH=$P/src
export PYTHONUNBUFFERED=1
"$H/pandwm_venv/bin/python" \
  "$P/scripts/omninwm/prepare_common12hz_sixview.py" \
  --common-pkl "$H/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl" \
  --pano-rgb-dir "$H/pandwm_data/rgb" \
  --out "$OUT" \
  --max-clips 1 \
  --device cuda

test "$(find "$OUT/images" -type f -name '*.jpg' | wc -l)" -eq 198
"$H/pandwm_venv/bin/python" - "$OUT" <<'PY'
import json,pickle,sys
from pathlib import Path
root=Path(sys.argv[1])
manifest=json.loads((root/'manifest.json').read_text())
payload=pickle.load((root/'omninwm_common149_12hz.pkl').open('rb'))
assert manifest['clips']==1
assert len(payload['infos'])==33
assert len(payload['scene_tokens'])==1
assert len(payload['scene_tokens'][0])==33
indices=manifest['records'][0]['generated_indices_scored']
assert len(indices)==16 and indices==sorted(indices) and indices[0]>0
print('adapter_smoke_ok', indices)
PY
du -sh "$OUT"
