#!/bin/bash
#SBATCH --job-name=omni_assets
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail

H=/path/to/pandwm_root
ROOT=$H/external_models/omninwm
PY=$H/vista_venv/bin/python
mkdir -p "$ROOT"

export HF_HOME=$ROOT/.hf_cache
export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=600
export PYTHONUNBUFFERED=1

"$PY" -u - <<'PY'
import time
from huggingface_hub import hf_hub_download, snapshot_download

root = "/path/to/pandwm_root/external_models/omninwm"


def retry(label, fn):
    for attempt in range(1, 21):
        try:
            print(f"{label}: attempt {attempt}", flush=True)
            return fn()
        except Exception as exc:
            print(
                f"{label}: {type(exc).__name__}: {exc}; retrying",
                flush=True,
            )
            if attempt == 20:
                raise
            time.sleep(min(60, attempt * 5))


retry(
    "OmniNWM DiT and validation pickle",
    lambda: snapshot_download(
        repo_id="Arlolo0/OmniNWM",
        local_dir=root + "/release",
        local_dir_use_symlinks=True,
        allow_patterns=[
            "ckpt/dit/model/*",
            "pickle/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl",
        ],
        resume_download=True,
        max_workers=6,
    ),
)
retry(
    "Open-Sora-v2 Hunyuan VAE",
    lambda: hf_hub_download(
        repo_id="hpcai-tech/Open-Sora-v2",
        filename="hunyuan_vae.safetensors",
        local_dir=root,
        local_dir_use_symlinks=True,
        resume_download=True,
    ),
)
print("download_complete", flush=True)
PY

"$PY" -u - <<'PY'
import hashlib
from pathlib import Path

root = Path("/path/to/pandwm_root/external_models/omninwm")
expected = {
    "release/ckpt/dit/model/model-00001.safetensors": "bfeb95bde39af017088abcfbbda406154b5157d64ef3798989a7ce4c7d4d06e1",
    "release/ckpt/dit/model/model-00002.safetensors": "c0747c4774d34fb5607915b4f4615fac2a5922444ced408e92dedae76d1e5c56",
    "release/ckpt/dit/model/model-00003.safetensors": "a7e8129129c717d3918a55111e8aa67462c2c89f8ad4fa81f7b72e9eb2a10fe8",
    "release/ckpt/dit/model/model-00004.safetensors": "6b2bb0f0da0f1fb5a6fbccda9d34f20c72807356bc6a3ebba30a263fdd1247fb",
    "release/ckpt/dit/model/model-00005.safetensors": "8b69b0807c0776693d638ce5b2e33603d2ca693584fa4a2c70b064966b91bbd8",
    "release/ckpt/dit/model/model-00006.safetensors": "878845ef1343675e231a30efa1541a1d24207f5e49810547f7b6e9b63e33e420",
    "release/pickle/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl": "c419d256a7409b64d626ec189de0e927efcc18cbb631f69c4e353a2ea080c548",
    "hunyuan_vae.safetensors": "003e7c25ea79298e4c41c0b3e77ec2c977a7f810d0ab84847e4e5c1c3e9fe9c5",
}
for relative, wanted in expected.items():
    path = root / relative
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    got = digest.hexdigest()
    if got != wanted:
        raise RuntimeError(f"SHA256 mismatch for {path}: {got} != {wanted}")
    print(f"verified {relative} {got}", flush=True)
PY
echo "omninwm_official_assets_verified"
