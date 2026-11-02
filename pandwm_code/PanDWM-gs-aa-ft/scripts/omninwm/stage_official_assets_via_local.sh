#!/bin/bash
# Stage public official OmniNWM assets through this workstation because the
# HPC compute network has no outbound HTTPS route. Every LFS file is verified
# before upload and again on HPC, then removed from the bounded local stage.
set -euo pipefail

STAGE=/path/to/home/.cache/pandwm_official_stage/omninwm
REMOTE_ROOT=/path/to/pandwm_root/external_models/omninwm
mkdir -p "$STAGE"

export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_DISABLE_XET=1

transfer_asset() {
  local repo=$1
  local filename=$2
  local expected_sha=$3
  local remote_relative=$4
  local local_root=$STAGE/${repo//\//__}
  local local_file=$local_root/$filename
  local remote_file=$REMOTE_ROOT/$remote_relative
  local remote_dir=${remote_file%/*}

  mkdir -p "$local_root"
  echo "download repo=$repo file=$filename"
  hf download "$repo" "$filename" --local-dir "$local_root"
  test -f "$local_file"
  echo "$expected_sha  $local_file" | sha256sum --check --strict

  ssh -o BatchMode=yes hpc "mkdir -p '$remote_dir'"
  rsync -a --partial --info=progress2 "$local_file" "hpc:$remote_file"
  ssh -o BatchMode=yes hpc \
    "echo '$expected_sha  $remote_file' | sha256sum --check --strict"

  rm -f "$local_file"
  echo "staged_and_verified remote=$remote_file sha256=$expected_sha"
}

transfer_asset Arlolo0/OmniNWM \
  ckpt/dit/model/model-00001.safetensors \
  bfeb95bde39af017088abcfbbda406154b5157d64ef3798989a7ce4c7d4d06e1 \
  release/ckpt/dit/model/model-00001.safetensors
transfer_asset Arlolo0/OmniNWM \
  ckpt/dit/model/model-00002.safetensors \
  c0747c4774d34fb5607915b4f4615fac2a5922444ced408e92dedae76d1e5c56 \
  release/ckpt/dit/model/model-00002.safetensors
transfer_asset Arlolo0/OmniNWM \
  ckpt/dit/model/model-00003.safetensors \
  a7e8129129c717d3918a55111e8aa67462c2c89f8ad4fa81f7b72e9eb2a10fe8 \
  release/ckpt/dit/model/model-00003.safetensors
transfer_asset Arlolo0/OmniNWM \
  ckpt/dit/model/model-00004.safetensors \
  6b2bb0f0da0f1fb5a6fbccda9d34f20c72807356bc6a3ebba30a263fdd1247fb \
  release/ckpt/dit/model/model-00004.safetensors
transfer_asset Arlolo0/OmniNWM \
  ckpt/dit/model/model-00005.safetensors \
  8b69b0807c0776693d638ce5b2e33603d2ca693584fa4a2c70b064966b91bbd8 \
  release/ckpt/dit/model/model-00005.safetensors
transfer_asset Arlolo0/OmniNWM \
  ckpt/dit/model/model-00006.safetensors \
  878845ef1343675e231a30efa1541a1d24207f5e49810547f7b6e9b63e33e420 \
  release/ckpt/dit/model/model-00006.safetensors
transfer_asset Arlolo0/OmniNWM \
  ckpt/dit/model/model.safetensors.index.json \
  1e0168b50b4768b09e080d6e1784e424df10656e80139e4f5b9b98815d9485a7 \
  release/ckpt/dit/model/model.safetensors.index.json
transfer_asset Arlolo0/OmniNWM \
  pickle/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl \
  c419d256a7409b64d626ec189de0e927efcc18cbb631f69c4e353a2ea080c548 \
  release/pickle/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl
transfer_asset hpcai-tech/Open-Sora-v2 \
  hunyuan_vae.safetensors \
  003e7c25ea79298e4c41c0b3e77ec2c977a7f810d0ab84847e4e5c1c3e9fe9c5 \
  hunyuan_vae.safetensors

echo "omninwm_official_assets_staged_via_local"
