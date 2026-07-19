#!/usr/bin/env bash
set -euo pipefail

# Storage box target.
STORAGE_BOX_HOST="u618234@u618234.your-storagebox.de"
STORAGE_BOX_PORT="23"
REMOTE_BASE_DIR="opendwm"
REMOTE_DATASETS_DIR="${REMOTE_BASE_DIR}/datasets"
REMOTE_PRETRAINED_DIR="${REMOTE_BASE_DIR}/pretrained"
REMOTE_CHECKPOINTS_DIR="${REMOTE_BASE_DIR}/checkpoints"

# Local staging for tars (node-local disk).
TAR_STAGING_DIR="/local/${USER}/opendwm_backup"

# Dataset source parents (tar members live directly under these).
NUSCENES_360_DIR="/data/fs201130/jn20658/tenants/clientB/nuscenes-360"
OPENDWM_DATA_DIR="/scratch/fs201130/jn20658/tenants/clientB/workspace/OpenDWM/data"
NUSCENES_META_PARENT_DIR="/data/fs201130/jn20658/tenants/clientB/nuscenes"

# Pretrained source parents and files.
OPENDWM_CKPTS_DIR="/data/fs201130/jn20658/tenants/clientB/OpenDWM/ckpts"
I3D_FVD_CHECKPOINT="/scratch/fs201130/jn20658/tenants/clientB/workspace/OpenDWM/externals/TATS/tats/fvd/i3d_pretrained_400.pt"

# Stage checkpoint source files.
STAGE0_BIDIRECTIONAL_TEACHER_CHECKPOINT="/data/fs201130/jn20658/tenants/clientB/OpenDWM/runs/train_ctsd_unimlvg_stage3_tirda_bidirectional_nwa_20260504_192921/checkpoints/20000.pth"
STAGE1_CAUSAL_AR_CHECKPOINT="/data/fs201130/jn20658/tenants/clientB/OpenDWM/runs/train_ctsd_unimlvg_stage3_tirda_causal_nwa_20260501_182010/checkpoints/20000.pth"
STAGE2_CONSISTENCY_DISTILLATION_CHECKPOINT="/data/fs201130/jn20658/tenants/clientB/OpenDWM/runs/ctsd_unimlvg_stage3_tirda_causal_nwa_consistency_distillation_20260521_023203/checkpoints/10000_ema.pth"

# rsync transport and flags.
RSYNC_SSH_COMMAND="ssh -p ${STORAGE_BOX_PORT}"
RSYNC_FLAGS="-avhP --partial"

mkdir -p "${TAR_STAGING_DIR}"
ssh -p "${STORAGE_BOX_PORT}" "${STORAGE_BOX_HOST}" "mkdir -p ${REMOTE_DATASETS_DIR} ${REMOTE_PRETRAINED_DIR} ${REMOTE_CHECKPOINTS_DIR}"

# Dataset: nuScenes-360 (train + bbox + hdmap; seg excluded, unused by configs).
tar -cf "${TAR_STAGING_DIR}/nuscenes-360.tar" -C "${NUSCENES_360_DIR}" train bbox hdmap
rsync ${RSYNC_FLAGS} -e "${RSYNC_SSH_COMMAND}" "${TAR_STAGING_DIR}/nuscenes-360.tar" "${STORAGE_BOX_HOST}:${REMOTE_DATASETS_DIR}/nuscenes-360.tar"

# Dataset: validation RGB.
tar -cf "${TAR_STAGING_DIR}/val_rgb.tar" -C "${OPENDWM_DATA_DIR}" val_rgb
rsync ${RSYNC_FLAGS} -e "${RSYNC_SSH_COMMAND}" "${TAR_STAGING_DIR}/val_rgb.tar" "${STORAGE_BOX_HOST}:${REMOTE_DATASETS_DIR}/val_rgb.tar"

# Dataset: nuScenes metadata.
tar -cf "${TAR_STAGING_DIR}/v1.0-trainval.tar" -C "${NUSCENES_META_PARENT_DIR}" v1.0-trainval
rsync ${RSYNC_FLAGS} -e "${RSYNC_SSH_COMMAND}" "${TAR_STAGING_DIR}/v1.0-trainval.tar" "${STORAGE_BOX_HOST}:${REMOTE_DATASETS_DIR}/v1.0-trainval.tar"

# Dataset: index / caption pickles.
tar -cf "${TAR_STAGING_DIR}/pickle.tar" -C "${OPENDWM_DATA_DIR}" pickle
rsync ${RSYNC_FLAGS} -e "${RSYNC_SSH_COMMAND}" "${TAR_STAGING_DIR}/pickle.tar" "${STORAGE_BOX_HOST}:${REMOTE_DATASETS_DIR}/pickle.tar"

# Pretrained: SD3.5-medium base.
tar -cf "${TAR_STAGING_DIR}/stable-diffusion-3.5-medium.tar" -C "${OPENDWM_CKPTS_DIR}" stable-diffusion-3.5-medium
rsync ${RSYNC_FLAGS} -e "${RSYNC_SSH_COMMAND}" "${TAR_STAGING_DIR}/stable-diffusion-3.5-medium.tar" "${STORAGE_BOX_HOST}:${REMOTE_PRETRAINED_DIR}/stable-diffusion-3.5-medium.tar"

# Pretrained: I3D network for FVD.
rsync ${RSYNC_FLAGS} -e "${RSYNC_SSH_COMMAND}" "${I3D_FVD_CHECKPOINT}" "${STORAGE_BOX_HOST}:${REMOTE_PRETRAINED_DIR}/i3d_pretrained_400.pt"

# Checkpoint: Stage 0 bidirectional teacher (renamed; all stages share name 20000.pth).
rsync ${RSYNC_FLAGS} -e "${RSYNC_SSH_COMMAND}" "${STAGE0_BIDIRECTIONAL_TEACHER_CHECKPOINT}" "${STORAGE_BOX_HOST}:${REMOTE_CHECKPOINTS_DIR}/stage0_bidirectional_nwa_20000.pth"

# Checkpoint: Stage 1 causal AR.
rsync ${RSYNC_FLAGS} -e "${RSYNC_SSH_COMMAND}" "${STAGE1_CAUSAL_AR_CHECKPOINT}" "${STORAGE_BOX_HOST}:${REMOTE_CHECKPOINTS_DIR}/stage1_causal_nwa_20000.pth"

# Checkpoint: Stage 2 consistency distillation.
rsync ${RSYNC_FLAGS} -e "${RSYNC_SSH_COMMAND}" "${STAGE2_CONSISTENCY_DISTILLATION_CHECKPOINT}" "${STORAGE_BOX_HOST}:${REMOTE_CHECKPOINTS_DIR}/stage2_cd_10000_ema.pth"

echo "OpenDWM backup complete"
