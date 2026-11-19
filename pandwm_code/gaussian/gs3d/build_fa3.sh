#!/usr/bin/env bash
# Minimal FA3 build: fp16 / hdim64 / forward / sm90 only. Everything else
# disabled to cut compile from 1-2h to ~minutes. Matches this model's spatial
# attention (flash_fwd_kernel<64,...> fp16, non-causal).
set -e
module load CUDA/12.6.0
export CUDA_HOME=/cvmfs/software.asc.ac.at/versions/2025.06/software/linux/x86_64/amd/zen4/software/CUDA/12.6.0
PYBIN=/path/to/pandwm_root/workspace/miniconda3/envs/opendwm/bin
export PATH=$CUDA_HOME/bin:$PYBIN:$PATH

FAROOT=/path/to/pandwm_root/workspace/OpenDWM/flash-attention
FA=$FAROOT/hopper
# Redirect all temp/cache to /scratch (default /tmp,$HOME are quota'd -> Errno 122).
export TMPDIR=$FAROOT/.btmp PIP_CACHE_DIR=$FAROOT/.pipcache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
# Shrink the source tree: drop unused ROCm submodule + git history.
rm -rf "$FAROOT/third_party/aiter" "$FAROOT/.git" 2>/dev/null || true
cd "$FA"

# Build reducers
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM128=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
# keep HDIM64 + FP16 + CLUSTER (cluster cheap, keep default)

export MAX_JOBS=24
export NVCC_THREADS=2

echo "=== nvcc: $(nvcc --version | grep release) ==="
echo "=== building FA3 (fp16/hdim64/fwd/sm90) in-place (no pip copy) ==="
# build_ext --inplace compiles flash_attn_3/_C*.so directly in the source tree,
# bypassing pip's full-tree copy that hit the tmp quota.
"$PYBIN/python" setup.py build_ext --inplace -j 24 2>&1
echo "=== build exit $? ; verifying import (hopper on PYTHONPATH) ==="
PYTHONPATH="$FA:$PYTHONPATH" "$PYBIN/python" - <<'PY'
import torch
try:
    import flash_attn_interface as fa3
    fns = [x for x in dir(fa3) if 'func' in x.lower()]
    print("OK import flash_attn_interface; funcs:", fns[:8])
except Exception as e:
    import traceback; traceback.print_exc()
    print("IMPORT FAILED:", repr(e))
PY
