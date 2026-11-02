#!/usr/bin/env bash
# Build FA3 entirely on compute-node /local (6.8TB, NO WekaFS quota). /scratch
# has a per-user quota that the .o/.so intermediates + setup.py's nvcc-toolchain
# tarball extraction blow past (Errno 122). Redirect HOME/TMP/cache to /local.
set -e
module load CUDA/12.6.0
export CUDA_HOME=/cvmfs/software.asc.ac.at/versions/2025.06/software/linux/x86_64/amd/zen4/software/CUDA/12.6.0
PYBIN=/path/to/pandwm_root/workspace/miniconda3/envs/opendwm/bin
export PATH=$CUDA_HOME/bin:$PYBIN:$PATH

ROOT=/local/fa3
export HOME=$ROOT/home TMPDIR=$ROOT/tmp XDG_CACHE_HOME=$ROOT/cache PIP_CACHE_DIR=$ROOT/pipcache
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"

SRC=/path/to/pandwm_root/workspace/OpenDWM/flash-attention
DST=$ROOT/flash-attention
if [ ! -d "$DST/hopper" ]; then
  echo "=== copying source to /local ==="
  cp -r "$SRC" "$DST"
fi
FA=$DST/hopper
cd "$FA"

export FLASH_ATTENTION_FORCE_BUILD=TRUE
# backward ENABLED this time (training needs it); keep everything else minimal.
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM128=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE FLASH_ATTENTION_DISABLE_VARLEN=TRUE
export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE FLASH_ATTENTION_DISABLE_LOCAL=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
export MAX_JOBS=24 NVCC_THREADS=2

echo "=== nvcc: $(nvcc --version | grep release) ; building on /local ==="
"$PYBIN/python" setup.py build_ext --inplace -j 24 2>&1
RC=$?
echo "=== build_ext exit=$RC ; .so files: ==="
find "$FA" -name "*.so" 2>/dev/null
# Place built .so into an importable package dir (build_ext --inplace can't
# create it). Then stage the importable bits to /scratch (shared, small) so
# BOTH profile nodes (013+014) can import it — /local is node-013-only.
mkdir -p "$FA/flash_attn_3"
cp -f "$FA"/build/lib.*/flash_attn_3/_C*.so "$FA/flash_attn_3/" 2>/dev/null
touch "$FA/flash_attn_3/__init__.py"
SHARED=/path/to/pandwm_root/workspace/OpenDWM/flash-attention/hopper
mkdir -p "$SHARED/flash_attn_3"
cp -f "$FA/flash_attn_3/"*.so "$SHARED/flash_attn_3/"
touch "$SHARED/flash_attn_3/__init__.py"
echo "=== staged .so to /scratch: ==="; ls -la "$SHARED/flash_attn_3/"

echo "=== verify import (from /scratch shared) ==="
PYTHONPATH="$SHARED:$PYTHONPATH" "$PYBIN/python" - <<'PY'
import torch
try:
    import flash_attn_interface as fa3
    print("OK flash_attn_interface; funcs:",
          [x for x in dir(fa3) if 'func' in x.lower()][:8])
except Exception as e:
    import traceback; traceback.print_exc()
    print("IMPORT FAILED:", repr(e))
PY
