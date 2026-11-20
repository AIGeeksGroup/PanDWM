# Environment for gsplat-based GS jobs (gsplat 1.5.3, CUDA 12.8 toolkit for the JIT build, torch 2.7.1+cu128).
# GOLDEN=1 builds into the shared golden dir; otherwise each job copies the golden build into its own dir.
DR=/path/to/pandwm_root
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$DR/pandwm_gs_pkgs/bin:$PATH
# System gcc is too old for the torch 2.7 headers (needs >= 9): use gcc 12.5 as host compiler.
GCC=/data/apps/gcc/12.5
export PATH=$GCC/bin:$PATH CC=$GCC/bin/gcc CXX=$GCC/bin/g++ CUDAHOSTCXX=$GCC/bin/g++
export NVCC_PREPEND_FLAGS="-ccbin $GCC/bin/g++"
export LD_LIBRARY_PATH=$GCC/lib64:$GCC/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="9.0"
# wandb: no-op stub; the GS pipeline imports it at module level.
export PYTHONPATH=$DR/pandwm_gs_pkgs/site:$DR/pandwm_gs_pkgs/stubs${PYTHONPATH:+:$PYTHONPATH}
GOLD=$DR/.cache/pandwm_ext_golden_hpc3
if [ "${GOLDEN:-0}" = 1 ]; then
    export TORCH_EXTENSIONS_DIR=$GOLD
else
    EXT=/tmp/pandwm_ext_${SLURM_JOB_ID:-$$}; rm -rf $EXT; mkdir -p $EXT
    cp -a $GOLD/gsplat_cuda $EXT/ 2>/dev/null || true
    export TORCH_EXTENSIONS_DIR=$EXT
fi
