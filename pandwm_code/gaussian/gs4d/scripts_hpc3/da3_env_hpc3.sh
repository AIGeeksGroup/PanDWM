# Depth Anything 3 runtime, sourced after the common env script.
# Code: pandwm_code/third_party/depth_anything_3/repo (Depth-Anything-3, Apache-2.0).
# Weights: weights/DA3-LARGE-1.1 (CC BY-NC 4.0) and weights/DA3METRIC-LARGE (Apache-2.0).
# Extra pure-python packages (omegaconf, antlr4, addict) are installed in pandwm_da3_pkgs/site only.
DA3_HOME=$DR/pandwm_code/third_party/depth_anything_3
DA3_SRC=$DA3_HOME/repo/src
DA3_ANYVIEW=$DA3_HOME/weights/DA3-LARGE-1.1
DA3_METRIC=$DA3_HOME/weights/DA3METRIC-LARGE
export PYTHONPATH=$PYTHONPATH:$DR/pandwm_da3_pkgs/site
