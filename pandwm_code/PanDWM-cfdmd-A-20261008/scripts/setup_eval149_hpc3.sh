#!/usr/bin/env bash
# 2026-10-11: build an HPC3 copy of the HPC1 frozen 149-clip evaluation tree, so the v8.2 milestones can be evaluated on
# HPC3 with the same code (the user chose this after the laptop moved to a phone hotspot: the HPC3 -> laptop -> HPC1
# relay would cost ~27 GB of mobile data per checkpoint).  Run on the HPC3 login node:
#   bash scripts/setup_eval149_hpc3.sh <overlay dir>
# <overlay dir> holds the HPC1 files (eval_v7/hpc1_eval_code/ plus eval_v7/realism_arms.py): run_eval149_videos.py
# compute_eval149_metrics.py lowmem_launch.py run_long_rollout.py run_history_probe.py evaluate_4step.py
# cd_eval_hpc3.json realism_arms.py md5_hpc1.txt.
# src/ and vendor/ are copied from HPC3's PanDWM_long_horizon_20260928; 180 of its 181 .py/.pt files are identical to
# HPC1's, the exception src/dwm_pano/evaluate_4step.py is replaced by HPC1's (it only adds a history guidance that is
# off unless PANDWM_HISTORY_GUIDANCE is set).  The config is HPC1's with /hpc/home/<user> -> /path/to/pandwm_root
# (the only differences from HPC3's own cd_eval_hpc3.json are those paths and memory_efficient_batch 4 instead of 12).
# Every file is then checked against HPC1's md5 list.  Refuses if the target exists.
set -euo pipefail
OV=$(cd "${1:?overlay dir}" && pwd)
DR=/path/to/pandwm_root
SRC=$DR/pandwm_code/PanDWM_long_horizon_20260928
C=$DR/pandwm_code/PanDWM_eval149_20261011
[ -e "$C" ] && { echo "EXISTS $C"; exit 1; }
for f in run_eval149_videos.py compute_eval149_metrics.py lowmem_launch.py run_long_rollout.py run_history_probe.py \
         evaluate_4step.py cd_eval_hpc3.json realism_arms.py md5_hpc1.txt; do
  [ -f "$OV/$f" ] || { echo "MISSING_OVERLAY $f"; exit 1; }
done
mkdir -p "$C"
cp -a "$SRC/src" "$SRC/vendor" "$C/"
find "$C" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$OV/evaluate_4step.py" "$C/src/dwm_pano/evaluate_4step.py"
for f in run_eval149_videos.py compute_eval149_metrics.py lowmem_launch.py run_long_rollout.py run_history_probe.py \
         realism_arms.py; do
  cp "$OV/$f" "$C/$f"
done
sed 's#/hpc/home/<user>#/path/to/pandwm_root#g' "$OV/cd_eval_hpc3.json" > "$C/cd_eval_frozen149_hpc3.json"
cp "$OV/md5_hpc1.txt" "$C/md5_hpc1.txt"
# 1) code and weights against HPC1's list (paths relative to pandwm_code; TATS_fvd is shared)
cd $DR/pandwm_code
sed 's#^\([0-9a-f]*  \)PanDWM_long_horizon_20260928/#\1PanDWM_eval149_20261011/#' "$OV/md5_hpc1.txt" > "$C/md5_check.txt"
md5sum -c --quiet "$C/md5_check.txt"
n=$(find PanDWM_eval149_20261011/src PanDWM_eval149_20261011/vendor -name "*.py" -o -name "*.pt" -o -name "*.pth" | wc -l)
m=$(grep -c PanDWM_eval149_20261011/ "$C/md5_check.txt")
[ "$n" = "$m" ] || { echo "FILE_COUNT_MISMATCH tree=$n list=$m"; exit 1; }
# 2) top-level scripts: the md5s printed on HPC1 on 2026-10-11
cd "$C"
md5sum -c --quiet <<'MD5'
f63528c17f93cb30764b292d4a6b9fcc  run_eval149_videos.py
6f04932cbdeb15c7c946174327ce820a  compute_eval149_metrics.py
fa4a3c25a68c2641037f2950287cfc72  lowmem_launch.py
eedacc0a665cb848f89ce3e944ad8c24  run_history_probe.py
dc9724a22b0a5422d75da595d9c710f3  run_long_rollout.py
e886ae512eebe300d8509e8bd8f3cfc7  src/dwm_pano/evaluate_4step.py
MD5
# 3) config: back-substituting the paths gives HPC1's file byte for byte
[ "$(sed 's#/path/to/pandwm_root#/hpc/home/<user>#g' cd_eval_frozen149_hpc3.json | md5sum | cut -d' ' -f1)" \
  = 7ab515ce3a7eacc4b24fc51b581cdd0d ] || { echo "CONFIG_MISMATCH"; exit 1; }
grep -o '/path/to/pandwm_root/[^"]*' cd_eval_frozen149_hpc3.json | sort -u | while read -r p; do
  [ -e "$p" ] || { echo "CONFIG_PATH_MISSING $p"; exit 1; }
done
echo "SETUP_EVAL149_HPC3_OK $C files=$n"
