#!/usr/bin/env bash
#SBATCH --job-name=pandwm_gs_aa
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/pandwm_root/pandwm_gs_antialiased_%j.log

# Evaluation-only rasterizer A/B.  It uses the existing G2 and Stage-2
# checkpoints; no model parameter or training configuration is changed.
set -euo pipefail

USER_CODE="$HOME/pandwm_code/PanDWM"
source "$USER_CODE/scripts/h20/env.sh"

# The runtime H20 mount is distinct from the login-node checkout.  Replace
# exactly one source file only when it is the audited pre-A/B version; abort
# rather than overwriting an unexpected concurrent edit.
runtime_file="$PANDWM_ROOT/code/PanDWM/src/dwm_pano/pipelines/ctsd_pano_gs_decoder.py"
source_file="$USER_CODE/src/dwm_pano/pipelines/ctsd_pano_gs_decoder.py"
old_sha="4fac81d3abdb4da8a6271d5b8eeb6cb05fb9a9b8d8ba626317698aaf6ee4a53c"
new_sha="fe42338d3951c27285afb8576f8dba98f4c970a5172bbd840c946f336737ef61"
runtime_sha="$(sha256sum "$runtime_file" | awk '{print $1}')"
if [[ "$runtime_sha" == "$old_sha" ]]; then
    install -m 0644 "$source_file" "$runtime_file"
elif [[ "$runtime_sha" != "$new_sha" ]]; then
    printf 'Refusing to overwrite unexpected runtime source: %s\n' "$runtime_sha" >&2
    exit 2
fi

export PANDWM_GS_RASTERIZE_MODE=antialiased
export PANDWM_MIN_FREE_MB=70000
export OUT="$PANDWM_ROOT/runs/gs_g2_ft_stage2mix75_250_offset2500/eval_decode_compare/250_holdout5_antialiased"

printf 'Rasterizer A/B: %s\n' "$PANDWM_GS_RASTERIZE_MODE"
printf 'Output: %s\n' "$OUT"
bash "$PANDWM_ROOT/code/PanDWM/scripts/h20/eval_gs_decode_compare_h20.sh"
