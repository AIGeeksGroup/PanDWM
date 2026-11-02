#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

export CFG="${CFG:-$PANDWM_ROOT/code/PanDWM/configs/ctsd/pandwm/h20/gs_decoder_temporal_visibility_multiorigin.json}"
export RUN="${RUN:-$PANDWM_ROOT/runs/gs_temporal_visibility_multiorigin_v3_diagnostic}"

python3 - "$CFG" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = json.load(handle)
training = config["pipeline"]["training_config"]
decoder = config["pipeline"]["decoder"]
required = {
    "calibrated camera origins": decoder.get("use_camera_origins") is True,
    "Plucker ray embedding": decoder.get("ray_embedding_mode") == "plucker",
    "masked depth resize": 0.0 < training.get(
        "range_min_valid_fraction", 0.0) <= 1.0,
    "camera-origin warmup": training.get(
        "camera_origin_warmup_steps", 0) > 0,
    "absolute visibility tolerance": training.get(
        "cross_depth_abs_tolerance_m", 0.0) > 0.0,
    "relative visibility tolerance": training.get(
        "cross_depth_rel_tolerance", 0.0) >= 0.0,
    "robust structural loss": 0.0 <= training.get(
        "cross_ssim_weight", -1.0) < 1.0,
    "render-visibility photometric diagnostic": training.get(
        "cross_photometric_mask") == "render_visibility",
    "weak alpha coverage diagnostic": training.get(
        "weight_cross_alpha_coverage") == 0.002,
    "conservative target alpha": training.get(
        "cross_target_alpha") == 0.25,
    "SSIM cross pressure disabled": training.get(
        "cross_ssim_weight") == 0.0,
}
failed = [name for name, valid in required.items() if not valid]
if failed:
    raise RuntimeError(
        "Temporal GS diagnostic guard failed: " + "; ".join(failed))
print("Temporal GS geometry/diagnostic guard passed.")
PY

exec "$SCRIPT_DIR/train_gs_temporal_h20.sh"
