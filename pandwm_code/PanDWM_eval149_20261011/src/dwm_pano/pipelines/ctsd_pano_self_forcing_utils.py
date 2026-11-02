"""Small, CPU-testable helpers for paired self-forcing experiments."""

import dwm.functional


def validate_rollout_config(config):
    result = {
        "enabled": bool(config.get("enabled", True)),
        "rollout_windows": int(config.get("rollout_windows", 4)),
        "window_length": int(config.get("window_length", 19)),
        "reference_frame_count": int(config.get("reference_frame_count", 3)),
        "window_stride": int(config.get("window_stride", 16)),
        "curriculum_steps": int(config.get("curriculum_steps", 2000)),
        "sampler_mode": config.get("sampler_mode", "stochastic"),
    }
    if result["rollout_windows"] < 2:
        raise ValueError("rollout_windows must be >= 2")
    if result["reference_frame_count"] < 1:
        raise ValueError("reference_frame_count must be positive")
    if result["window_stride"] != result["window_length"] - \
            result["reference_frame_count"]:
        raise ValueError("stride must equal window length minus history")
    if result["curriculum_steps"] < 0:
        raise ValueError("curriculum_steps must be nonnegative")
    if result["sampler_mode"] not in ("stochastic", "deterministic"):
        raise ValueError("sampler_mode must be stochastic or deterministic")
    return result


def slice_sequence_batch(batch, start, length, exceptions):
    """Slice every time-dependent condition exactly like AR inference."""
    end = start + length
    if start < 0 or end > batch["vae_images"].shape[1]:
        raise ValueError("window escapes batch time axis")
    return {
        key: (value if key in exceptions else
              dwm.functional.take_sequence_clip(value, start, end))
        for key, value in batch.items()
    }
