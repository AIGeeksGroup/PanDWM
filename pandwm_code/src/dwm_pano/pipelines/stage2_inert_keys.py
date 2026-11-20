"""Stage-2 config keys that are accepted and recorded but ignored.

Kept free of diffusers/einops imports so the set can be checked anywhere.
"""
# Keys this pipeline accepts and records but never acts on. The Stage-2 configs
# follow the official Causal-Forcing Stage-2 trainer, whose argument namespace
# is wider than what this pipeline implements; the unused keys are kept so the
# configs still construct. This pipeline implements plain consistency
# distillation: no critic is built, there is no distribution-matching term, and
# the reference-context behaviour is controlled by
# training_config["consistency_prefix_corruption_std"], not by `teacher_forcing`.
#
# A key is inert when the constructor parameter is never read outside the
# bookkeeping dict and `self.<key>` is never used.
INERT_OFFICIAL_STAGE2_KEYS = frozenset({
    "batch_size", "beta1", "beta1_critic", "beta2", "beta2_critic", "causal",
    "data_path", "denoising_loss_type", "dfake_gen_update_ratio",
    "distribution_loss", "fake_score_fsdp_wrap_strategy",
    "generator_fsdp_wrap_strategy", "gradient_checkpointing",
    "grad_accum", "image_or_video_shape", "is_causal", "load_raw_video",
    "log_iters", "lr", "lr_critic", "mixed_precision", "model_kwargs",
    "negative_prompt", "num_frame_per_block", "real_name",
    "real_score_fsdp_wrap_strategy", "seed", "sharding_strategy",
    "spatial_sf", "teacher_forcing", "text_encoder_fsdp_wrap_strategy",
    "timestep_shift", "total_batch_size", "trainer", "ts_schedule",
})

# Inert keys whose config value suggests a different training objective; the
# warning lists these first.
_MISLEADING_INERT_KEYS = (
    "distribution_loss", "teacher_forcing", "dfake_gen_update_ratio",
    "lr_critic", "denoising_loss_type", "timestep_shift")


def warn_inert_official_stage2_keys(parameters: dict):
    """Print which recorded parameters this pipeline ignores.

    Only keys the config actually set (not None) are reported.
    """
    supplied = sorted(
        k for k, v in parameters.items()
        if k in INERT_OFFICIAL_STAGE2_KEYS and v is not None)
    if not supplied:
        return
    leading = [k for k in _MISLEADING_INERT_KEYS if k in supplied]
    rest = [k for k in supplied if k not in leading]
    print(
        "[stage2] this pipeline implements plain consistency distillation; "
        "no critic is built and no distribution-matching term exists. "
        "{} config key(s) are recorded but inert.".format(len(supplied)),
        flush=True)
    if leading:
        print(
            "[stage2]   objective-misdescribing keys, ignored: {}".format(
                ", ".join(
                    "{}={!r}".format(k, parameters[k]) for k in leading)),
            flush=True)
    if rest:
        print(
            "[stage2]   also ignored: {}".format(", ".join(rest)), flush=True)
