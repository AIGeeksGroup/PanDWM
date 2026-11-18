"""Which Stage-2 config keys are recorded and then ignored.

Split out of the pipeline so the declaration and its machine check carry no
diffusers/einops import: the whole point is that the set stays verifiable
anywhere, including on a machine that cannot load the model stack.
"""
# Keys this pipeline accepts, records, and then never acts on.
#
# The Stage-2 configs were written against the official Causal-Forcing Stage-2
# trainer, whose argument namespace is much wider than what this pipeline
# implements.  The unused ones are kept as accepted parameters so the existing
# configs still construct, but reading the config gives a badly wrong picture
# of the recipe: `distribution_loss="dmd"`, `dfake_gen_update_ratio=5`,
# `lr_critic`, `beta*_critic` and `real_score/fake_score_fsdp_wrap_strategy`
# all suggest a distribution-matching objective trained against a critic, and
# `teacher_forcing=True` suggests the flag does something.  **This pipeline
# implements plain consistency distillation.  No critic is built, no
# distribution-matching term exists, and the reference-context behaviour is
# controlled by `training_config["consistency_prefix_corruption_std"]`, not by
# `teacher_forcing`.**
#
# Derived by AST analysis, not by hand: a key is inert when the constructor
# parameter has no load outside the bookkeeping dict *and* `self.<key>` is
# never read anywhere in the module.  `tests/test_inert_config_keys.py`
# re-derives this set and fails if it drifts, so wiring one of these up is
# caught rather than silently contradicting this comment.
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

# Of the inert keys, the ones whose config value actively misdescribes the
# training objective. Named separately so the warning leads with them.
_MISLEADING_INERT_KEYS = (
    "distribution_loss", "teacher_forcing", "dfake_gen_update_ratio",
    "lr_critic", "denoising_loss_type", "timestep_shift")


def warn_inert_official_stage2_keys(parameters: dict):
    """Say out loud which recorded parameters this pipeline ignores.

    Only reports keys the config actually set, because a `None` left at its
    default misdescribes nothing.
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
