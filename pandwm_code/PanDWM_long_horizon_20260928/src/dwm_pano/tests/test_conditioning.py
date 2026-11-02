"""CPU-only locality tests for PanDWM pipeline conditioning helpers.

Authoritative h20 command:
    cd /path/to/pandwm_root/code/PanDWM && PYTHONPATH=src /path/to/pandwm_root/envs/pandwm/bin/python3 -m pytest src/dwm_pano/tests -v

These tests call pipeline helpers directly and never construct a model. Heavy
pipeline imports are deferred to a fixture so missing optional dependencies
produce pytest skips instead of collection errors.
"""

import pytest


torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def pipeline_class():
    """Import the pipeline lazily to keep collection dependency-safe."""
    pytest.importorskip("diffusers")
    try:
        from dwm_pano.pipelines.ctsd_pano import CrossviewTemporalSD
    except Exception as exc:
        pytest.skip(
            "pipeline conditioning dependencies are unavailable: {}".format(
                exc
            )
        )
    return CrossviewTemporalSD


def _make_batch(sequence_length=4, view_count=2):
    """Build the smallest calibrated multi-frame batch used by the helpers."""
    batch_size = 1
    eye4 = torch.eye(4, dtype=torch.float32)

    ego_transforms = eye4.view(1, 1, 1, 4, 4).expand(
        batch_size, sequence_length, view_count, 4, 4
    ).clone()
    translations = torch.arange(sequence_length, dtype=torch.float32) * 0.1
    ego_transforms[..., 0, 3] = translations.view(1, sequence_length, 1)

    camera_transforms = eye4.view(1, 1, 1, 4, 4).expand(
        batch_size, sequence_length, view_count, 4, 4
    ).clone()
    if view_count > 1:
        camera_transforms[:, :, 1, 0, 3] = 0.25

    intrinsic = torch.tensor(
        [[8.0, 0.0, 8.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_intrinsics = intrinsic.view(1, 1, 1, 3, 3).expand(
        batch_size, sequence_length, view_count, 3, 3
    ).clone()
    image_size = torch.tensor(
        [16.0, 8.0], dtype=torch.float32
    ).view(1, 1, 1, 2).expand(
        batch_size, sequence_length, view_count, 2
    ).clone()

    return {
        "ego_transforms": ego_transforms,
        "camera_transforms": camera_transforms,
        "camera_intrinsics": camera_intrinsics,
        "image_size": image_size,
        "pts": torch.zeros(batch_size, sequence_length, 1),
        "fps": torch.full((batch_size,), 10.0),
    }


def _clone_batch(batch):
    return {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def _get_explicit_conditions(pipeline_class, batch):
    batch_size, sequence_length, view_count = batch["camera_transforms"].shape[:3]
    return pipeline_class.get_conditions(
        None,
        None,
        None,
        {"explicit_view_modeling": True},
        (batch_size, sequence_length, view_count, 4, 4, 4),
        batch,
        torch.device("cpu"),
        torch.float32,
    )


ACTION_CONFIG = {"camera_ego_sensor_indices": [0]}


@pytest.mark.parametrize("frame_index", (2, 3), ids=lambda value: "t-{}".format(value))
def test_get_action_ids_past_invariance(pipeline_class, frame_index):
    """Changing pose[t] leaves action indices strictly before t unchanged."""
    batch = _make_batch()
    baseline = pipeline_class.get_action_ids(batch, ACTION_CONFIG)

    changed_batch = _clone_batch(batch)
    changed_batch["ego_transforms"][:, frame_index, :, 0, 3] += 0.37
    changed = pipeline_class.get_action_ids(changed_batch, ACTION_CONFIG)

    assert torch.equal(baseline[:, :frame_index], changed[:, :frame_index])
    assert not torch.equal(
        baseline[:, frame_index:], changed[:, frame_index:]
    ), "pose perturbation did not reach current or future actions"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="get_action_ids action[0] contains the frame0-to-frame1 delta",
)
def test_get_action_ids_known_frame0_future_leak(pipeline_class):
    """Document the duplicated frame0-to-frame1 transition in action[0].

    The leak is currently inert: the shipped config has cond_with_action=false;
    added_time_ids is consumed only when perspective_modeling_type='implicit'.
    """
    batch = _make_batch()
    baseline = pipeline_class.get_action_ids(batch, ACTION_CONFIG)

    changed_batch = _clone_batch(batch)
    changed_batch["ego_transforms"][:, 1, :, 0, 3] += 0.75
    changed = pipeline_class.get_action_ids(changed_batch, ACTION_CONFIG)

    if torch.equal(baseline[:, 1], changed[:, 1]):
        raise RuntimeError(
            "frame 1 perturbation did not reach its transition action"
        )
    assert torch.equal(
        baseline[:, :1], changed[:, :1]
    ), "action_ids[:, 0] changed after perturbing pose[1]"


@pytest.mark.parametrize(
    ("field", "affected_condition"),
    (
        ("ego_transforms", "camera2referego"),
        ("camera_intrinsics", "camera_intrinsics_norm"),
        ("image_size", "camera_intrinsics_norm"),
    ),
    ids=("ego", "intrinsics", "image-size"),
)
def test_get_conditions_future_locality(
    pipeline_class, field, affected_condition
):
    """Future calibration changes leave past explicit-view conditions local."""
    split = 2
    batch = _make_batch()
    baseline = _get_explicit_conditions(pipeline_class, batch)
    changed_batch = _clone_batch(batch)

    if field == "ego_transforms":
        changed_batch[field][:, split:, :, 0, 3] += 0.5
    elif field == "camera_intrinsics":
        changed_batch[field][:, split:, :, 0, 0] += 1.0
    else:
        changed_batch[field][:, split:, :, 0] += 4.0

    changed = _get_explicit_conditions(pipeline_class, changed_batch)
    for condition_name in ("camera2referego", "camera_intrinsics_norm"):
        assert torch.equal(
            baseline[condition_name][:, :split],
            changed[condition_name][:, :split],
        ), "{} changed past {}".format(field, condition_name)

    assert not torch.equal(
        baseline[affected_condition][:, split:],
        changed[affected_condition][:, split:],
    ), "{} perturbation did not reach {}".format(field, affected_condition)


def test_reference_frame_indicator_is_prefix(pipeline_class):
    """The all-visible prediction branch selects exactly the requested prefix."""
    torch.manual_seed(7)
    batch_size, sequence_length, view_count = 2, 5, 2
    reference_latent_count = 3
    latents = torch.randn(
        batch_size, sequence_length, view_count, 4, 2, 2
    )
    noisy_input = torch.randn_like(latents)
    timesteps = torch.full(
        (batch_size, sequence_length, view_count), 500.0
    )
    training_config = {
        "generation_task_ratio": 0.0,
        "image_generation_ratio": 0.0,
        "all_reference_visible_ratio": 1.0,
        "reference_visible_rate": 0.0,
    }
    common_config = {"frame_prediction_style": "ctsd"}

    _, _, _, indicator = pipeline_class.try_make_input_for_prediction(
        noisy_input,
        latents,
        timesteps,
        training_config,
        common_config,
        generator=torch.Generator().manual_seed(7),
        reference_latent_count=reference_latent_count,
    )

    expected = torch.arange(sequence_length).view(1, sequence_length, 1) < \
        reference_latent_count
    expected = expected.expand(batch_size, sequence_length, view_count)
    assert indicator.dtype == torch.bool
    assert torch.equal(indicator, expected)
    assert not indicator[:, reference_latent_count:].any()
