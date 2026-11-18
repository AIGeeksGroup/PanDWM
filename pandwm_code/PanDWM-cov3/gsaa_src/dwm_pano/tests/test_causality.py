"""Causality tests for the PanDWM DiT temporal attention path.

Authoritative h20 command:
    cd /path/to/pandwm_root/code/PanDWM && PYTHONPATH=src \
      /path/to/pandwm_root/envs/pandwm/bin/python3 -m pytest \
      src/dwm_pano/tests -v

FlexAttention backward is CUDA-only, so CPU runs skip Test 2. Sync
``src/dwm_pano/tests/`` into the cluster overlay before the h20 run. The tiny
model is random-init and uses fp32 without autocast or checkpointing.
"""

import math

import pytest

torch = pytest.importorskip("torch")


SPLIT_POINTS = (1, 2, 3)
GRADIENT_KEYS = (
    "sample",
    "encoder_hidden_states",
    "pooled_projections",
    "camera_intrinsics_norm",
    "camera2referego",
)
PER_FRAME_KEYS = {
    "sample",
    "timestep",
    "encoder_hidden_states",
    "pooled_projections",
    "camera_intrinsics_norm",
    "camera2referego",
}


def _frame_seqlen(case, attention_type="full"):
    """Mirror the pipeline formula after the model patch embedding."""
    patch_size = 2
    height = case["height"] // patch_size
    width = case["width"] // patch_size
    if attention_type == "rowwise":
        return width
    if attention_type == "full":
        return height * width
    raise ValueError("unsupported temporal attention type: {}".format(
        attention_type
    ))


def _forward(model, inputs, mask):
    result, _, _ = model(
        inputs["sample"],
        inputs["timestep"],
        encoder_hidden_states=inputs["encoder_hidden_states"],
        pooled_projections=inputs["pooled_projections"],
        disable_crossview=inputs["disable_crossview"],
        disable_temporal=inputs["disable_temporal"],
        temporal_attention_mask=mask,
        camera_intrinsics_norm=inputs["camera_intrinsics_norm"],
        camera2referego=inputs["camera2referego"],
    )
    return result[0]


def _run(model, inputs, mask):
    with torch.no_grad():
        return _forward(model, inputs, mask)


def _clone_inputs(inputs):
    return {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in inputs.items()
    }


def _prefix_inputs(inputs, count):
    result = {}
    for name, value in inputs.items():
        if name in PER_FRAME_KEYS:
            result[name] = value[:, :count].clone()
        else:
            result[name] = value.clone() if torch.is_tensor(value) else value
    return result


def _perturb(inputs, case_name, split):
    changed = _clone_inputs(inputs)
    generator = torch.Generator(device="cpu").manual_seed(9000 + split)

    def delta(shape, scale=1.0):
        return torch.randn(
            shape, generator=generator, dtype=torch.float32
        ).to(changed["sample"].device) * scale

    if case_name == "sample":
        changed["sample"][:, split:] += delta(
            changed["sample"][:, split:].shape, scale=10.0
        )
    elif case_name == "timestep":
        changed["timestep"][:, split:] = 700.0
    elif case_name == "encoder_hidden_states":
        changed["encoder_hidden_states"][:, split:] += delta(
            changed["encoder_hidden_states"][:, split:].shape
        )
    elif case_name == "pooled_projections":
        changed["pooled_projections"][:, split:] += delta(
            changed["pooled_projections"][:, split:].shape
        )
    elif case_name == "camera":
        changed["camera_intrinsics_norm"][:, split:] += delta(
            changed["camera_intrinsics_norm"][:, split:].shape, scale=0.1
        )
        changed["camera2referego"][:, split:] += delta(
            changed["camera2referego"][:, split:].shape, scale=0.1
        )
    else:
        raise AssertionError("unknown perturbation case: {}".format(case_name))
    return changed


@pytest.mark.parametrize(
    "perturbation",
    (
        "sample",
        "timestep",
        "encoder_hidden_states",
        "pooled_projections",
        "camera",
    ),
    ids=(
        "case-a-sample",
        "case-b-timestep",
        "case-c-encoder",
        "case-d-pooled",
        "case-e-camera",
    ),
)
@pytest.mark.parametrize("split", SPLIT_POINTS, ids=lambda k: "k-{}".format(k))
def test_future_perturbation_is_causal(
    tiny_model,
    input_builder,
    block_mask_builder,
    spatial_case,
    perturbation,
    split,
):
    """Future changes must not alter prefix outputs under a BlockMask."""
    inputs = input_builder(
        spatial_case, device=next(tiny_model.parameters()).device
    )
    mask = block_mask_builder(
        inputs["sample"].device,
        num_frames=inputs["sample"].shape[1],
        frame_seqlen=_frame_seqlen(spatial_case),
    )
    pred_a = _run(tiny_model, inputs, mask)
    pred_b = _run(tiny_model, _perturb(inputs, perturbation, split), mask)

    if not torch.equal(pred_a[:, :split], pred_b[:, :split]):
        pred_repeat = _run(tiny_model, inputs, mask)
        baseline_is_deterministic = torch.equal(pred_a, pred_repeat)
        max_abs_diff = (
            pred_a[:, :split] - pred_b[:, :split]
        ).abs().max().item()
        pytest.fail(
            "{} changed prefix before frame {}; max abs diff={:.8g}; "
            "identical baseline repeat={}".format(
                perturbation,
                split,
                max_abs_diff,
                baseline_is_deterministic,
            )
        )
    assert not torch.equal(pred_a[:, split:], pred_b[:, split:]), (
        "power check failed: {} perturbation had no future effect".format(
            perturbation
        )
    )


@pytest.mark.parametrize("split", SPLIT_POINTS, ids=lambda k: "k-{}".format(k))
def test_unmasked_negative_control_changes_prefix(
    tiny_model, input_builder, spatial_case, split
):
    """The same sample perturbation must change the prefix without a mask."""
    inputs = input_builder(
        spatial_case, device=next(tiny_model.parameters()).device
    )
    changed = _perturb(inputs, "sample", split)
    pred_a = _run(tiny_model, inputs, None)
    pred_b = _run(tiny_model, changed, None)
    assert not torch.equal(pred_a[:, :split], pred_b[:, :split]), (
        "mask=None negative control did not expose future attention"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Test 2 requires CUDA FlexAttention backward (run on h20)",
)
@pytest.mark.parametrize("split", SPLIT_POINTS, ids=lambda k: "k-{}".format(k))
def test_future_gradients_are_exactly_zero(
    tiny_model,
    input_builder,
    block_mask_builder,
    spatial_case,
    split,
    cuda_device,
):
    """CUDA FlexAttention backward has exact zero future-to-past gradients."""
    model = tiny_model.to(cuda_device)
    inputs = input_builder(
        spatial_case, device=cuda_device, requires_grad=True, seed=4321
    )
    mask = block_mask_builder(
        cuda_device,
        num_frames=inputs["sample"].shape[1],
        frame_seqlen=_frame_seqlen(spatial_case),
    )
    _forward(model, inputs, mask)[:, :split].sum().backward()

    for name in GRADIENT_KEYS:
        gradient = inputs[name].grad
        assert gradient is not None, "missing gradient for {}".format(name)
        assert torch.all(gradient[:, split:] == 0), (
            "future {} gradient is not exact zero".format(name)
        )
    assert inputs["sample"].grad[:, :split].abs().sum() > 0, (
        "power check failed: prefix output has no prefix sample gradient"
    )

    model.zero_grad(set_to_none=True)
    negative_inputs = input_builder(
        spatial_case, device=cuda_device, requires_grad=True, seed=4321
    )
    _forward(model, negative_inputs, None)[:, :split].sum().backward()
    assert negative_inputs["sample"].grad[:, split:].abs().sum() > 0, (
        "mask=None gradient negative control did not expose future input"
    )


def test_prefix_parity_precursor(
    tiny_model, input_builder, block_mask_builder, spatial_case
):
    """A prefix run and a full-window run agree before the missing cache API."""
    inputs = input_builder(
        spatial_case, device=next(tiny_model.parameters()).device
    )
    total_frames = inputs["sample"].shape[1]
    split = 2
    full_mask = block_mask_builder(
        inputs["sample"].device,
        num_frames=total_frames,
        frame_seqlen=_frame_seqlen(spatial_case),
    )
    prefix_inputs = _prefix_inputs(inputs, split)
    prefix_mask = block_mask_builder(
        inputs["sample"].device,
        num_frames=split,
        frame_seqlen=_frame_seqlen(spatial_case),
    )
    full_prefix = _run(tiny_model, inputs, full_mask)[:, :split]
    prefix_output = _run(tiny_model, prefix_inputs, prefix_mask)
    max_abs_diff = (full_prefix - prefix_output).abs().max().item()
    assert torch.allclose(full_prefix, prefix_output, atol=1e-5, rtol=1e-4), (
        "prefix parity max abs diff: {:.8g}".format(max_abs_diff)
    )


@pytest.mark.parametrize(
    "attention_type", ("rowwise", "full"), ids=("rowwise", "full")
)
def test_block_mask_shape_parity(
    block_mask_builder, spatial_case, attention_type
):
    """Pipeline frame sizes produce the causal gold mask and pad diagonal."""
    total_frames = 4
    frame_seqlen = _frame_seqlen(spatial_case, attention_type)
    total_length = total_frames * frame_seqlen
    padded_length = math.ceil(total_length / 128) * 128
    mask = block_mask_builder(
        torch.device("cpu"),
        num_frames=total_frames,
        frame_seqlen=frame_seqlen,
    )
    dense = mask.to_dense()
    while dense.dim() > 2:
        dense = dense[0]
    dense = dense.to(dtype=torch.bool)
    # BlockMask.to_dense may return block occupancy instead of token detail.
    # Evaluate mask_mod directly when necessary to obtain the token matrix.
    if tuple(dense.shape[-2:]) != (padded_length, padded_length):
        q_idx = torch.arange(padded_length).view(-1, 1)
        kv_idx = torch.arange(padded_length).view(1, -1)
        dense = mask.mask_mod(None, None, q_idx, kv_idx).to(dtype=torch.bool)

    expected_unpadded = torch.kron(
        torch.tril(torch.ones((total_frames, total_frames), dtype=torch.bool)),
        torch.ones((frame_seqlen, frame_seqlen), dtype=torch.bool),
    )
    assert torch.equal(
        dense[:total_length, :total_length], expected_unpadded
    )
    expected = torch.zeros(
        (padded_length, padded_length), dtype=torch.bool
    )
    expected[:total_length, :total_length] = expected_unpadded
    if padded_length > total_length:
        expected[total_length:, total_length:] = torch.eye(
            padded_length - total_length, dtype=torch.bool
        )
    assert torch.equal(dense, expected)


def test_dit_has_no_temporal_resnet_block(tiny_model):
    """The shipped DiT must not contain the UNet symmetric temporal conv."""
    diffusers = pytest.importorskip("diffusers")
    try:
        import diffusers.models.resnet as dres
        import dwm.models.crossview_temporal as ct
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip("ResNet guard dependencies are unavailable: {}".format(exc))
    forbidden = (dres.TemporalResnetBlock, ct.ResBlock)
    offenders = [
        type(module).__name__
        for module in tiny_model.modules()
        if isinstance(module, forbidden)
    ]
    assert not offenders, "DiT contains temporal ResNet modules: {}".format(
        offenders
    )
    _ = diffusers


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "ResBlock still uses diffusers 0.31.0 TemporalResnetBlock Conv3d "
        "kernel=(3,1,1) with symmetric padding=(1,0,0)"
    ),
)
def test_resblock_symmetric_conv_leak_is_detectable():
    """This strict xfail proves the guard catches a revived UNet lineage."""
    pytest.importorskip("diffusers")
    try:
        import dwm.models.crossview_temporal as ct
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip("ResBlock dependencies are unavailable: {}".format(exc))

    torch.manual_seed(19)
    # diffusers 0.31.0 hard-codes GroupNorm(groups=32), so 32 channels are
    # required to reach the intended causal assertion instead of failing setup.
    channels = 32
    block = ct.ResBlock(
        in_channels=channels, temb_channels=channels
    ).eval().float()
    hidden_states = torch.randn(1, 4, 1, channels, 4, 4)
    temb = torch.randn(1, 4, 1, channels)
    changed = hidden_states.clone()
    changed[:, 1:] += torch.randn_like(changed[:, 1:]) * 10.0
    disable_temporal = torch.zeros(1, dtype=torch.bool)
    with torch.no_grad():
        output_a = block(
            hidden_states, temb=temb, disable_temporal=disable_temporal
        )
        output_b = block(
            changed, temb=temb, disable_temporal=disable_temporal
        )
    assert torch.equal(output_a[:, :1], output_b[:, :1])


@pytest.mark.skip(
    reason=(
        "No first-party KV cache exists: FIFO recomputes every full window, "
        "an assertion requires frame_prediction_style=='diffusion_forcing' "
        "(the shipped style is 'ctsd'), and its "
        "streaming conditions omit temporal_attention_mask (bidirectional). "
        "A future API should cache per-temporal-block K/V for frames 0..t-1 "
        "with layout (B*V, heads, t*HW, head_dim) for full attention; "
        "crossview, spatial, adapter, and ray-map paths are frame-local and "
        "there is no temporal-conv state."
    )
)
def test_cache_no_cache_parity():
    """Reserved for the first-party KV-cache streaming API."""
    raise AssertionError("No cache API is present yet")
