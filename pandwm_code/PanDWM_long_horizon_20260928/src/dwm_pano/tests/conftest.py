"""Shared fixtures for the PanDWM causality tests.

Authoritative h20 command::

    cd /path/to/pandwm_root/code/PanDWM && PYTHONPATH=src \
      /path/to/pandwm_root/envs/pandwm/bin/python3 -m pytest \
      src/dwm_pano/tests -v

FlexAttention backward is CUDA-only, so CPU runs skip Test 2. Sync
``src/dwm_pano/tests/`` into the cluster checkout and any active OpenDWM
overlay before the authoritative run.
"""

import pytest

torch = pytest.importorskip("torch")


CUDA_AVAILABLE = bool(torch.cuda.is_available())
CUDA_ONLY = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="Test 2 requires CUDA FlexAttention backward (run on h20)",
)


@pytest.fixture(
    params=(
        {"height": 8, "width": 16, "id": "h8-w16-no-pad"},
        {"height": 8, "width": 12, "id": "h8-w12-forced-pad"},
    ),
    ids=lambda case: case["id"],
)
def spatial_case(request):
    """The unpadded and right-padded latent layouts required by the spec."""
    return request.param


@pytest.fixture(scope="function")
def tiny_model():
    """Construct the random-init tiny DiT without loading any checkpoint."""
    pytest.importorskip("diffusers")
    try:
        # torch 2.5 can execute FlexAttention eagerly on CPU, but its Inductor
        # lowering is CUDA-only. Keep CPU-only runs usable without changing the
        # production module; newer CPU builds retain the compiled path.
        if not CUDA_AVAILABLE:
            version = torch.__version__.split("+")[0].split(".")
            if len(version) >= 2 and (int(version[0]), int(version[1])) < (2, 6):
                import dwm_pano.models.flex_block_causal as flex_impl
                from torch.nn.attention.flex_attention import flex_attention
                flex_impl._compiled_flex_attention = flex_attention
        from dwm_pano.models.crossview_temporal_dit_pano import (
            DiTCrossviewTemporalConditionModel,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip("PanDWM DiT dependencies are unavailable: {}".format(exc))

    torch.manual_seed(0)
    # Keep this constructor in lockstep with the audited source signature;
    # the head-width adjustment below is required by h20's torch 2.5 Triton.
    model = DiTCrossviewTemporalConditionModel(
        sample_size=16,
        patch_size=2,
        in_channels=4,
        out_channels=4,
        num_layers=2,
        # torch 2.5 Triton requires FlexAttention head dimensions >= 16.
        # Keep inner_dim=32 by using two 16-wide heads.
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        caption_projection_dim=32,
        pooled_projection_dim=32,
        pos_embed_max_size=32,
        enable_crossview=True,
        crossview_attention_type="rowwise",
        crossview_block_layers=[1],
        enable_temporal=True,
        temporal_attention_type="full",
        temporal_block_layers=[1],
        mixer_type="AlphaBlender",
        merge_factor=2,
        merge_strategy="learned_with_images",
        perspective_modeling_type="explicit",
        condition_image_adapter_config=None,
    )
    # torch 2.5 FlexAttention has no CPU lowering, while newer CPU builds do.
    # Prefer CUDA when present and retain the CPU path for CPU-only hosts.
    device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    return model.eval().float().to(device)


def _build_inputs(case, *, batch_size=1, sequence_length=4, view_count=2,
                  device=None, requires_grad=False, seed=1234):
    """Build deterministic per-frame tensors matching DiT.forward exactly."""
    device = torch.device("cpu" if device is None else device)
    height = int(case["height"])
    width = int(case["width"])
    if min(batch_size, sequence_length, view_count, height, width) <= 0:
        raise ValueError("input dimensions must be positive")
    if height % 2 or width % 2:
        raise ValueError("height and width must be divisible by patch_size=2")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def randn(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32)

    sample = randn((batch_size, sequence_length, view_count, 4, height, width))
    timestep = torch.full(
        (batch_size, sequence_length, view_count), 500.0, dtype=torch.float32
    )
    encoder_hidden_states = randn(
        (batch_size, sequence_length, view_count, 8, 32)
    )
    pooled_projections = randn((batch_size, sequence_length, view_count, 32))

    intrinsic = torch.tensor(
        [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_intrinsics_norm = intrinsic.view(1, 1, 1, 3, 3).expand(
        batch_size, sequence_length, view_count, 3, 3
    ).clone()
    camera2referego = torch.eye(4, dtype=torch.float32).view(
        1, 1, 1, 4, 4
    ).expand(batch_size, sequence_length, view_count, 4, 4).clone()

    result = {
        "sample": sample,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_projections": pooled_projections,
        "camera_intrinsics_norm": camera_intrinsics_norm,
        "camera2referego": camera2referego,
        "disable_crossview": torch.zeros(batch_size, dtype=torch.bool),
        "disable_temporal": torch.zeros(batch_size, dtype=torch.bool),
    }
    result = {
        name: value.to(device=device) if torch.is_tensor(value) else value
        for name, value in result.items()
    }
    if requires_grad:
        for name in (
            "sample",
            "encoder_hidden_states",
            "pooled_projections",
            "camera_intrinsics_norm",
            "camera2referego",
        ):
            result[name].requires_grad_(True)
    return result


@pytest.fixture
def input_builder():
    """Return the deterministic input builder used by all model tests."""
    torch.manual_seed(1)
    return _build_inputs


@pytest.fixture
def block_mask_builder():
    """Return a helper for the audited full-temporal BlockMask recipe."""
    try:
        from dwm_pano.models.flex_block_causal import (
            make_blockwise_causal_block_mask,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip("FlexAttention dependencies are unavailable: {}".format(exc))

    def build(device, *, num_frames, frame_seqlen):
        return make_blockwise_causal_block_mask(
            torch.device(device),
            num_frames=int(num_frames),
            frame_seqlen=int(frame_seqlen),
        )

    return build


@pytest.fixture
def cuda_device():
    """Provide a CUDA device or cleanly skip on CPU-only hosts."""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA is unavailable; authoritative Test 2 run is on h20")
    return torch.device("cuda")
