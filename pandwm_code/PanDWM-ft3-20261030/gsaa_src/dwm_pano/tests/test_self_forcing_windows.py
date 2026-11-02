"""Check the temporal slices and pair symmetry used by on-policy CD."""

import pytest

torch = pytest.importorskip("torch")

from dwm_pano.pipelines.ctsd_pano_self_forcing_utils import (
    slice_sequence_batch, validate_rollout_config)


def test_four_windows_cover_exactly_67_frames_without_future_context():
    config = validate_rollout_config({})
    batch = {
        "vae_images": torch.arange(67).reshape(1, 67, 1),
        "pts": torch.arange(67).reshape(1, 67, 1),
        "crossview_mask": torch.ones(1, 2, 2),
        "fps": torch.tensor([10.0]),
        "clip_text": [[str(i) for i in range(67)]],
    }
    for window in range(4):
        start = window * config["window_stride"]
        part = slice_sequence_batch(
            batch, start, config["window_length"], ["crossview_mask"])
        assert part["vae_images"].shape[1] == 19
        assert part["pts"][0, 0, 0].item() == start
        assert part["pts"][0, -1, 0].item() == start + 18
        assert part["clip_text"][0][0] == str(start)
        assert part["crossview_mask"].shape == (1, 2, 2)
        if window:
            assert start == (window - 1) * 16 + 16
    assert start + 18 == 66


def test_invalid_rollout_stride_is_rejected():
    with pytest.raises(ValueError, match="stride"):
        validate_rollout_config({"window_stride": 15})
