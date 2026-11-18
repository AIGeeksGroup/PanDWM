import pytest
import torch

from scripts.horizondrive.evaluate_real_vs_real import load_future
from scripts.horizondrive.summarize_common_multiseed import (
    metric_summary,
    parse_pandwm_log,
)


def test_real_vs_real_loader_preserves_uint8(tmp_path):
    path = tmp_path / "clip.pt"
    source = torch.arange(12 * 3 * 2 * 2, dtype=torch.uint8).reshape(
        12, 3, 2, 2)
    torch.save(source, path)
    loaded = load_future(path, start=1, frames=10)
    assert loaded.dtype == torch.uint8
    assert torch.equal(loaded, source[1:11])


def test_parse_pandwm_log_uses_per_camera_front_not_pooled_value():
    text = """
Step 10000, front_fvd: 147.167 front_fid: 19.159
Step 10000, per-camera n6bridge: CAM_BACK=360.248 CAM_FRONT=403.228
"""
    result = parse_pandwm_log(text)
    assert result["fid"] == pytest.approx(19.159)
    assert result["fvd"] == pytest.approx(403.228)
    assert result["pooled_six_camera_fvd"] == pytest.approx(147.167)


def test_metric_summary_uses_sample_standard_deviation():
    result = metric_summary([
        {"fid": 1.0, "fvd": 10.0},
        {"fid": 2.0, "fvd": 20.0},
        {"fid": 3.0, "fvd": 30.0},
    ])
    assert result["fid"]["mean"] == pytest.approx(2.0)
    assert result["fid"]["sample_std"] == pytest.approx(1.0)
    assert result["fvd"]["mean"] == pytest.approx(20.0)
    assert result["fvd"]["sample_std"] == pytest.approx(10.0)
