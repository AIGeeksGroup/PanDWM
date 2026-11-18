"""Pin the 3D-box -> V=1 panorama projection used by the consistency metric."""
import numpy as np
import pytest

from dwm_pano.metrics.instance_boxes import (
    PANO_W_V1, pano_column_to_v1)
from dwm_pano.datasets.pano_projection import PANO_W


def test_v1_mapping_covers_the_canvas_monotonically():
    tx = np.linspace(0, PANO_W - 1, 4001)
    v1 = pano_column_to_v1(tx)
    assert v1.min() >= -1.0
    assert v1.max() <= PANO_W_V1 + 1.0
    assert np.all(np.diff(v1) > 0), "mapping must be strictly increasing"


def test_v1_mapping_is_continuous_at_the_half_boundary():
    """Both branches must agree where the halves meet, else the seam jumps."""
    left = float(pano_column_to_v1(5376.0 - 1e-6))
    right = float(pano_column_to_v1(5376.0 + 1e-6))
    assert abs(left - right) < 1.0, f"{left} vs {right}"
    assert left == pytest.approx(2304.0, abs=1.0)


def test_v1_mapping_anchors():
    # Straight ahead (tx = 4800) must land inside CAM_FRONT's V=1 range
    # [1744, 2381) used by the FVD patch cuts.
    ahead = float(pano_column_to_v1(4800.0))
    assert 1744 <= ahead < 2381, ahead
    # The wrap ends of the canvas.
    assert float(pano_column_to_v1(0.0)) == pytest.approx(0.0, abs=1.0)
    assert float(pano_column_to_v1(PANO_W - 1)) == pytest.approx(
        PANO_W_V1, abs=2.0)
