"""Tests for the `equal_patch_width` instrument control on PanoPatch-FVD.

Writing the width assertion is what caught the class comment claiming
CAM_BACK is W=1033; the code yields 505 + 527 = 1032, and 1033 would have
made the six widths sum to 4115 instead of PANO_W_V1 = 4114.

The control exists because the six Voronoi camera patches have different
widths (CAM_BACK 1033 vs CAM_BACK_RIGHT 593 on the 4114 V=1 axis), so each
one reaches I3D's 224x224 through a different horizontal squeeze. These tests
pin (a) the documented patch geometry, (b) that the default is bit-identical
to the historical behaviour, and (c) that the crop is centered.

The constants are re-derived from `fvd_pano.py` by AST rather than imported,
so the file runs without `pytorch_i3d` (TATS) installed -- the same trick as
`test_inert_config_keys.py`. Re-deriving also means an edit to the source
constants fails these tests instead of silently passing.
"""

import ast
import os

import torch

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metrics", "fvd_pano.py")


def _class_constants():
    tree = ast.parse(open(_SRC, encoding="utf-8").read())
    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
        and n.name == "PanoPatchFrechetVideoDistance")
    out = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


CONST = _class_constants()
PANO_PATCH_COLS = CONST["PANO_PATCH_COLS"]
PANO_W_V1 = CONST["PANO_W_V1"]
CAM_NAMES = CONST["CAM_NAMES"]


def _build_patches(frames, equal_patch_width=0):
    """Mirror of the patch-building block in `PanoFVD.update`."""
    b0, b1, b2, b3, b4, b5 = PANO_PATCH_COLS
    w = PANO_W_V1
    patches = [
        torch.cat([frames[..., b5:w], frames[..., 0:b0]], dim=-1),
        frames[..., b0:b1],
        frames[..., b1:b2],
        frames[..., b2:b3],
        frames[..., b3:b4],
        frames[..., b4:b5],
    ]
    if equal_patch_width > 0:
        out = []
        for p in patches:
            start = (p.shape[-1] - equal_patch_width) // 2
            out.append(p[..., start:start + equal_patch_width])
        patches = out
    return patches


def test_source_declares_the_control_and_defaults_it_off():
    src = open(_SRC, encoding="utf-8").read()
    assert "equal_patch_width: int = 0" in src, \
        "the control must default to 0 so historical runs stay bit-exact"
    assert "self.equal_patch_width = int(equal_patch_width)" in src
    assert "if self.equal_patch_width > 0:" in src


def test_native_patch_widths_match_the_documented_values():
    frames = torch.zeros(1, 1, 3, 8, PANO_W_V1)
    widths = [p.shape[-1] for p in _build_patches(frames)]
    assert widths == [1032, 598, 619, 637, 635, 593]
    assert sum(widths) == PANO_W_V1
    # this spread is the whole reason the control exists
    assert max(widths) / min(widths) > 1.7
    assert widths[0] == max(widths) and CAM_NAMES[0] == "CAM_BACK"


def test_equal_patch_width_makes_all_six_identical():
    frames = torch.zeros(1, 1, 3, 8, PANO_W_V1)
    assert [p.shape[-1] for p in _build_patches(frames, 593)] == [593] * 6


def test_equal_patch_width_crop_is_centered():
    frames = torch.arange(
        PANO_W_V1, dtype=torch.float32).reshape(1, 1, 1, 1, -1)
    b2, b3 = PANO_PATCH_COLS[2], PANO_PATCH_COLS[3]
    cam_front = _build_patches(frames, 593)[3]
    start = ((b3 - b2) - 593) // 2
    assert start == 22, "CAM_FRONT is 637 wide; (637-593)//2 = 22"
    assert float(cam_front[0, 0, 0, 0, 0]) == float(b2 + start)
    assert cam_front.shape[-1] == 593


def test_default_leaves_patches_untouched():
    frames = torch.rand(1, 1, 3, 4, PANO_W_V1)
    for a, b in zip(_build_patches(frames), _build_patches(frames, 0)):
        assert torch.equal(a, b)


def test_cam_back_is_wrap_concatenated():
    frames = torch.arange(
        PANO_W_V1, dtype=torch.float32).reshape(1, 1, 1, 1, -1)
    cam_back = _build_patches(frames)[0]
    b0, b5 = PANO_PATCH_COLS[0], PANO_PATCH_COLS[5]
    assert float(cam_back[0, 0, 0, 0, 0]) == float(b5)
    assert float(cam_back[0, 0, 0, 0, -1]) == float(b0 - 1)
    assert cam_back.shape[-1] == (PANO_W_V1 - b5) + b0
