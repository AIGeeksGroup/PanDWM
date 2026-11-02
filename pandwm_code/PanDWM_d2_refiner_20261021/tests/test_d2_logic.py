"""CPU logic tests for the D2 refiner (2026-10-21, D2_PLAN_20261021.md Tasks 1-3).

    PYTHONPATH=<D2>/src:<D2>/src_base python tests/test_d2_logic.py
"""

import json
import sys
import tempfile
import traceback
from pathlib import Path

import torch

from dwm_pano.models.crossview_temporal_dit_pano import DiTCrossviewTemporalConditionModel
from dwm_pano_d2.data import D2RenderDataset
from dwm_pano_d2.model import D2RenderDiT
from dwm_pano.pipelines.ctsd_pano import CrossviewTemporalSD
from dwm_pano_d2.pipeline import (CLIP, WINDOW, RenderInjector, d2_render_dropped, d2_render_init, d2_window_start,
                                  render_init_input, swap_input_builder, take_window)

TINY = dict(sample_size=8, patch_size=2, in_channels=16, num_layers=2, attention_head_dim=8, num_attention_heads=2,
            joint_attention_dim=32, caption_projection_dim=16, pooled_projection_dim=8, out_channels=16,
            pos_embed_max_size=16)


def tiny_model():
    torch.manual_seed(0)
    return D2RenderDiT(render_condition_channels=17, render_condition_scale=10.0, **TINY).eval()


def tiny_inputs(b=1, t=3, v=2, h=8, w=16):
    g = torch.Generator().manual_seed(1)
    return {"sample": torch.randn(b, t, v, 16, h, w, generator=g),
            "timestep": torch.full((b, t, v), 500.0),
            "encoder_hidden_states": torch.randn(b, t, v, 5, 32, generator=g),
            "pooled_projections": torch.randn(b, t, v, 8, generator=g)}


def tiny_render(b=1, t=3, v=2, h=8, w=16, seed=2):
    return torch.randn(b, t, v, 17, h, w, generator=torch.Generator().manual_seed(seed))


def run(model, inputs, **kw):
    sample = inputs["sample"]
    rest = {k: v for k, v in inputs.items() if k not in ("sample", "timestep")}
    return model(sample, inputs["timestep"], **rest, **kw)[0][0]


# ---------------------------------------------------------------------------------------------- Task 1: model
def test_model_zero_init_is_base_bit_for_bit():
    m = tiny_model()
    x = tiny_inputs()
    with torch.no_grad():
        out_none = run(m, x)
        out_render = run(m, x, render_condition=tiny_render())
        base = DiTCrossviewTemporalConditionModel(**TINY).eval()
        state = {k: v for k, v in m.state_dict().items() if not k.startswith("render_embed.")}
        base.load_state_dict(state, strict=True)
        out_base = run(base, x)
    assert torch.equal(out_none, out_render), "zero-init render branch changed the output"
    assert torch.equal(out_none, out_base), "D2RenderDiT without render differs from the base model"


def test_model_render_changes_output_and_gets_gradient():
    m = tiny_model()
    with torch.no_grad():
        m.render_embed.weight.normal_(0, 0.02)
    x = tiny_inputs()
    r = tiny_render()
    out_none = run(m, x).detach()
    out = run(m, x, render_condition=r)
    assert not torch.allclose(out_none, out.detach()), "render condition has no effect"
    out.square().mean().backward()
    g = m.render_embed.weight.grad
    assert g is not None and float(g.abs().sum()) > 0, "no gradient reaches render_embed"
    assert m._d2_pending is None, "pending render tokens were not cleared"


def test_model_state_dict_keys():
    m = tiny_model()
    base = DiTCrossviewTemporalConditionModel(**TINY)
    extra = set(m.state_dict()) - set(base.state_dict())
    missing = set(base.state_dict()) - set(m.state_dict())
    assert extra == {"render_embed.weight", "render_embed.bias"}, extra
    assert not missing, missing
    assert float(m.render_embed.weight.abs().sum()) == 0 and float(m.render_embed.bias.abs().sum()) == 0


def test_model_rejects_misshaped_render():
    m = tiny_model()
    x = tiny_inputs()
    for bad in (tiny_render(t=2), tiny_render(h=4), torch.randn(1, 3, 2, 16, 8, 16)):
        try:
            run(m, x, render_condition=bad)
        except ValueError:
            continue
        raise AssertionError(f"misshaped render {tuple(bad.shape)} was accepted")
    assert m._d2_pending is None


def test_model_pending_cleared_after_error():
    m = tiny_model()
    x = tiny_inputs()
    x_bad = dict(x)
    x_bad["pooled_projections"] = torch.randn(1, 3, 2, 7)            # wrong width -> error inside the base forward
    try:
        run(m, x_bad, render_condition=tiny_render())
    except Exception:
        pass
    assert m._d2_pending is None, "pending render tokens leaked after an exception"


# ------------------------------------------------------------------------------------------- Task 2: pipeline
def test_window_start_deterministic_and_covering():
    seen = [d2_window_start(s, 0) for s in range(1000)]
    assert seen == [d2_window_start(s, 0) for s in range(1000)]
    assert set(seen) == set(range(CLIP - WINDOW + 1)), sorted(set(seen))
    assert d2_window_start(5, 0) != d2_window_start(5, 1) or d2_window_start(6, 0) != d2_window_start(6, 1)


def test_render_dropout_rate():
    draws = [d2_render_dropped(s, 0, r, 0.1) for s in range(500) for r in range(8)]
    rate = sum(draws) / len(draws)
    assert 0.08 < rate < 0.12, rate
    assert not any(d2_render_dropped(s, 0, 0, 0.0) for s in range(100))


class FakeWrapper:
    def __init__(self):
        self.mask_module = None
        self.calls = []
        self.mode = None

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "out"

    def train(self):
        self.mode = "train"


def test_injector_adds_render_and_forwards_attributes():
    w = FakeWrapper()
    r = torch.ones(1)
    inj = RenderInjector(w, r)
    inj.train()
    assert w.mode == "train"
    assert getattr(inj, "mask_module", "missing") is None
    assert inj("a", k=1) == "out"
    assert w.calls[0][1]["render_condition"] is r and w.calls[0][1]["k"] == 1 and inj.calls == 1
    try:
        inj("a", render_condition=r)
    except RuntimeError:
        pass
    else:
        raise AssertionError("double render_condition accepted")


def fake_clip_batch():
    t = torch.arange(CLIP, dtype=torch.float32)
    return {"vae_images": t.view(1, CLIP, 1, 1, 1, 1).expand(1, CLIP, 2, 3, 4, 4).clone(),
            "pts": t.view(1, CLIP, 1).expand(1, CLIP, 2).clone(),
            "frame_timestamps_us": (t * 100).long().view(1, CLIP),
            "clip_text": [[f"f{i}" for i in range(CLIP)]],
            "scene_token": ["s"], "split": ["train"], "fps": torch.tensor([10.0]),
            "d2_render": t.view(1, CLIP, 1, 1, 1, 1).expand(1, CLIP, 2, 17, 2, 2).clone()}


def test_take_window_aligns_batch_and_render():
    exceptions = ["crossview_mask", "scene_token", "split", "dataset_index", "anchor_timestamp_us", "fps"]
    for start in (0, 7, 16):
        batch = fake_clip_batch()
        window, render = take_window(batch, start, exceptions)
        frames = list(range(start, start + WINDOW))
        assert "d2_render" not in window and "d2_render" not in batch
        assert window["vae_images"].shape[1] == WINDOW and render.shape[1] == WINDOW
        assert window["vae_images"][0, :, 0, 0, 0, 0].tolist() == frames
        assert render[0, :, 0, 0, 0, 0].tolist() == frames
        assert window["frame_timestamps_us"][0].tolist() == [100 * f for f in frames]
        assert window["clip_text"][0] == [f"f{f}" for f in frames]
        assert window["scene_token"] == ["s"] and window["fps"].tolist() == [10.0]


# ------------------------------------------------------------------------------- section 43.3: render-init
def test_render_init_input_swaps_only_high_sigma_entries():
    g = torch.Generator().manual_seed(3)
    b, t, v = 1, 4, 2
    x0, eps, r = (torch.randn(b, t, v, 16, 2, 3, generator=g) for _ in range(3))
    timesteps = torch.tensor([[900.0, 600.0, 499.0, 100.0]]).unsqueeze(-1).expand(b, t, v)
    sigma = (timesteps / 1000).view(b, t, v, 1, 1, 1)
    noisy = sigma * eps + (1 - sigma) * x0
    out, used = render_init_input(noisy, x0, timesteps, r, 0.5)
    expect = torch.where(sigma >= 0.5, sigma * eps + (1 - sigma) * r, noisy)
    assert torch.allclose(out, expect, atol=1e-6)
    assert used[0, :, 0, 0, 0, 0].tolist() == [True, True, False, False]


def test_swap_input_builder_restores_and_counts():
    original = CrossviewTemporalSD.__dict__["try_make_input_for_prediction"]
    seen = {}

    def transform(noisy, latents, timesteps):
        seen["called"] = True
        return noisy + 1.0
    with swap_input_builder(transform) as builder:
        assert CrossviewTemporalSD.__dict__["try_make_input_for_prediction"] is not original
        x = torch.zeros(1, 2, 2, 16, 2, 2)
        out = CrossviewTemporalSD.try_make_input_for_prediction(
            x, x.clone(), torch.full((1, 2, 2), 500.0), {"reference_frame_count": 0}, {}, None, None)
    assert seen.get("called") and builder.calls == 1
    assert float(out[0].mean()) == 1.0, "the transformed input did not reach the original builder"
    assert CrossviewTemporalSD.__dict__["try_make_input_for_prediction"] is original
    try:
        with swap_input_builder(transform):
            raise KeyError("boom")
    except KeyError:
        pass
    assert CrossviewTemporalSD.__dict__["try_make_input_for_prediction"] is original, "not restored on error"


def test_render_init_rate():
    draws = [d2_render_init(s, 0, r, 0.5) for s in range(500) for r in range(8)]
    assert 0.46 < sum(draws) / len(draws) < 0.54
    assert not any(d2_render_init(s, 0, 0, 0.0) for s in range(100))


# -------------------------------------------------------------------------------------------- Task 3: dataset
class FakeInner:
    def __init__(self, scenes):
        self.clips, self.clip_audit = [], []
        for token, anchor in scenes:
            self.clips.append([{"timestamp": anchor + 100000 * k} for k in range(CLIP)])
            self.clip_audit.append({"scene_token": token})

    def __getitem__(self, i):
        return {"frame_timestamps_us": torch.tensor([f["timestamp"] for f in self.clips[i]]), "index": i}


class FakeAdapter:
    def __init__(self, inner):
        self.base_dataset = inner

    def __getitem__(self, i):
        return self.base_dataset[i]


def write_render(root, name, token, anchor, shift_us=0, with_pt=True):
    meta = {"scene_token": token, "anchor_timestamp_us": anchor,
            "frame_timestamps_us": [anchor + 100000 * k + shift_us for k in range(CLIP)]}
    if with_pt:
        mask = torch.full((CLIP, 2, 4, 8), 255, dtype=torch.uint8)
        mask[:, :, :, :4] = 0
        torch.save({"render": torch.randn(CLIP, 2, 16, 4, 8).half(), "mask": mask, "meta": meta},
                   root / f"{name}.pt")
    (root / f"{name}.json").write_text(json.dumps(meta))


def test_dataset_pairs_and_checks():
    inner = FakeInner([("a", 1000), ("b", 5000), ("c", 9000)])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_render(root, "clip00000", "a", 1000)
        write_render(root, "clip00001", "b", 5000, shift_us=1)          # one frame timestamp off -> must raise
        write_render(root, "clip00002", "z", 1000)                      # not a base clip -> unmatched
        write_render(root, "clip00003", "c", 9000, with_pt=False)       # json without pt -> skipped
        ds = D2RenderDataset(FakeAdapter(inner), root)
        assert len(ds) == 2 and ds.unmatched == 1, (len(ds), ds.unmatched)
        item = ds[0]
        assert item["index"] == 0
        r = item["d2_render"]
        assert tuple(r.shape) == (CLIP, 2, 17, 4, 8) and r.dtype == torch.float16
        assert float(r[:, :, 16, :, :4].abs().max()) == 0 and float(r[:, :, 16, :, 4:].min()) == 1.0
        try:
            ds[1]
        except ValueError:
            pass
        else:
            raise AssertionError("frame-timestamp mismatch was accepted")
        try:
            D2RenderDataset(FakeAdapter(inner), root, min_items=3)
        except RuntimeError:
            pass
        else:
            raise AssertionError("min_items not enforced")


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {name}\n{traceback.format_exc()}", flush=True)
    print(f"D2_TESTS {len(tests) - failed}/{len(tests)} passed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
