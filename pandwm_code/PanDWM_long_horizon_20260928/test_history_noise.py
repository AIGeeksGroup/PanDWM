import json
from pathlib import Path

import torch

from dwm_pano.history_noise import (
    flow_corrupt_reference_context, make_history_for_window)
from summarize_eval_tasks import paired_delta


def test_noise_preserves_future_and_matches_labeled_flow_sigma():
    latents = torch.full((2, 4, 1, 1, 1, 1), 0.75)
    noise = torch.full_like(latents, -0.25)
    visible = torch.zeros(2, 4, 1, 1, 1, 1, dtype=torch.bool)
    visible[:, :2] = True
    sigma = torch.tensor([0.0, 0.2]).view(2, 1, 1, 1, 1, 1)
    result = flow_corrupt_reference_context(latents, visible, noise, sigma)
    torch.testing.assert_close(result[0], latents[0])
    torch.testing.assert_close(result[1, :2],
                               torch.full_like(result[1, :2], 0.55))
    torch.testing.assert_close(result[:, 2:], latents[:, 2:])


def test_eval_tasks_never_touch_heldout_scenes():
    root = Path(__file__).parent
    manifest = root.parent / "PanDWM-history-probe-20260928/results/long_rollout/split_195_scene_manifest.json"
    split = json.loads(manifest.read_text())
    allowed = {scene["scene_token"] for scene in split["dev"]}
    forbidden = {scene["scene_token"] for scene in split["heldout"]}
    for filename in ("tasks_baseline_dev195.json", "tasks_diagnostic_dev195.json",
                     "tasks_diagnostic_validate_dev195.json",
                     "tasks_noise_pilot_dev195.json"):
        tasks = json.loads((root / filename).read_text())["tasks"]
        tokens = {task["scene_token"] for task in tasks}
        assert tokens <= allowed
        assert not tokens & forbidden


def test_inference_noise_is_reproducible_and_keeps_initial_observation_clean():
    reference = torch.ones(1, 3, 2, 4, 2, 2)
    initial_stats = (torch.ones(1, 1, 1, 4, 1, 1),
                     torch.ones(1, 1, 1, 4, 1, 1))
    initial, t0 = make_history_for_window(
        reference, initial_stats, "flow_noise", 0.02, 0.5, 42, 0)
    assert initial is reference and t0 == 0
    global_rng_before = torch.get_rng_state().clone()
    a, ta = make_history_for_window(
        reference, initial_stats, "flow_noise", 0.02, 0.5, 42, 1)
    torch.testing.assert_close(torch.get_rng_state(), global_rng_before)
    b, tb = make_history_for_window(
        reference, initial_stats, "flow_noise", 0.02, 0.5, 42, 1)
    torch.testing.assert_close(a, b)
    assert ta == tb == 20.0
    assert not torch.equal(a, reference)


def test_paired_summary_uses_scene_as_unit_after_averaging_seeds():
    rows = []
    for scene, seed, control, candidate in (
            ("a", 0, 10, 11), ("a", 42, 10, 13),
            ("b", 0, 10, 9), ("b", 42, 10, 9)):
        for arm, value in (("noise_control", control), ("noise", candidate)):
            rows.append({"arm_key": (arm, "clean"), "scene_token": scene,
                         "seed": seed, "overall": value,
                         "early": value, "middle": value, "late": value})
    result = paired_delta(rows, ("noise", "clean"),
                          ("noise_control", "clean"))
    assert result["paired_scenes"] == 2
    assert result["per_scene"]["a"]["late"] == 2
    assert result["per_scene"]["b"]["late"] == -1
    assert result["late"]["mean"] == 0.5
