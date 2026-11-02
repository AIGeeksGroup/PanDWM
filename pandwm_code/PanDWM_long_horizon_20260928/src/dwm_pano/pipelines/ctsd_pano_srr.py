"""Small Stage-1 teacher recovery pilot on its own cached rollout history.

The target window's future remains real RGB encoded by the native Stage-1
flow-matching train_step.  Only the three visible reference latents change.
"""

from pathlib import Path
import hashlib
import json

import torch

from dwm_pano.generated_latent_cache import scalar_int
from dwm_pano.pipelines.ctsd_pano import PanoCrossviewTemporalSD
from dwm_pano.pipelines.ctsd_pano_self_forcing_utils import slice_sequence_batch


def history_cache_path(cache_dir, dataset_index, anchor_timestamp_us, target_window):
    return Path(cache_dir) / (
        f"{int(dataset_index):08d}_{int(anchor_timestamp_us)}_"
        f"w{int(target_window):02d}.pt")


def target_window_for_step(step, seed, max_target_window):
    """Arm-independent schedule; neither arm consumes a random draw here."""
    digest = hashlib.sha256(f"{seed}:{step}".encode()).digest()
    return 1 + int.from_bytes(digest[:8], "little") % max_target_window


def use_generated_history_for_step(step, seed, ratio):
    """Arm-independent Bernoulli(ratio) draw; consumes no model RNG."""
    if ratio >= 1:
        return True
    if ratio <= 0:
        return False
    digest = hashlib.sha256(f"{seed}:{step}:history".encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2 ** 64 < ratio


def validate_history_payload(payload, batch, target_window, checkpoint):
    if payload.get("schema_version") != 1 or payload.get("source") != "stage1_teacher_50step_closed_loop":
        raise RuntimeError("not a Stage-1 closed-loop history cache")
    if payload.get("checkpoint") != checkpoint:
        raise RuntimeError("cache checkpoint differs from teacher initialization")
    expected = {
        "dataset_index": scalar_int(batch["dataset_index"]),
        "anchor_timestamp_us": scalar_int(batch["anchor_timestamp_us"]),
        "target_window": target_window,
        "scene_token": batch["scene_token"][0],
        "split": "train",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"cache {key}: {payload.get(key)!r} != {value!r}")
    start = target_window * 16
    times = batch["frame_timestamps_us"][0, start:start + 3].cpu()
    if not torch.equal(payload["history_timestamps_us"], times):
        raise RuntimeError("cache history timestamps do not match current clip")
    history = payload["generated_history"]
    if not isinstance(history, torch.Tensor) or history.ndim != 5 or \
            tuple(history.shape[:2]) != (3, 2) or \
            not history.is_floating_point() or not bool(torch.isfinite(history).all()):
        raise RuntimeError("invalid generated history latent")
    return history


class PanoCrossviewTemporalSDRecovery(PanoCrossviewTemporalSD):
    def __init__(self, *args, srr_config=None, **kwargs):
        self.srr_config = dict(srr_config or {})
        self.srr_arm = self.srr_config.get("arm", "control")
        if self.srr_arm not in ("control", "recovery"):
            raise ValueError("srr arm must be control or recovery")
        self.srr_max_target_window = int(
            self.srr_config.get("max_target_window", 1))
        if self.srr_max_target_window not in (1, 2, 3):
            raise ValueError("67-frame pilot supports target windows 1..3")
        self.srr_cache_dir = self.srr_config.get("cache_dir")
        self.srr_checkpoint = self.srr_config.get("checkpoint")
        self.srr_history_end_blend = float(
            self.srr_config.get("history_end_blend", 0.0))
        if not 0 <= self.srr_history_end_blend <= 1:
            raise ValueError("history_end_blend must be in [0,1]")
        # Fraction of recovery-arm steps that read the cached generated
        # history; the rest use GT history exactly like the control arm.
        self.srr_recovery_ratio = float(
            self.srr_config.get("recovery_ratio", 1.0))
        if not 0 < self.srr_recovery_ratio <= 1:
            raise ValueError("recovery_ratio must be in (0,1]")
        self.srr_window_curriculum = self.srr_config.get("window_curriculum")
        if self.srr_window_curriculum is not None:
            for phase in self.srr_window_curriculum:
                if int(phase["target_window"]) not in (1, 2, 3):
                    raise ValueError("invalid curriculum target window")
        manifest_path = self.srr_config.get("forbidden_scene_manifest")
        if not manifest_path:
            raise ValueError("SRR pilot requires frozen dev/heldout manifest")
        manifest = json.loads(Path(manifest_path).read_text())
        self.srr_forbidden_scenes = {
            item["scene_token"] for key in ("dev", "heldout")
            for item in manifest[key]
        }
        if self.srr_arm == "recovery" and not self.srr_cache_dir:
            raise ValueError("recovery arm needs cache_dir")
        super().__init__(*args, **kwargs)

    def train_step(self, batch, global_step):
        if tuple(batch["vae_images"].shape[:3])[1:] != (67, 2):
            raise ValueError("SRR pilot expects 67-frame, two-view clips")
        if batch["split"][0] != "train":
            raise ValueError("SRR cache may only use train split")
        if batch["scene_token"][0] in self.srr_forbidden_scenes:
            raise ValueError("SRR training clip overlaps dev/heldout")
        if self.srr_window_curriculum is None:
            window = target_window_for_step(
                global_step, self.config["generator_seed"],
                self.srr_max_target_window)
        else:
            phases = self.srr_window_curriculum
            window = int(next((phase["target_window"] for phase in phases
                               if global_step < int(phase["until_step"])),
                              phases[-1]["target_window"]))
        start = 16 * window
        exceptions = self.inference_config.get(
            "autoregression_data_exception_for_take_sequence", [])
        target_batch = slice_sequence_batch(batch, start, 19, exceptions)
        if self.srr_arm == "recovery" and use_generated_history_for_step(
                global_step, self.config["generator_seed"],
                self.srr_recovery_ratio):
            cache_path = history_cache_path(
                self.srr_cache_dir, scalar_int(batch["dataset_index"]),
                scalar_int(batch["anchor_timestamp_us"]), window)
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            history = validate_history_payload(
                payload, batch, window, self.srr_checkpoint)
            target_batch["_srr_generated_history"] = history.unsqueeze(0)
            target_batch["_srr_history_end_blend"] = \
                self.srr_history_end_blend
        # Both arms use the identical native flow matching target, masks,
        # timesteps, data order and model RNG sequence.
        return super().train_step(target_batch, global_step)
