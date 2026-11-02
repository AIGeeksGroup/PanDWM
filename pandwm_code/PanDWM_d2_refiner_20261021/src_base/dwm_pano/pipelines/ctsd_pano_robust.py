"""Stage-1 teacher fine-tune v3 (2026-10-15): recovery from the CF-DMD student's own history.

Why (PanDWM-cfdmd-A-20261008/eval_v7/V82_EVAL_HPC3_20261011.md sections 27.10-27.12): the student's late-window
drift is a per-window bias that it inherits from the teacher (both drift the same way), so the DMD target itself
has to improve.  The teacher learns to predict the REAL target window from the history the student actually
produces (HorizonDrive's self-rollout recovery, with the student's closed loop in place of the teacher's, since the
DMD real score is always evaluated on student histories).

Every step trains the native Stage-1 flow-matching loss on ONE 19-frame target window of a train clip of
16 W + 19 frames: target window w in 0..W (frames 16w .. 16w+18), drawn by an arm-independent hash of (seed, step).
The 16 future frames are always the GT latents.  The 3 visible history frames are
  - w = 0: the clip's first 3 GT frames;
  - w >= 1, arm "control": the GT frames 16w .. 16w+2;
  - w >= 1, arm "robust": with probability recovery_ratio (hash of (seed, step)) the student's closed-loop history
    for this clip and window (eval_v7/cache_student_history.py), otherwise the GT frames.
History noise (training_config["history_noise"]) and the time/weather text (the dataset's time_weather_source) are
separate config switches; both act on every history source alike.  Both arms see the same clips, windows, model
RNG draws and target noise; only the history source (and the switches) differ.
"""
import json
from pathlib import Path

import torch

from dwm_pano.pipelines.ctsd_pano import PanoCrossviewTemporalSD
from dwm_pano.pipelines.ctsd_pano_self_forcing_utils import slice_sequence_batch
from dwm_pano.pipelines.ctsd_pano_srr import (target_window_for_step,
                                              use_generated_history_for_step)

STUDENT_CACHE_SOURCE = "cfdmd_student_4step_closed_loop"


def student_cache_path(cache_dir, dataset_index, anchor_timestamp_us, target_window):
    return Path(cache_dir) / f"{int(dataset_index):08d}_{int(anchor_timestamp_us)}_w{int(target_window):02d}.pt"


def ramp_weights(length, n_reference, ramp):
    """(length,) loss weights: 0 on the references, j / ramp on target frame j < ramp (j = 0 right after the
    history), 1 afterwards."""
    w = torch.ones(length)
    w[:n_reference] = 0.0
    for j in range(min(ramp, length - n_reference)):
        w[n_reference + j] = j / ramp
    return w


def _scalar(x):
    return int(x.reshape(-1)[0].item()) if isinstance(x, torch.Tensor) else int(x)


def validate_student_history(payload, batch, target_window, checkpoint):
    """The cached history must belong to exactly this clip and window."""
    if payload.get("schema_version") != 1 or payload.get("source") != STUDENT_CACHE_SOURCE:
        raise RuntimeError("not a CF-DMD student closed-loop history cache")
    if checkpoint and payload.get("checkpoint") != checkpoint:
        raise RuntimeError(f"cache checkpoint {payload.get('checkpoint')} != {checkpoint}")
    expected = {"dataset_index": _scalar(batch["dataset_index"]),
                "anchor_timestamp_us": _scalar(batch["anchor_timestamp_us"]),
                "target_window": int(target_window), "scene_token": batch["scene_token"][0], "split": "train"}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"cache {key}: {payload.get(key)!r} != {value!r}")
    start = 16 * target_window
    times = batch["frame_timestamps_us"][0, start:start + 3].cpu()
    if not torch.equal(payload["history_timestamps_us"], times):
        raise RuntimeError("cache history timestamps do not match the clip")
    history = payload["generated_history"]
    if (not isinstance(history, torch.Tensor) or history.ndim != 5 or tuple(history.shape[:2]) != (3, 2)
            or not history.is_floating_point() or not bool(torch.isfinite(history).all())):
        raise RuntimeError("invalid generated history latent")
    return history


class PanoCrossviewTemporalSDRobust(PanoCrossviewTemporalSD):
    def __init__(self, *args, robust_config=None, **kwargs):
        rc = dict(robust_config or {})
        self.robust_arm = rc.get("arm", "control")
        if self.robust_arm not in ("control", "robust"):
            raise ValueError("robust arm must be control or robust")
        self.robust_max_window = int(rc.get("max_target_window", 4))
        if not 1 <= self.robust_max_window <= 6:
            raise ValueError("max_target_window must be 1..6")
        self.robust_cache_dir = rc.get("cache_dir")
        self.robust_cache_checkpoint = rc.get("cache_checkpoint")
        self.robust_recovery_ratio = float(rc.get("recovery_ratio", 0.5))
        if not 0 <= self.robust_recovery_ratio <= 1:
            raise ValueError("recovery_ratio must be in [0, 1]")
        if self.robust_arm == "robust" and self.robust_recovery_ratio > 0 and not self.robust_cache_dir:
            raise ValueError("the robust arm needs cache_dir")
        # "error" (the pilot): a missing cache entry stops training; "gt" (smoke tests only): fall back to the GT
        # history and count the miss.
        self.robust_missing_cache = rc.get("missing_cache", "error")
        # v3b: on samples with the student's history, the loss weight of the first `loss_ramp_frames` target frames
        # rises linearly from 0 (frame 3, right after the history) to 1, so the model may stay continuous with the
        # generated history at the boundary and pull toward the GT content gradually; 0 = the v3 pilot (no ramp).
        self.robust_loss_ramp = int(rc.get("loss_ramp_frames", 0))
        if not 0 <= self.robust_loss_ramp <= 16:
            raise ValueError("loss_ramp_frames must be 0..16")
        if self.robust_missing_cache not in ("error", "gt"):
            raise ValueError("missing_cache must be error or gt")
        manifest_path = rc.get("forbidden_scene_manifest")
        if not manifest_path:
            raise ValueError("the teacher fine-tune needs the frozen dev/heldout manifest")
        manifest = json.loads(Path(manifest_path).read_text())
        self.robust_forbidden_scenes = {item["scene_token"] for key in ("dev", "heldout") for item in manifest[key]}
        self.robust_counts = {"steps": 0, "student_history": 0, "cache_miss": 0}
        super().__init__(*args, **kwargs)

    def train_step(self, batch, global_step):
        need = 16 * self.robust_max_window + 19
        if tuple(batch["vae_images"].shape[1:3]) != (need, 2):
            raise ValueError(f"expected {need}-frame two-view clips, got {tuple(batch['vae_images'].shape)}")
        if batch["split"][0] != "train":
            raise ValueError("the teacher fine-tune reads the train split only")
        if batch["scene_token"][0] in self.robust_forbidden_scenes:
            raise ValueError("training clip overlaps dev/heldout")
        seed = self.config["generator_seed"]
        window = target_window_for_step(global_step, seed, self.robust_max_window + 1) - 1   # 0..W
        exceptions = self.inference_config.get("autoregression_data_exception_for_take_sequence", [])
        target_batch = slice_sequence_batch(batch, 16 * window, 19, exceptions)
        self.robust_counts["steps"] += 1
        self._robust_last = {"window": window, "student_history": 0}
        if (self.robust_arm == "robust" and window >= 1 and self.robust_recovery_ratio > 0
                and use_generated_history_for_step(global_step, seed, self.robust_recovery_ratio)):
            path = student_cache_path(self.robust_cache_dir, _scalar(batch["dataset_index"]),
                                      _scalar(batch["anchor_timestamp_us"]), window)
            if not path.is_file() and self.robust_missing_cache == "gt":
                self.robust_counts["cache_miss"] += 1
            else:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                history = validate_student_history(payload, batch, window, self.robust_cache_checkpoint)
                target_batch["_srr_generated_history"] = history.unsqueeze(0)
                if self.robust_loss_ramp > 0:
                    target_batch["_loss_frame_weights"] = ramp_weights(19, 3, self.robust_loss_ramp)
                self.robust_counts["student_history"] += 1
                self._robust_last["student_history"] = 1
        if global_step % 10 == 0 and getattr(self, "should_save", True):
            print(f"ROBUST step={global_step} arm={self.robust_arm} window={window} "
                  f"student_history={self._robust_last['student_history']} "
                  f"noise_sigma={getattr(self, '_last_history_noise_sigma', None)} counts={self.robust_counts}",
                  flush=True)
        return super().train_step(target_batch, global_step)
