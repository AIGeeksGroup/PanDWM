import json
from pathlib import Path

from scripts.hpc3.make_box_track_config import build_config


def _base():
    return {
        "global_state": {"device_mesh": {"mesh_shape": [2, 4]}},
        "generator_seed": 42,
        "optimizer": {"lr": 1e-4},
        "training_dataset": {"base_dataset": {
            "pkl_path": "val.pkl", "rgb_dir": "val/rgb",
            "bad_timestamps_us": [1],
        }},
        "pipeline": {
            "training_config": {},
            "model_checkpoint_path": "old",
        },
    }


def _build(arm):
    return build_config(
        _base(), arm, 7, "student.pth", "teacher.pth", "train.pkl",
        "train/rgb", "bbox", "hdmap", 0.05, 96, 2e-6,
        [1531884295049111])


def test_control_and_track_configs_differ_only_in_registered_arm_fields():
    control = _build("control")
    track = _build("track")
    assert control["pipeline"]["training_config"][
        "box_track_loss_weight"] == 0.0
    assert track["pipeline"]["training_config"][
        "box_track_loss_weight"] == 0.05
    assert control["training_dataset"] == track["training_dataset"]
    assert control["generator_seed"] == track["generator_seed"] == 7
    assert control["informations"]["arm"] == "control"
    assert track["informations"]["arm"] == "track"


def test_config_uses_true_train_paths_and_task_matched_window():
    config = _build("track")
    dataset = config["training_dataset"]["base_dataset"]
    training = config["pipeline"]["training_config"]
    assert dataset["pkl_path"] == "train.pkl"
    assert dataset["rgb_dir"] == "train/rgb"
    assert dataset["bad_timestamps_us"] == [1531884295049111]
    assert config["informations"]["excluded_corrupt_timestamps_us"] == [
        1531884295049111]
    assert dataset["emit_instance_tracks"] is True
    assert dataset["max_instance_tracks"] == 96
    assert dataset["instance_track_min_visibility"] == 2
    assert dataset["instance_track_excluded_classes"] == [
        "barrier", "traffic_cone"]
    assert training["reference_frame_count"] == 3
    assert training["generation_task_ratio"] == 0.0
    assert training["image_generation_ratio"] == 0.0
    assert training["all_reference_visible_ratio"] == 1.0
    assert config["pipeline"]["consistency_teacher_checkpoint_path"] == \
        "teacher.pth"
