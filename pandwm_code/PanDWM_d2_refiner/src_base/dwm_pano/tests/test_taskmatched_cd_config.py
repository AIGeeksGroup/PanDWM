import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location(
    "make_cd_taskmatched_config",
    ROOT / "scripts/h20/make_cd_taskmatched_config.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_config = MODULE.build_config


def test_taskmatched_config_uses_exact_deployment_task():
    path = ROOT / "configs/ctsd/pandwm/h20/cd_eval_h20.json"
    base = json.loads(path.read_text(encoding="utf-8"))
    teacher = "/tmp/stage1.pth"
    config = build_config(base, teacher, mesh_size=4, eval_items=5)
    pipeline = config["pipeline"]
    training = pipeline["training_config"]

    assert config["global_state"]["device_mesh"]["mesh_shape"] == [1, 4]
    assert config["training_dataset"]["base_dataset"]["sequence_length"] == 19
    assert training["reference_frame_count"] == 3
    assert training["generation_task_ratio"] == 0.0
    assert training["image_generation_ratio"] == 0.0
    assert training["all_reference_visible_ratio"] == 1.0
    assert training["reference_visible_rate"] == 1.0
    assert pipeline["model_checkpoint_path"] == teacher
    assert pipeline["consistency_teacher_checkpoint_path"] == teacher
    assert pipeline["consistency_ema_checkpoint_path"] == teacher
