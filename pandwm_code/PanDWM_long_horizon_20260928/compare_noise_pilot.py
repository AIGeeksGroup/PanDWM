"""Compare a noise-trained model with its exact continuation control."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from run_eval_task import task_directory_name


def windows(path):
    record = json.loads(path.read_text())
    values = np.array([w["psnr_mean_db"] for w in record["windows"]])
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ValueError(f"invalid 12-window PSNR in {path}")
    return record, values


def bootstrap_scenes(rows):
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene_token"]].append(row)
    per_scene = {scene: np.mean([row["delta"] for row in group], axis=0).tolist()
                 for scene, group in by_scene.items()}
    scene_values = np.array(list(per_scene.values()))
    rng = np.random.default_rng(20260929)
    sample = rng.choice(len(scene_values),
                        size=(10000, len(scene_values)), replace=True)
    distribution = scene_values[sample].mean(axis=1)
    return {
        "n_scenes": len(per_scene),
        "n_scene_seed_pairs": len(rows),
        "mean_by_window": scene_values.mean(axis=0).tolist(),
        "overall_mean_db": float(scene_values.mean(axis=1).mean()),
        "early_mean_db": float(scene_values[:, :2].mean(axis=1).mean()),
        "late_mean_db": float(scene_values[:, 8:].mean(axis=1).mean()),
        "late_positive_scenes": int((scene_values[:, 8:].mean(axis=1) > 0).sum()),
        "ci95_overall_db": np.quantile(distribution.mean(axis=1),
                                       [0.025, 0.975]).tolist(),
        "ci95_late_db": np.quantile(distribution[:, 8:].mean(axis=1),
                                    [0.025, 0.975]).tolist(),
        "per_scene_by_window": per_scene,
    }


def compare(tasks, result_root, baseline_root):
    measurements, missing = {}, []
    for task in tasks:
        path = result_root / task_directory_name(task) / "summary.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        record, values = windows(path)
        if record["scene_token"] != task["scene_token"] or \
                record["seed"] != task["seed"]:
            raise ValueError(f"scene/seed mismatch: {path}")
        key = (task["scene_token"], task["scene_index"], task["seed"],
               task["history_mode"], float(task["noise_sigma"]), task["arm"])
        if key in measurements:
            raise ValueError(f"duplicate task: {key}")
        measurements[key] = (record, values)

    comparisons = defaultdict(list)
    complete_triplets = 0
    for task in tasks:
        if task["arm"] != "noise_control":
            continue
        key = (task["scene_token"], task["scene_index"], task["seed"],
               task["history_mode"], float(task["noise_sigma"]))
        control = measurements.get((*key, "noise_control"))
        noise = measurements.get((*key, "noise"))
        if control is None or noise is None:
            continue
        complete_triplets += 1
        comparisons[f"noise_minus_control_{task['history_mode']}_{task['noise_sigma']:.3f}"].append(
            {"scene_token": task["scene_token"], "seed": task["seed"],
             "delta": noise[1] - control[1]})
        if task["history_mode"] == "clean":
            path = (baseline_root /
                    f"baseline_clean_scene{task['scene_index']}_seed{task['seed']}" /
                    "summary.json")
            baseline, bvalues = windows(path)
            if baseline["scene_token"] != task["scene_token"] or \
                    baseline["seed"] != task["seed"]:
                raise ValueError(f"baseline mismatch: {path}")
            for arm, values in (("control", control[1]), ("noise", noise[1])):
                comparisons[f"{arm}_minus_B0_clean"].append({
                    "scene_token": task["scene_token"], "seed": task["seed"],
                    "delta": values - bvalues})

    for task in tasks:
        if task["history_mode"] != "clean":
            continue
        base = (task["scene_token"], task["scene_index"], task["seed"])
        clean = measurements.get((*base, "clean", 0.0, task["arm"]))
        flow = measurements.get((*base, "flow_noise", 0.02, task["arm"]))
        if clean is None or flow is None:
            continue
        if (clean[0]["windows"][0]["psnr_per_frame_db"] !=
                flow[0]["windows"][0]["psnr_per_frame_db"]):
            raise ValueError(f"first-window RNG regression: {base}")
        comparisons[f"{task['arm']}_flow_minus_clean"].append({
            "scene_token": task["scene_token"], "seed": task["seed"],
            "delta": flow[1] - clean[1]})

    return {
        "expected": len(tasks), "complete": len(measurements),
        "missing": missing, "complete_arm_pairs": complete_triplets,
        "comparisons": {
            label: bootstrap_scenes(rows) for label, rows in comparisons.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text())["tasks"]
    result = compare(tasks, args.result_root, args.baseline_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "complete": result["complete"], "expected": result["expected"],
        "comparisons": {name: {
            "n_scene_seed_pairs": x["n_scene_seed_pairs"],
            "overall_mean_db": x["overall_mean_db"],
            "late_mean_db": x["late_mean_db"],
            "ci95_late_db": x["ci95_late_db"],
        } for name, x in result["comparisons"].items()}
    }, indent=2))


if __name__ == "__main__":
    main()
