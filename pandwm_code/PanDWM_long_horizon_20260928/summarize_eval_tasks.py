"""Audit array outputs and report scene-paired 12-window differences."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from run_eval_task import task_directory_name


def arm_key(task):
    mode = task["history_mode"]
    if mode == "flow_noise":
        mode += f"_{task['noise_sigma']:.3f}"
    if mode == "moment_align":
        mode += f"_{task['align_alpha']:.3f}"
    return task["arm"], mode


def read_one(root, task):
    path = root / task_directory_name(task) / "summary.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    if (result["scene_token"] != task["scene_token"] or
            result["seed"] != task["seed"] or
            len(result["windows"]) != 12):
        raise ValueError(f"summary protocol mismatch: {path}")
    means = np.array([w["psnr_mean_db"] for w in result["windows"]])
    for j, w in enumerate(result["windows"]):
        if w["window"] != j + 1 or len(w["psnr_per_frame_db"]) != 16:
            raise ValueError(f"invalid window or frame count: {path}")
    return {"scene_token": task["scene_token"], "seed": task["seed"],
            "arm_key": arm_key(task), "overall": float(means.mean()),
            "early": float(means[:2].mean()),
            "middle": float(means[3:8].mean()),
            "late": float(means[8:12].mean()),
            "windows": means.tolist(), "path": str(path)}


def paired_delta(rows, candidate, control):
    # Collapse seeds within each scene before bootstrapping scenes.
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["arm_key"], row["scene_token"], row["seed"])].append(row)
    scene_deltas = defaultdict(list)
    for row in rows:
        if row["arm_key"] != candidate:
            continue
        match = buckets.get((control, row["scene_token"], row["seed"]))
        if not match:
            continue
        if len(match) != 1:
            raise ValueError("duplicate control result")
        scene_deltas[row["scene_token"]].append({
            key: row[key] - match[0][key]
            for key in ("overall", "early", "middle", "late")})
    if not scene_deltas:
        return None
    per_scene = {scene: {key: float(np.mean([d[key] for d in values]))
                         for key in ("overall", "early", "middle", "late")}
                 for scene, values in scene_deltas.items()}
    result = {"candidate": candidate, "control": control,
              "paired_scenes": len(per_scene), "per_scene": per_scene}
    rng = np.random.default_rng(20260928)
    for key in ("overall", "early", "middle", "late"):
        values = np.array([entry[key] for entry in per_scene.values()])
        sampled = rng.choice(values, size=(10000, len(values)), replace=True)
        result[key] = {"mean": float(values.mean()),
                       "bootstrap_scene_ci95": np.quantile(
                           sampled.mean(axis=1), [0.025, 0.975]).tolist()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text())["tasks"]
    rows = []
    missing = []
    for task in tasks:
        result = read_one(args.results_root, task)
        if result is None:
            missing.append(task_directory_name(task))
        else:
            rows.append(result)
    groups = defaultdict(list)
    for row in rows:
        groups[row["arm_key"]].append(row)
    group_summary = {"/".join(key): {
        "count": len(values),
        **{metric: float(np.mean([r[metric] for r in values]))
           for metric in ("overall", "early", "middle", "late")}}
        for key, values in groups.items()}
    pairs = []
    for candidate, control in ((('self', 'clean'), ('control', 'clean')),
                               (('noise', 'clean'), ('noise_control', 'clean')),
                               (('noise', 'flow_noise_0.020'),
                                ('noise_control', 'flow_noise_0.020'))):
        result = paired_delta(rows, candidate, control)
        if result is not None:
            pairs.append(result)
    output = {"expected": len(tasks), "complete": len(rows),
              "missing": missing, "group_summary": group_summary,
              "paired_comparisons": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(rows)}/{len(tasks)} complete; {len(missing)} missing")
    print(json.dumps({"group_summary": group_summary,
                      "paired_comparisons": pairs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
