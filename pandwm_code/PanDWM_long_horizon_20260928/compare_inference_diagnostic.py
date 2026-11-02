"""Pair a generated-history diagnostic with the exact clean B0 scene/seed."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from run_eval_task import task_directory_name


def window_values(path):
    record = json.loads(path.read_text())
    if len(record["windows"]) != 12:
        raise ValueError(f"not 12 windows: {path}")
    return record, np.array([w["psnr_mean_db"] for w in record["windows"]])


def compare(tasks, baseline_root, diagnostic_root):
    rows, missing = [], []
    for task in tasks:
        baseline = (baseline_root /
                    f"baseline_clean_scene{task['scene_index']}_seed{task['seed']}" /
                    "summary.json")
        candidate = diagnostic_root / task_directory_name(task) / "summary.json"
        if not baseline.is_file() or not candidate.is_file():
            missing.append({"baseline": str(baseline), "candidate": str(candidate)})
            continue
        b, bx = window_values(baseline)
        c, cx = window_values(candidate)
        if (b["scene_token"] != task["scene_token"] or
                c["scene_token"] != task["scene_token"] or
                b["seed"] != task["seed"] or c["seed"] != task["seed"]):
            raise ValueError(f"scene/seed mismatch: {candidate}")
        if b["windows"][0]["psnr_per_frame_db"] != \
                c["windows"][0]["psnr_per_frame_db"]:
            raise ValueError(f"first-window RNG regression: {candidate}")
        delta = cx - bx
        rows.append({"scene_token": task["scene_token"],
                     "scene_index": task["scene_index"], "seed": task["seed"],
                     "overall": float(delta.mean()),
                     "early": float(delta[:2].mean()),
                     "late": float(delta[8:].mean()),
                     "window_deltas": delta.tolist()})
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene_token"]].append(row)
    per_scene = {scene: {
        "seeds": [r["seed"] for r in group],
        "overall": float(np.mean([r["overall"] for r in group])),
        "early": float(np.mean([r["early"] for r in group])),
        "late": float(np.mean([r["late"] for r in group]))}
        for scene, group in by_scene.items()}
    summary = {}
    if per_scene:
        rng = np.random.default_rng(20260929)
        for key in ("overall", "early", "late"):
            values = np.array([s[key] for s in per_scene.values()])
            bootstrap = rng.choice(values, size=(10000, len(values)), replace=True)
            summary[key] = {
                "mean": float(values.mean()),
                "positive_scenes": int((values > 0).sum()),
                "scene_bootstrap_ci95": np.quantile(
                    bootstrap.mean(axis=1), [0.025, 0.975]).tolist()}
        summary["per_window_mean"] = np.mean(
            [r["window_deltas"] for r in rows], axis=0).tolist()
    return {"expected": len(tasks), "complete": len(rows), "missing": missing,
            "per_task": rows, "per_scene": per_scene, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text())["tasks"]
    result = compare(tasks, args.baseline_root, args.diagnostic_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": result["complete"],
                      "expected": result["expected"],
                      "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
