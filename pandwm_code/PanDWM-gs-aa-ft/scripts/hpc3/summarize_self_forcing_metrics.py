#!/usr/bin/env python3
"""Summarize only complete, protocol-matched Self-Forcing evaluation runs."""

import argparse
import json
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator


ARMS = ("baseline", "control", "self")
HORIZONS = ("short", "long")
SEEDS = (0, 42, 1000)


def read_run(root, horizon, arm, seed):
    run = root / f"{horizon}_{arm}_seed{seed}"
    with (run / "eval_configs" / "config.json").open(
            encoding="utf-8") as handle:
        config = json.load(handle)
    info = config["informations"]
    expected_count = 149 if horizon == "short" else 148
    expected_fvd_frames = 16 if horizon == "short" else 64
    assert info["scene_count"] == expected_count
    assert info["fvd_generated_frames"] == expected_fvd_frames
    assert info["schedule"] == [1000, 625, 250, 42]
    assert info["sampler"] == "stochastic"
    assert info["pano_blend_width"] == 128
    assert config["generator_seed"] == seed
    assert config["pipeline"]["metrics"]["fvd"]["sequence_count"] == \
        expected_fvd_frames

    event = EventAccumulator(str(run / "log"))
    event.Reload()
    result = {"seed": seed, "checkpoint": info["checkpoint"]}
    for metric in ("fid", "fvd"):
        entries = event.Scalars(f"evaluation/{metric}")
        if len(entries) != 1 or entries[0].step != 10000:
            raise ValueError(
                f"{run}: expected one evaluation/{metric} at step 10000")
        result[metric] = float(entries[0].value)
    return result


def summarize(root):
    result = {}
    for horizon in HORIZONS:
        result[horizon] = {}
        for arm in ARMS:
            runs = [read_run(root, horizon, arm, seed) for seed in SEEDS]
            item = {"runs": runs}
            for metric in ("fid", "fvd"):
                item[metric] = {
                    "mean": statistics.mean(run[metric] for run in runs),
                    "sample_std": statistics.stdev(
                        run[metric] for run in runs),
                }
            result[horizon][arm] = item
        result[horizon]["self_minus_control"] = {
            metric: {
                "paired_differences": [
                    result[horizon]["self"]["runs"][i][metric] -
                    result[horizon]["control"]["runs"][i][metric]
                    for i in range(len(SEEDS))
                ],
                "mean_difference": (
                    result[horizon]["self"][metric]["mean"] -
                    result[horizon]["control"][metric]["mean"]),
            }
            for metric in ("fid", "fvd")
        }
    return result


def markdown(result):
    lines = [
        "# Self-Forcing 风格训练配对评测",
        "",
        "完整 common149 短期协议与 common148 长期协议分别计算；",
        "短期 FID 为生成 32 帧、FVD 为前 16 帧，长期两者均为 64 帧。",
        "每项为 seeds 0/42/1000 的均值 ± 样本标准差。FID/FVD 越低越好。",
        "",
        "| Horizon | 方法 | FID ↓ | FVD ↓ |",
        "|---|---|---:|---:|",
    ]
    for horizon in HORIZONS:
        for arm in ARMS:
            item = result[horizon][arm]
            lines.append(
                f"| {horizon} | {arm} | "
                f"{item['fid']['mean']:.3f} ± {item['fid']['sample_std']:.3f} | "
                f"{item['fvd']['mean']:.3f} ± {item['fvd']['sample_std']:.3f} |")
        diff = result[horizon]["self_minus_control"]
        lines.extend([
            "",
            f"{horizon} 配对差（Self − Control）："
            f"FID {diff['fid']['mean_difference']:+.3f}，"
            f"FVD {diff['fvd']['mean_difference']:+.3f}。"
            "负值表示 Self-Forcing 风格训练改善。",
            "",
        ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    (args.root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (args.root / "summary.md").write_text(
        markdown(result) + "\n", encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
