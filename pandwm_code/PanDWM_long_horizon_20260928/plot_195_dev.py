"""Plot native panorama PSNR by window without mixing evaluation protocols."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_values(root, name):
    item = json.loads((root / name / "summary.json").read_text())
    return np.array([w["psnr_mean_db"] for w in item["windows"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dev = json.loads(args.split.read_text())["dev"]
    scenes = [s["index"] for s in dev]
    baseline = np.array([
        [load_values(args.baseline_root,
                     f"baseline_clean_scene{s}_seed{seed}")
         for seed in (0, 42, 1000)] for s in scenes])
    baseline_by_scene = baseline.mean(axis=1)
    control = np.array([load_values(
        args.baseline_root, f"control_clean_scene{s}_seed42")
        for s in scenes])
    self_model = np.array([load_values(
        args.baseline_root, f"self_clean_scene{s}_seed42")
        for s in scenes])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=180)
    x = np.arange(1, 13)
    for name, values, color in (("B0 (3 seeds)", baseline_by_scene, "#1f77b4"),
                                ("paired control (seed 42)", control, "#2ca02c"),
                                ("Self-inspired (seed 42)", self_model, "#d62728")):
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
        axes[0].plot(x, mean, marker="o", markersize=3, label=name, color=color)
        axes[0].fill_between(x, mean - sem, mean + sem, color=color, alpha=0.12)
    axes[0].set(title="195-frame closed-loop rollout", xlabel="16-frame window",
                ylabel="Panorama PSNR (dB)", xticks=x)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    old_scenes = [14, 32, 49]
    clean = np.array([load_values(
        args.baseline_root, f"baseline_clean_scene{s}_seed42")
        for s in old_scenes])
    for mode, color in (("flow_noise_0.020", "#ff7f0e"),
                        ("flow_noise_0.050", "#9467bd"),
                        ("vae_roundtrip", "#8c564b"),
                        ("moment_align_0.500", "#17becf")):
        values = np.array([load_values(
            args.diagnostic_root, f"baseline_{mode}_scene{s}_seed42")
            for s in old_scenes]) - clean
        axes[1].plot(x, values.mean(axis=0), marker="o", markersize=3,
                     label=mode, color=color)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(title="History diagnostics vs B0 clean (3 scenes, seed 42)",
                xlabel="16-frame window", ylabel="Paired PSNR change (dB)",
                xticks=x)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
