"""Summarize matched G2 adaptation metrics and apply explicit quality gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dwm_pano.evaluate_gs import atomic_json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=METRICS_JSON",
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def headline(metrics: dict[str, Any]) -> dict[str, float]:
    future = metrics["future_selected_frames"]
    context = metrics["per_frame"]["rgb"]["stage2_gs_vs_gt"]["2"]
    context_no_gs = metrics["per_frame"]["rgb"]["no_gs_vs_gt"]["2"]
    return {
        "future_psnr": future["rgb"]["stage2_gs_vs_gt"]["PSNR"],
        "future_ssim": future["rgb"]["stage2_gs_vs_gt"]["SSIM"],
        "future_lpips": future["rgb"]["stage2_gs_vs_gt"]["LPIPS"],
        "future_gs_minus_no_gs_psnr":
            future["gs_minus_no_gs_against_gt"]["PSNR"],
        "future_gs_minus_no_gs_lpips":
            future["gs_minus_no_gs_against_gt"]["LPIPS"],
        "context_psnr_cost": context_no_gs["PSNR"] - context["PSNR"],
        "context_lpips_cost": context["LPIPS"] - context_no_gs["LPIPS"],
        "clean_future_depth_l1":
            future["depth"]["gt_latent_gs_vs_gt"]["DepthL1"],
        "generated_future_depth_l1":
            future["depth"]["stage2_gs_vs_gt"]["DepthL1"],
    }


def evaluate_arm(
    values: dict[str, float], baseline: dict[str, float]
) -> dict[str, Any]:
    gates = {
        "context_psnr_cost_le_1p5db": values["context_psnr_cost"] <= 1.5,
        "future_lpips_gap_le_0p01":
            values["future_gs_minus_no_gs_lpips"] <= 0.01,
        "future_psnr_gap_ge_minus_0p2db":
            values["future_gs_minus_no_gs_psnr"] >= -0.2,
        "clean_depth_within_5pct": values["clean_future_depth_l1"]
            <= baseline["clean_future_depth_l1"] * 1.05,
    }
    delta = {key: values[key] - baseline[key] for key in values}
    # Lower score is better.  It prioritizes closing the no-GS/GS appearance
    # gap, with soft penalties for context damage and clean-latent depth drift.
    score = (
        10.0 * values["future_gs_minus_no_gs_lpips"]
        + max(0.0, -values["future_gs_minus_no_gs_psnr"])
        + 0.2 * max(0.0, values["context_psnr_cost"])
        + max(
            0.0,
            values["clean_future_depth_l1"]
            / baseline["clean_future_depth_l1"] - 1.0,
        )
    )
    return {
        "metrics": values,
        "delta_vs_g2_2500": delta,
        "gates": gates,
        "passed_all_gates": all(gates.values()),
        "selection_score_lower_is_better": score,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# G2 adaptation matched evaluation",
        "",
        "| arm | future PSNR | future LPIPS | GS−noGS PSNR | "
        "GS−noGS LPIPS | context PSNR cost | clean depth L1 | gates |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    rows = {"g2_2500": {"metrics": payload["baseline"]}}
    rows.update(payload["arms"])
    for name, result in rows.items():
        values = result["metrics"]
        gate = "baseline" if name == "g2_2500" else (
            "PASS" if result["passed_all_gates"] else "FAIL")
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:+.4f} | {:+.4f} | {:.4f} | "
            "{:.4f} | {} |".format(
                name,
                values["future_psnr"],
                values["future_lpips"],
                values["future_gs_minus_no_gs_psnr"],
                values["future_gs_minus_no_gs_lpips"],
                values["context_psnr_cost"],
                values["clean_future_depth_l1"],
                gate,
            )
        )
    if payload.get("best_arm"):
        lines.extend(["", f"Current best arm: **{payload['best_arm']}**"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    baseline = headline(load_metrics(args.baseline))
    arms = {}
    for specification in args.arm:
        if "=" not in specification:
            raise ValueError(f"Invalid --arm: {specification}")
        name, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        if path.is_file():
            arms[name] = evaluate_arm(headline(load_metrics(path)), baseline)
    best_arm = None
    if arms:
        passing = {
            name: result for name, result in arms.items()
            if result["passed_all_gates"]
        }
        candidates = passing or arms
        best_arm = min(
            candidates,
            key=lambda name: candidates[name][
                "selection_score_lower_is_better"],
        )
    payload = {
        "protocol": "matched-no-gs-vs-gs-vs-gt-v1/holdout5",
        "baseline": baseline,
        "arms": arms,
        "best_arm": best_arm,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(payload, args.out / "summary.json")
    report = markdown_report(payload)
    (args.out / "summary.md").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
