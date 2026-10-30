"""Figures from committed Phase 2 and sample-benchmark artifacts.

This script does not train a model and does not invent proxy predictions.
Per-row proxy outputs are not in the committed report, so the scatter is
analytic HPWL against routed wirelength on the Phase 2 corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from chip.netlist_parser import load_chip_problem


REPORT_PATH = ROOT / "experiments" / "chip" / "phase2_proxy_frozen_base_alpha0_full" / "phase2_proxy_report.json"
DATASET_PATH = ROOT / "experiments" / "chip" / "phase2_dataset_full_stratified_main.json"
DATASET_SUMMARY_PATH = ROOT / "experiments" / "chip" / "phase2_dataset_full_stratified_main_summary.json"
SAMPLE_LEF = ROOT / "chip" / "benchmarks" / "sample.lef"
SAMPLE_DEF = ROOT / "chip" / "benchmarks" / "sample.def"


def load_report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def corpus_points(report: dict) -> list[dict[str, object]]:
    summary = json.loads(DATASET_SUMMARY_PATH.read_text(encoding="utf-8"))
    rows = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if int(summary["success_count"]) != 300 or int(summary["total_rows"]) != 300:
        raise ValueError("Phase 2 dataset summary is not the committed 300/300 success corpus")
    points = []
    for row in rows:
        if row.get("status") != "success":
            raise ValueError(f"Unexpected dataset status: {row.get('status')}")
        points.append(
            {
                "benchmark": row["benchmark"],
                "hpwl": float(row["hpwl"]),
                "total_routed_wirelength": float(row["total_routed_wirelength"]),
            }
        )
    if len(points) != 300:
        raise ValueError(f"Expected 300 corpus rows, found {len(points)}")
    signal = report["target_signal_summary"]["benchmarks"]
    for benchmark, group in signal.items():
        count = sum(1 for point in points if point["benchmark"] == benchmark)
        if count != int(group["row_count"]):
            raise ValueError(f"{benchmark} row count {count} != report {group['row_count']}")
    return points


def plot_spearman(report: dict, output_path: Path) -> None:
    designs = []
    for fold in report["same_design_folds"]:
        if fold["benchmark"] not in designs:
            designs.append(fold["benchmark"])
    same = {fold["benchmark"]: float(fold["mean_spearman"]) for fold in report["same_design_folds"]}
    cross = {fold["holdout_benchmark"]: float(fold["mean_spearman"]) for fold in report["cross_design_folds"]}
    positions = list(range(len(designs)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    axis.bar(
        [position - width / 2 for position in positions],
        [same[design] for design in designs],
        width=width,
        label="same-design fold",
        color="#1f4e79",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [cross[design] for design in designs],
        width=width,
        label="cross-design fold",
        color="#d4760a",
    )
    axis.axhline(
        float(report["same_design_mean_spearman"]),
        color="#1f4e79",
        linestyle="--",
        linewidth=1,
        label=f"same-design mean {report['same_design_mean_spearman']:.6g}",
    )
    axis.axhline(
        float(report["cross_design_mean_spearman"]),
        color="#d4760a",
        linestyle="--",
        linewidth=1,
        label=f"cross-design mean {report['cross_design_mean_spearman']:.6g}",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(designs, rotation=15, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("mean Spearman (HPWL and routed WL)")
    axis.set_title("Frozen linear proxy, Phase 2 folds")
    axis.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.55))
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def plot_corpus(report: dict, output_path: Path) -> None:
    points = corpus_points(report)
    signal = report["target_signal_summary"]["benchmarks"]
    colors = {
        "mgc_des_perf_2": "#1f4e79",
        "mgc_pci_bridge32_1": "#d4760a",
        "mgc_pci_bridge32_2": "#2f6f4e",
    }
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))
    for axis, (benchmark, color) in zip(axes, colors.items(), strict=True):
        group = [point for point in points if point["benchmark"] == benchmark]
        committed = signal[benchmark]["hpwl_vs_routed_spearman"]
        axis.scatter(
            [point["hpwl"] for point in group],
            [point["total_routed_wirelength"] for point in group],
            s=18,
            alpha=0.8,
            color=color,
        )
        axis.set_title(f"{benchmark}\ncommitted Spearman {committed:.4f}", fontsize=9)
        axis.set_xlabel("analytic HPWL")
    axes[0].set_ylabel("routed wirelength")
    figure.suptitle("Phase 2 corpus, 300/300 success. Not proxy predictions.", fontsize=11)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def plot_sample_placement(output_path: Path) -> None:
    _, problem = load_chip_problem(SAMPLE_LEF, SAMPLE_DEF)
    figure, axis = plt.subplots(figsize=(5.2, 5.2))
    die = problem.die_area
    axis.add_patch(
        Rectangle(
            (die.min_x, die.min_y),
            die.width,
            die.height,
            fill=False,
            linewidth=1.2,
            edgecolor="#222222",
        )
    )
    for macro in problem.macros.values():
        axis.add_patch(
            Rectangle(
                (macro.x, macro.y),
                macro.width,
                macro.height,
                facecolor="#d6e2f0",
                edgecolor="#1f4e79",
                linewidth=1,
            )
        )
        axis.text(
            macro.x + macro.width / 2,
            macro.y + macro.height / 2,
            macro.name,
            ha="center",
            va="center",
            fontsize=8,
        )
    axis.set_aspect("equal")
    axis.set_xlim(die.min_x, die.max_x)
    axis.set_ylim(die.min_y, die.max_y)
    axis.set_title("Committed sample netlist\nchip/benchmarks/sample.def")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot committed Phase 2 and sample-benchmark artifacts.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    report = load_report()
    plot_spearman(report, args.output_dir / "phase2_spearman_folds.png")
    plot_corpus(report, args.output_dir / "phase2_hpwl_vs_routed.png")
    plot_sample_placement(args.output_dir / "sample_benchmark_placement.png")
    print(f"wrote figures under {args.output_dir}")


if __name__ == "__main__":
    main()
