"""Plot a completed masked PPO vs simulated annealing run.

Reads summary.json and placements.json written by chip.experiments.ppo_vs_sa.
Does not recompute the experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_hpwl(summary: dict, output_path: Path) -> None:
    benchmarks = list(summary["benchmarks"])
    methods = ("masked_ppo", "simulated_annealing")
    labels = {"masked_ppo": "masked PPO", "simulated_annealing": "simulated annealing"}
    colors = {"masked_ppo": "#1f4e79", "simulated_annealing": "#d4760a"}
    positions = list(range(len(benchmarks)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for offset, method in ((-width / 2, methods[0]), (width / 2, methods[1])):
        means = []
        errors = []
        for name in benchmarks:
            stats = summary["benchmarks"][name]["methods"][method]["final_hpwl"]
            means.append(float(stats["mean"]))
            errors.append(0.0 if stats["std"] is None else float(stats["std"]))
        axis.bar(
            [position + offset for position in positions],
            means,
            width=width,
            yerr=errors,
            capsize=3,
            color=colors[method],
            label=labels[method],
        )
    axis.set_xticks(positions, benchmarks)
    axis.set_ylabel("analytic HPWL (mean ± sample std)")
    axis.set_title(f"{summary.get('config_name')} · lower is better")
    axis.legend(frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def plot_placements(summary: dict, placements: dict, output_path: Path) -> None:
    names = [name for name in summary["benchmarks"] if name in placements]
    if not names:
        raise ValueError("placements.json has no benchmark in the summary")
    name = max(names, key=lambda item: summary["benchmarks"][item]["movable_macro_count"])
    panels = (
        ("initial", "initial"),
        ("masked_ppo", "masked PPO, seed 0"),
        ("simulated_annealing", "SA, seed 0"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))
    die = placements[name]["die_area"]
    width = float(die["max_x"]) - float(die["min_x"])
    height = float(die["max_y"]) - float(die["min_y"])
    pad_x = 0.03 * width if width else 0.5
    pad_y = 0.08 * height if height else 0.5
    for axis, (key, title) in zip(axes, panels, strict=True):
        axis.add_patch(
            Rectangle(
                (die["min_x"], die["min_y"]),
                width,
                height,
                fill=False,
                linewidth=1,
                edgecolor="#222222",
            )
        )
        for macro_name, macro in placements[name][key].items():
            color = "#f4e1c1" if macro["is_fixed"] else "#d6e2f0"
            macro_width = float(macro["width"])
            macro_height = float(macro["height"])
            if macro_width <= 0 or macro_height <= 0:
                axis.scatter([macro["x"]], [macro["y"]], s=18, color="#8a5a00", zorder=3)
            else:
                axis.add_patch(
                    Rectangle(
                        (macro["x"], macro["y"]),
                        macro_width,
                        macro_height,
                        facecolor=color,
                        edgecolor="#1f4e79",
                        linewidth=0.8,
                    )
                )
            if macro["kind"] != "PIN":
                axis.text(macro["x"], macro["y"], macro_name, fontsize=6, va="bottom")
        axis.set_xlim(float(die["min_x"]) - pad_x, float(die["max_x"]) + pad_x)
        axis.set_ylim(float(die["min_y"]) - pad_y, float(die["max_y"]) + pad_y)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title, fontsize=9)
    figure.suptitle(f"{name} placements from this run (seed 0)", fontsize=11)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a masked PPO vs SA summary.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    placements = json.loads(args.placements.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise ValueError("Refusing to plot a run that is not status=completed")
    plot_hpwl(summary, args.output_dir / "ppo_vs_sa_hpwl.png")
    plot_placements(summary, placements, args.output_dir / "ppo_vs_sa_placements.png")
    print(f"wrote figures under {args.output_dir}")


if __name__ == "__main__":
    main()
