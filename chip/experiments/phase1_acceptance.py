from __future__ import annotations

import json
from pathlib import Path

from chip.ispd_benchmarks import ACCEPTANCE_BENCHMARKS


def summarize_acceptance_row(
    benchmark_name: str,
    scale_bucket: str,
    design_stats: dict[str, int],
    sa_summary: dict[str, float | bool],
    built_in_metrics: dict[str, float | int],
    sa_metrics: dict[str, float | int],
) -> dict[str, object]:
    return {
        "benchmark": benchmark_name,
        "scale_bucket": scale_bucket,
        "node_count": design_stats["node_count"],
        "net_count": design_stats["net_count"],
        "sa_hpwl": sa_summary["hpwl"],
        "sa_is_valid": sa_summary["is_valid"],
        "sa_runtime_seconds": sa_summary["runtime_seconds"],
        "built_in_total_routed_wirelength": built_in_metrics["total_routed_wirelength"],
        "built_in_wns": built_in_metrics["wns"],
        "built_in_tns": built_in_metrics["tns"],
        "built_in_drc_violations": built_in_metrics["drc_violations"],
        "built_in_runtime_seconds": built_in_metrics["runtime_seconds"],
        "sa_total_routed_wirelength": sa_metrics["total_routed_wirelength"],
        "sa_wns": sa_metrics["wns"],
        "sa_tns": sa_metrics["tns"],
        "sa_drc_violations": sa_metrics["drc_violations"],
        "sa_openroad_runtime_seconds": sa_metrics["runtime_seconds"],
    }


def run_acceptance_suite(
    benchmark_root: Path,
    output_path: Path,
    benchmark_names: list[str] | None = None,
) -> list[dict[str, object]]:
    selected_benchmarks = ACCEPTANCE_BENCHMARKS
    if benchmark_names is not None:
        selected_names = set(benchmark_names)
        selected_benchmarks = [benchmark for benchmark in ACCEPTANCE_BENCHMARKS if benchmark.name in selected_names]
    rows = [run_single_benchmark_acceptance(benchmark_root, benchmark) for benchmark in selected_benchmarks]
    Path(output_path).write_text(json.dumps(rows, indent=2))
    return rows


def run_single_benchmark_acceptance(benchmark_root: Path, benchmark) -> dict[str, object]:
    return {"benchmark": benchmark.name, "error": "not_implemented"}
