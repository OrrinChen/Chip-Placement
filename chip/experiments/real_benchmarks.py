from __future__ import annotations

from pathlib import Path

from chip.experiments.baseline import run_baseline_experiment
from chip.netlist_parser import load_chip_problem
from chip.orfs_export import ORFSDesignExporter


def summarize_design_result(design_name: str, design_stats: dict[str, int], result: dict[str, object]) -> dict[str, object]:
    placements = result["placements"]
    initial = placements["initial"]
    sa = placements["sa"]
    analytical = placements["analytical"]
    return {
        "design": design_name,
        "node_count": design_stats["node_count"],
        "fixed_node_count": design_stats["fixed_node_count"],
        "net_count": design_stats["net_count"],
        "initial_hpwl": initial["hpwl"],
        "initial_is_valid": initial["is_valid"],
        "sa_hpwl": sa["hpwl"],
        "sa_is_valid": sa["is_valid"],
        "analytical_status": analytical["status"],
        "openroad_available": result["openroad_available"],
    }


def summarize_many_designs(results: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for design_name, payload in results.items():
        summary = payload["summary"]
        if summary is not None:
            rows.append(summary)
        else:
            rows.append({"design": design_name, "error": payload["error"]})
    return rows


def design_artifact_paths(export_root: Path, design_name: str, platform: str = "nangate45") -> dict[str, Path]:
    bundle_root = Path(export_root) / platform / design_name
    return {
        "lef_path": bundle_root / "NangateOpenCellLibrary.tech.lef",
        "def_path": bundle_root / "3_place.def",
        "macro_lef_path": bundle_root / "NangateOpenCellLibrary.macro.mod.lef",
        "sdc_path": bundle_root / "3_place.sdc",
    }


def run_real_benchmark_suite(
    work_root: Path,
    export_root: Path,
    design_names: list[str],
    iterations: int = 20,
    analytical_node_limit: int | None = 200,
    platform: str = "nangate45",
) -> list[dict[str, object]]:
    exporter = ORFSDesignExporter(work_root=Path(work_root), export_root=Path(export_root), platform=platform)
    rows: list[dict[str, object]] = []
    for design_name in design_names:
        export_result = exporter.export_design(design_name)
        status = export_result["status"]
        if status not in {"ready", "exported"}:
            rows.append({"design": design_name, "error": status})
            continue

        artifact_paths = design_artifact_paths(export_root, design_name, platform)
        graph, problem = load_chip_problem(artifact_paths["macro_lef_path"], artifact_paths["def_path"])
        baseline_result = run_baseline_experiment(
            artifact_paths["macro_lef_path"],
            artifact_paths["def_path"],
            iterations=iterations,
            bridge_binary="definitely-missing-openroad",
            analytical_node_limit=analytical_node_limit,
        )
        baseline_result["openroad_available"] = True
        rows.append(summarize_design_result(design_name, collect_design_stats(graph, problem), baseline_result))
    return rows


def collect_design_stats(graph, problem) -> dict[str, int]:
    fixed_nodes = graph.global_features.get("fixed_nodes", set())
    return {
        "node_count": len(graph.node_ids),
        "fixed_node_count": len(fixed_nodes),
        "net_count": len(problem.nets),
    }
