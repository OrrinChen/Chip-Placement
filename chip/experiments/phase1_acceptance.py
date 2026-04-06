from __future__ import annotations

import json
import os
import time
from pathlib import Path

from chip.constraints import build_constraint_registry
from chip.cost import hpwl_cost
from chip.experiments.baseline import _neighbor
from chip.ispd_benchmarks import ISPDBenchmark
from chip.ispd_benchmarks import ACCEPTANCE_BENCHMARKS
from chip.ispd_benchmarks import prepare_benchmark_artifacts
from chip.netlist_parser import load_chip_problem
from chip.openroad_bridge import OpenROADBridge
from chip.openroad_eval import build_evaluation_request
from chip.openroad_eval import evaluate_openroad_request
from chip.problem import ChipProblem
from engine.sa_solver import SimulatedAnnealingSolver


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
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(rows, indent=2))
    return rows


def run_single_benchmark_acceptance(benchmark_root: Path, benchmark: ISPDBenchmark) -> dict[str, object]:
    try:
        prepared = prepare_benchmark_artifacts(benchmark_root=benchmark_root, benchmark=benchmark)
        graph, problem = load_chip_problem(prepared["cells_lef_path"], prepared["def_path"])
        design_stats = {"node_count": len(graph.node_ids), "net_count": len(problem.nets)}
        artifact_dir = Path(prepared["artifact_dir"])
        seed_def_name = "seed_placed.def"
        seed_def_path = artifact_dir / seed_def_name
        lef_paths = tuple(
            _relative_path(path, artifact_dir)
            for path in [prepared["tech_lef_path"], prepared["site_fix_lef_path"], prepared["cells_lef_path"]]
        )
        built_in_request = build_evaluation_request(
            benchmark_name=benchmark.name,
            artifact_dir=str(artifact_dir),
            lef_path=lef_paths[0],
            lef_paths=lef_paths,
            def_path=_relative_path(prepared["def_path"], artifact_dir),
            liberty_path=_relative_path(prepared["liberty_path"], artifact_dir),
            sdc_path=_relative_path(prepared["sdc_path"], artifact_dir),
            placement_mode="built_in",
            detailed_route_end_iter=benchmark.detailed_route_end_iter,
            placement_density=benchmark.placement_density,
            write_def_path=seed_def_name,
        )
        bridge = OpenROADBridge(docker_image="openroad/orfs")
        built_in_metrics = evaluate_openroad_request(built_in_request, bridge=bridge)
        _, seed_problem = load_chip_problem(prepared["cells_lef_path"], seed_def_path)
        sa_placement, sa_summary = _run_sa_baseline(seed_problem, iterations=benchmark.sa_iterations)
        injectable_placement, orientations = _select_injectable_instances(seed_problem, sa_placement)
        sa_metrics = evaluate_openroad_request(
            build_evaluation_request(
                benchmark_name=benchmark.name,
                artifact_dir=str(artifact_dir),
                lef_path=lef_paths[0],
                lef_paths=lef_paths,
                def_path=seed_def_name,
                liberty_path=_relative_path(prepared["liberty_path"], artifact_dir),
                sdc_path=_relative_path(prepared["sdc_path"], artifact_dir),
                placement_mode="sa_injected",
                detailed_route_end_iter=benchmark.detailed_route_end_iter,
            ),
            bridge=bridge,
            placement=injectable_placement,
            orientations=orientations,
        )
    except Exception as error:
        return {
            "benchmark": benchmark.name,
            "scale_bucket": benchmark.scale_bucket,
            "error": str(error),
        }

    return summarize_acceptance_row(
        benchmark_name=benchmark.name,
        scale_bucket=benchmark.scale_bucket,
        design_stats=design_stats,
        sa_summary=sa_summary,
        built_in_metrics=built_in_metrics,
        sa_metrics=sa_metrics,
    )


def _run_sa_baseline(problem, iterations: int) -> tuple[dict[str, tuple[float, float]], dict[str, float | bool]]:
    constraints = build_constraint_registry(problem)
    initial_placement = problem.placement()
    bounds = (
        problem.die_area.min_x,
        problem.die_area.max_x,
        problem.die_area.min_y,
        problem.die_area.max_y,
    )
    solver = SimulatedAnnealingSolver(seed=7)
    start = time.perf_counter()
    result = solver.solve(
        initial_state=initial_placement,
        objective_fn=lambda placement: hpwl_cost(problem, placement) + constraints.check_all(placement).total_penalty,
        neighbor_fn=lambda placement, rng: _neighbor(problem, placement, bounds, rng),
        iterations=iterations,
    )
    runtime_seconds = time.perf_counter() - start
    report = constraints.check_all(result.best_state)
    return result.best_state, {
        "hpwl": hpwl_cost(problem, result.best_state),
        "is_valid": report.is_valid,
        "runtime_seconds": runtime_seconds,
    }


def _select_injectable_instances(
    problem: ChipProblem,
    placement: dict[str, tuple[float, float]],
    site_pitch: float = 0.2,
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    injectable_placement: dict[str, tuple[float, float]] = {}
    orientations: dict[str, str] = {}
    for name, macro in problem.macros.items():
        if macro.is_fixed or macro.kind == "PIN":
            continue
        if name not in placement:
            continue
        target_x, _ = placement[name]
        snapped_x = round(target_x / site_pitch) * site_pitch
        clamped_x = min(max(snapped_x, problem.die_area.min_x), problem.die_area.max_x - macro.width)
        injectable_placement[name] = (round(clamped_x, 6), macro.y)
        orientations[name] = macro.orientation
    return injectable_placement, orientations


def _relative_path(path: Path, artifact_dir: Path) -> str:
    return Path(os.path.relpath(Path(path), artifact_dir)).as_posix()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    benchmark_root = repo_root / "tmp"
    output_path = repo_root / "experiments" / "chip" / "phase1_acceptance_results.json"
    run_acceptance_suite(benchmark_root=benchmark_root, output_path=output_path)


if __name__ == "__main__":
    main()
