from __future__ import annotations

import random
import subprocess
import time
from pathlib import Path

from chip.constraints import build_constraint_registry
from chip.cost import hpwl_cost
from chip.netlist_parser import load_chip_problem
from chip.openroad_bridge import OpenROADBridge, OpenROADUnavailableError
from engine.analytical_solver import QuadraticPlacementSolver
from engine.sa_solver import SimulatedAnnealingSolver


def run_baseline_experiment(
    lef_path: str | Path,
    def_path: str | Path,
    iterations: int = 50,
    bridge_binary: str = "openroad",
    docker_image: str | None = None,
    analytical_node_limit: int | None = 200,
) -> dict[str, object]:
    graph, problem = load_chip_problem(lef_path, def_path)
    constraints = build_constraint_registry(problem)
    sa_solver = SimulatedAnnealingSolver(seed=7)
    initial_placement = problem.placement()

    analytical_summary: dict[str, float | bool | str]
    if analytical_node_limit is not None and len(graph.node_ids) > analytical_node_limit:
        analytical_summary = {
            "status": "skipped",
            "reason": "node_limit_exceeded",
            "node_count": len(graph.node_ids),
        }
    else:
        analytical = QuadraticPlacementSolver().solve(graph)
        analytical_summary = _summarize(problem, constraints, analytical)

    bounds = graph.global_features["bounds"]
    sa_start = time.perf_counter()
    sa_result = sa_solver.solve(
        initial_state=initial_placement,
        objective_fn=lambda placement: _objective(problem, constraints, placement),
        neighbor_fn=lambda placement, rng: _neighbor(problem, placement, bounds, rng),
        iterations=iterations,
    )
    sa_runtime_seconds = time.perf_counter() - sa_start

    bridge = OpenROADBridge(binary_name=bridge_binary, docker_image=docker_image)
    openroad_metrics: dict[str, str] | None
    openroad_error: str | None = None
    try:
        openroad_metrics = bridge.evaluate_placement(lef_path, def_path, sa_result.best_state) if bridge.is_available() else None
    except (OpenROADUnavailableError, subprocess.CalledProcessError) as error:
        openroad_metrics = None
        openroad_error = getattr(error, "stderr", None) or str(error)

    return {
        "openroad_available": bridge.is_available(),
        "placements": {
            "initial": _summarize(problem, constraints, initial_placement),
            "analytical": analytical_summary,
            "sa": _summarize(problem, constraints, sa_result.best_state, runtime_seconds=sa_runtime_seconds),
        },
        "openroad": openroad_metrics,
        "openroad_error": openroad_error,
    }


def _objective(problem, constraints, placement: dict[str, tuple[float, float]]) -> float:
    report = constraints.check_all(placement)
    return hpwl_cost(problem, placement) + report.total_penalty


def _neighbor(problem, placement, bounds, rng: random.Random) -> dict[str, tuple[float, float]]:
    min_x, max_x, min_y, max_y = bounds
    movable_macros = [name for name, macro in problem.macros.items() if not macro.is_fixed]
    if not movable_macros:
        return dict(placement)
    updated = dict(placement)
    if problem.rows:
        if len(movable_macros) > 1 and rng.choice(["swap", "shift"]) == "swap":
            macro_name = rng.choice(movable_macros)
            macro = problem.macros[macro_name]
            compatible_macros = [
                other_name
                for other_name in movable_macros
                if other_name != macro_name
                and abs(problem.macros[other_name].width - macro.width) <= 1e-9
                and abs(problem.macros[other_name].height - macro.height) <= 1e-9
            ]
            if compatible_macros:
                other_name = rng.choice(compatible_macros)
                updated[macro_name], updated[other_name] = placement[other_name], placement[macro_name]
                return updated

        macro_name = rng.choice(movable_macros)
        current_x, current_y = placement[macro_name]
        axis = rng.choice(["x", "y"])
        if axis == "y" and len(problem.rows) > 1:
            current_row_index = problem.row_index(current_y)
            if current_row_index is None:
                current_row_index = 0
            row_delta = rng.choice([-1, 1])
            next_row_index = min(max(current_row_index + row_delta, 0), len(problem.rows) - 1)
            target_y = problem.rows[next_row_index].origin_y
            next_x, next_y, _ = problem.snap_movable_macro(macro_name, current_x, target_y)
        else:
            site_pitch = problem.site_pitch()
            delta_x = rng.choice([-5, -2, -1, 1, 2, 5]) * site_pitch
            next_x, next_y, _ = problem.snap_movable_macro(macro_name, current_x + delta_x, current_y)
        updated[macro_name] = (next_x, next_y)
        return updated

    macro_name = rng.choice(movable_macros)
    macro = problem.macros[macro_name]
    current_x, current_y = placement[macro_name]
    delta_x = rng.choice([-5.0, 5.0])
    delta_y = rng.choice([-5.0, 5.0])
    next_x = min(max(current_x + delta_x, min_x), max_x - macro.width)
    next_y = min(max(current_y + delta_y, min_y), max_y - macro.height)
    updated[macro_name] = (next_x, next_y)
    return updated


def _summarize(
    problem,
    constraints,
    placement: dict[str, tuple[float, float]],
    runtime_seconds: float | None = None,
) -> dict[str, float | bool]:
    report = constraints.check_all(placement)
    summary = {
        "status": "completed",
        "hpwl": hpwl_cost(problem, placement),
        "total_penalty": report.total_penalty,
        "is_valid": report.is_valid,
    }
    if runtime_seconds is not None:
        summary["runtime_seconds"] = runtime_seconds
    return summary
