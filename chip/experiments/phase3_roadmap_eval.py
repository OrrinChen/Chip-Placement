from __future__ import annotations

import tempfile
import time
from pathlib import Path

from chip.constraints import build_constraint_registry
from chip.cost import hpwl_cost
from chip.experiments.baseline import _neighbor
from chip.experiments.baseline import _objective
from chip.experiments.phase3_baselines import run_greedy_hpwl_baseline
from chip.experiments.phase3_baselines import run_masked_random_baseline
from chip.experiments.phase3_benchmarks import build_phase3_benchmark_ladder
from chip.experiments.phase3_imitation import train_behavior_cloning_policy
from chip.experiments.phase3_ppo import collect_policy_rollout
from chip.experiments.phase3_ppo import initialize_policy_for_problem
from chip.experiments.phase3_ppo import load_policy_warm_start
from chip.experiments.phase3_ppo import train_ppo_policy
from chip.experiments.phase3_search import build_demonstrations_from_trace
from chip.experiments.phase3_search import run_beam_search_baseline
from chip.openroad_bridge import OpenROADBridge
from chip.openroad_eval import build_evaluation_request
from chip.openroad_eval import evaluate_openroad_request
from chip.rl_adapter import ChipRLAdapter
from engine.analytical_solver import QuadraticPlacementSolver
from engine.sa_solver import SimulatedAnnealingSolver


def build_phase3_roadmap_eval_report() -> dict[str, object]:
    ladder = build_phase3_benchmark_ladder()
    benchmark_rows = {
        item["name"]: {
            "grid_shape": item["grid_shape"],
            "movable_macro_count": item["movable_macro_count"],
            "node_count": item["node_count"],
            "net_count": item["net_count"],
        }
        for item in ladder
    }
    method_rows: dict[str, dict[str, object]] = {
        "random": {},
        "greedy": {},
        "beam": {},
        "behavior_cloning": {},
        "ppo_scratch": {},
        "ppo_warm_start": {},
        "analytical": {},
        "sa": {},
        "openroad_default": {},
    }

    for item in ladder:
        name = str(item["name"])
        problem = item["problem"]
        grid_shape = item["grid_shape"]

        direct_rows = _run_direct_method_suite(problem=problem, grid_shape=grid_shape)
        for method_name, row in direct_rows.items():
            method_rows[method_name][name] = row

        analytical_row, sa_row = _run_classical_baselines(problem=problem, grid_shape=grid_shape)
        method_rows["analytical"][name] = analytical_row
        method_rows["sa"][name] = sa_row

        if "artifact_paths" in item:
            method_rows["openroad_default"][name] = _run_openroad_default(item["artifact_paths"])

    return {
        "roadmap_phase3_status": "partial",
        "benchmarks": tuple(item["name"] for item in ladder),
        "benchmark_rows": benchmark_rows,
        "method_rows": method_rows,
        "remaining_gaps": (
            "current_rl_mainline_still_uses_analytic_hpwl_in_primary_eval",
            "openroad_metrics_available_only_for_sample-backed_small_benchmark",
            "full_rl_vs_sa_vs_analytical_vs_openroad_table_not_yet_extended_to_three_real_chip_benchmarks",
        ),
    }


def _run_direct_method_suite(*, problem, grid_shape: tuple[int, int]) -> dict[str, dict[str, object]]:
    method_rows: dict[str, dict[str, object]] = {}
    movable_count = len([macro for macro in problem.macros.values() if not macro.is_fixed])

    start = time.perf_counter()
    random_result = run_masked_random_baseline(problem=problem, grid_shape=grid_shape, steps=movable_count, seed=0)
    method_rows["random"] = _with_runtime(random_result, start)

    start = time.perf_counter()
    greedy_result = run_greedy_hpwl_baseline(problem=problem, grid_shape=grid_shape)
    method_rows["greedy"] = _with_runtime(greedy_result, start)

    start = time.perf_counter()
    beam_result = run_beam_search_baseline(problem=problem, grid_shape=grid_shape, beam_width=2)
    method_rows["beam"] = _with_runtime(beam_result, start)

    demonstrations = build_demonstrations_from_trace(
        problem=problem,
        grid_shape=grid_shape,
        rollout_type="beam_search",
        trace=beam_result["trace"],
        final_score=beam_result["final_hpwl"],
    )

    with tempfile.TemporaryDirectory() as tempdir:
        checkpoint_path = Path(tempdir) / "bc_policy.pt"

        start = time.perf_counter()
        bc_result = train_behavior_cloning_policy(
            problem=problem,
            demonstrations=demonstrations,
            grid_shape=grid_shape,
            hidden_dim=16,
            epochs=120,
            learning_rate=1e-2,
            checkpoint_path=checkpoint_path,
        )
        bc_rollout = collect_policy_rollout(
            problem=problem,
            policy=bc_result["policy"],
            grid_shape=grid_shape,
            reward_source="analytic_hpwl",
            max_steps=max(1, len(demonstrations)),
            seed=0,
            deterministic=True,
        )
        method_rows["behavior_cloning"] = _with_runtime(
            {
                "steps": bc_rollout["steps"],
                "invalid_action_count": bc_rollout["invalid_action_count"],
                "final_hpwl": bc_rollout["final_hpwl"],
                "masked_accuracy": float(bc_result["masked_accuracy"]),
            },
            start,
        )

        scratch_policy = initialize_policy_for_problem(problem=problem, grid_shape=grid_shape, hidden_dim=16)
        start = time.perf_counter()
        scratch_training = train_ppo_policy(
            problem=problem,
            policy=scratch_policy,
            grid_shape=grid_shape,
            reward_source="analytic_hpwl",
            max_steps=max(1, len(demonstrations)),
            num_updates=3,
            learning_rate=1e-3,
            seed=0,
        )
        method_rows["ppo_scratch"] = _with_runtime(
            {
                "steps": scratch_training["final_rollout"]["steps"],
                "invalid_action_count": scratch_training["final_rollout"]["invalid_action_count"],
                "final_hpwl": scratch_training["final_rollout"]["final_hpwl"],
                "update_count": len(scratch_training["history"]),
            },
            start,
        )

        warm_policy = initialize_policy_for_problem(problem=problem, grid_shape=grid_shape, hidden_dim=16)
        load_policy_warm_start(warm_policy, checkpoint_path)
        start = time.perf_counter()
        warm_training = train_ppo_policy(
            problem=problem,
            policy=warm_policy,
            grid_shape=grid_shape,
            reward_source="analytic_hpwl",
            max_steps=max(1, len(demonstrations)),
            num_updates=3,
            learning_rate=1e-3,
            seed=0,
        )
        method_rows["ppo_warm_start"] = _with_runtime(
            {
                "steps": warm_training["final_rollout"]["steps"],
                "invalid_action_count": warm_training["final_rollout"]["invalid_action_count"],
                "final_hpwl": warm_training["final_rollout"]["final_hpwl"],
                "update_count": len(warm_training["history"]),
            },
            start,
        )

    return method_rows


def _run_classical_baselines(*, problem, grid_shape: tuple[int, int]) -> tuple[dict[str, object], dict[str, object]]:
    adapter = ChipRLAdapter(problem=problem, grid_shape=grid_shape)
    graph = adapter.observe(adapter.initial_state(), step_index=0)["graph"]
    constraints = build_constraint_registry(problem)

    start = time.perf_counter()
    analytical_placement = QuadraticPlacementSolver().solve(graph)
    analytical_runtime = time.perf_counter() - start
    analytical_row = {
        "final_hpwl": hpwl_cost(problem, analytical_placement),
        "is_valid": bool(constraints.check_all(analytical_placement).is_valid),
        "runtime_seconds": analytical_runtime,
    }

    bounds = graph.global_features["bounds"]
    start = time.perf_counter()
    sa_result = SimulatedAnnealingSolver(seed=7).solve(
        initial_state=problem.placement(),
        objective_fn=lambda placement: _objective(problem, constraints, placement),
        neighbor_fn=lambda placement, rng: _neighbor(problem, placement, bounds, rng),
        iterations=100,
    )
    sa_runtime = time.perf_counter() - start
    sa_row = {
        "final_hpwl": hpwl_cost(problem, sa_result.best_state),
        "is_valid": bool(constraints.check_all(sa_result.best_state).is_valid),
        "runtime_seconds": sa_runtime,
    }
    return analytical_row, sa_row


def _run_openroad_default(artifact_paths: dict[str, Path]) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    liberty_path = artifact_paths.get("liberty_path")
    sdc_path = artifact_paths.get("sdc_path")
    if liberty_path is None or sdc_path is None:
        return {"status": "unsupported_missing_timing_artifacts"}
    relative_lef = artifact_paths["lef_path"].resolve().relative_to(root).as_posix()
    relative_def = artifact_paths["def_path"].resolve().relative_to(root).as_posix()
    relative_liberty = liberty_path.resolve().relative_to(root).as_posix()
    relative_sdc = sdc_path.resolve().relative_to(root).as_posix()
    request = build_evaluation_request(
        benchmark_name="small",
        lef_path=relative_lef,
        def_path=relative_def,
        placement_mode="built_in",
        artifact_dir=str(root),
        lef_paths=(relative_lef,),
        liberty_path=relative_liberty,
        sdc_path=relative_sdc,
        run_detailed_route=False,
        script_stem="phase3_openroad_default",
    )
    bridge = OpenROADBridge(docker_image="openroad/orfs")
    if not bridge.is_available():
        return {"status": "unavailable"}
    try:
        metrics = evaluate_openroad_request(request, bridge=bridge, timeout_seconds=120)
        return {"status": "completed", **metrics}
    except Exception as error:  # pragma: no cover - environment dependent
        return {"status": "error", "error": str(error)}


def _with_runtime(result: dict[str, object], start_time: float) -> dict[str, object]:
    return {
        **result,
        "runtime_seconds": time.perf_counter() - start_time,
    }
