"""Masked PPO versus simulated annealing on the in-repo toy ladder.

The reported metric is analytic HPWL. This entrypoint executes only the
``tiny``, ``small``, and ``medium`` problems from
``chip.experiments.phase3_benchmarks``. It does not run Ariane, OpenROAD,
or ORFS, and it does not read or rewrite the committed Phase 2 / Ariane
artifacts.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from chip.constraints import build_constraint_registry
from chip.cost import hpwl_cost
from chip.experiments.baseline import _neighbor
from chip.experiments.baseline import _objective
from chip.experiments.phase3_benchmarks import build_phase3_benchmark_ladder
from chip.experiments.phase3_ppo import initialize_policy_for_problem
from chip.experiments.phase3_ppo import train_ppo_policy
from chip.problem import ChipProblem
from chip.rl_adapter import ChipRLAdapter
from engine.sa_solver import SimulatedAnnealingSolver


SUPPORTED_BENCHMARKS = ("tiny", "small", "medium")
METHOD_MASKED_PPO = "masked_ppo"
METHOD_SA = "simulated_annealing"


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        raise ValueError("summarize_numeric requires at least one value")
    mean = float(statistics.fmean(values))
    std = float(statistics.stdev(values)) if len(values) >= 2 else None
    return {"count": len(values), "mean": mean, "std": std}


def display_mean_std(summary: dict[str, float | int | None]) -> str:
    mean = float(summary["mean"])
    std = summary["std"]
    if std is None:
        return f"{mean:.6g} (std undefined, n={summary['count']})"
    return f"{mean:.6g} ± {float(std):.6g}"


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return config


def run_masked_ppo_vs_sa(config: dict[str, object]) -> dict[str, object]:
    benchmarks = _require_supported_benchmarks(config)
    seeds = _require_seeds(config)
    ppo_config = _require_mapping(config, "ppo")
    sa_config = _require_mapping(config, "sa")
    ladder = {str(item["name"]): item for item in build_phase3_benchmark_ladder()}
    missing = [name for name in benchmarks if name not in ladder]
    if missing:
        raise ValueError(f"Benchmark ladder is missing {missing}")

    benchmark_reports = {}
    placement_reports = {}
    for name in benchmarks:
        item = ladder[name]
        problem: ChipProblem = item["problem"]
        grid_shape = tuple(int(value) for value in item["grid_shape"])
        report, placements = _run_benchmark(
            name=name,
            problem=problem,
            grid_shape=grid_shape,
            movable_macro_count=int(item["movable_macro_count"]),
            node_count=int(item["node_count"]),
            net_count=int(item["net_count"]),
            seeds=seeds,
            ppo_config=ppo_config,
            sa_config=sa_config,
        )
        benchmark_reports[name] = report
        placement_reports[name] = placements

    return {
        "experiment": "masked_ppo_vs_simulated_annealing",
        "status": "completed",
        "metric": "analytic_hpwl",
        "lower_is_better": True,
        "std_definition": "sample standard deviation (statistics.stdev, ddof=1) across seeds; null when n < 2",
        "display_format": "Python format .6g applied to the stored mean and std; full precision is in this file",
        "config_name": config.get("name"),
        "config": config,
        "seeds": list(seeds),
        "benchmarks": benchmark_reports,
        "placements": placement_reports,
        "qualifiers": _qualifiers(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }


def write_run(report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(report)
    placements = summary.pop("placements")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "placements.json").write_text(json.dumps(placements, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare masked PPO with simulated annealing on the toy ladder.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    try:
        _require_supported_benchmarks(config)
    except ValueError as error:
        print(f"not yet run: {error}", file=sys.stderr)
        print("No metrics were written.", file=sys.stderr)
        return 2
    report = run_masked_ppo_vs_sa(config)
    report["command"] = "python -m chip.experiments.ppo_vs_sa --config " + _display_path(args.config) + " --output-dir " + _display_path(args.output_dir)
    write_run(report, args.output_dir)
    _print_display(report)
    return 0


def _run_benchmark(
    *,
    name: str,
    problem: ChipProblem,
    grid_shape: tuple[int, int],
    movable_macro_count: int,
    node_count: int,
    net_count: int,
    seeds: tuple[int, ...],
    ppo_config: dict[str, object],
    sa_config: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    adapter = ChipRLAdapter(problem=problem, grid_shape=grid_shape)
    max_steps = max(1, len(adapter.ordered_macros()))
    initial_placement = problem.placement()
    initial_hpwl = float(hpwl_cost(problem, initial_placement))
    constraints = build_constraint_registry(problem)
    initial_penalty = float(constraints.check_all(initial_placement).total_penalty)
    initial_legal = bool(constraints.check_all(initial_placement).is_valid)

    ppo_rows = []
    sa_rows = []
    example_seed = seeds[0]
    example_placements = {
        "die_area": {
            "min_x": problem.die_area.min_x,
            "min_y": problem.die_area.min_y,
            "max_x": problem.die_area.max_x,
            "max_y": problem.die_area.max_y,
        },
        "initial": _geometry(problem, initial_placement),
    }
    for seed in seeds:
        print(f"[{name}] seed {seed}: masked PPO", flush=True)
        ppo_row = _run_masked_ppo(
            problem=problem,
            grid_shape=grid_shape,
            max_steps=max_steps,
            seed=seed,
            ppo_config=ppo_config,
        )
        print(f"[{name}] seed {seed}: simulated annealing", flush=True)
        sa_row = _run_sa(
            problem=problem,
            seed=seed,
            sa_config=sa_config,
        )
        if seed == example_seed:
            example_placements[METHOD_MASKED_PPO] = _geometry(problem, _placement_dict(ppo_row.pop("placement")))
            example_placements[METHOD_SA] = _geometry(problem, _placement_dict(sa_row.pop("placement")))
        else:
            ppo_row.pop("placement", None)
            sa_row.pop("placement", None)
        ppo_rows.append(ppo_row)
        sa_rows.append(sa_row)

    ppo_summary = _method_summary(ppo_rows)
    sa_summary = _method_summary(sa_rows)
    deltas = [
        float(ppo_row["final_hpwl"]) - float(sa_row["final_hpwl"])
        for ppo_row, sa_row in zip(ppo_rows, sa_rows, strict=True)
    ]
    return (
        {
            "grid_shape": list(grid_shape),
            "movable_macro_count": movable_macro_count,
            "ordered_macro_count": max_steps,
            "node_count": node_count,
            "net_count": net_count,
            "row_count": len(problem.rows),
            "initial_hpwl": initial_hpwl,
            "initial_total_penalty": initial_penalty,
            "initial_is_legal": initial_legal,
            "methods": {
                METHOD_MASKED_PPO: ppo_summary,
                METHOD_SA: sa_summary,
            },
            "paired_hpwl_delta_ppo_minus_sa": summarize_numeric(deltas),
            "paired_hpwl_delta_ppo_minus_sa_display": display_mean_std(summarize_numeric(deltas)),
        },
        example_placements,
    )


def _run_masked_ppo(
    *,
    problem: ChipProblem,
    grid_shape: tuple[int, int],
    max_steps: int,
    seed: int,
    ppo_config: dict[str, object],
) -> dict[str, object]:
    hidden_dim = int(ppo_config.get("hidden_dim", 32))
    num_updates = int(ppo_config["num_updates"])
    policy = initialize_policy_for_problem(problem=problem, grid_shape=grid_shape, hidden_dim=hidden_dim)
    torch.manual_seed(seed)
    started = time.perf_counter()
    training = train_ppo_policy(
        problem=problem,
        policy=policy,
        grid_shape=grid_shape,
        reward_source="analytic_hpwl",
        max_steps=max_steps,
        num_updates=num_updates,
        learning_rate=float(ppo_config.get("learning_rate", 1e-3)),
        clip_epsilon=float(ppo_config.get("clip_epsilon", 0.2)),
        gamma=float(ppo_config.get("gamma", 0.99)),
        entropy_coef=float(ppo_config.get("entropy_coef", 0.01)),
        value_coef=float(ppo_config.get("value_coef", 0.5)),
        seed=seed,
    )
    runtime_seconds = time.perf_counter() - started
    rollout = training["final_rollout"]
    history = training["history"]
    return {
        "seed": seed,
        "final_hpwl": float(rollout["final_hpwl"]),
        "is_legal": bool(rollout["is_legal"]),
        "invalid_action_count": int(rollout["invalid_action_count"]),
        "completed_episode": bool(rollout["completed_episode"]),
        "steps": int(rollout["steps"]),
        "runtime_seconds": runtime_seconds,
        "num_updates": num_updates,
        "hidden_dim": hidden_dim,
        "eval": "deterministic argmax after training",
        "training_episode_final_hpwl": [float(row["final_hpwl"]) for row in history],
        "placement": rollout["placement"],
    }


def _run_sa(
    *,
    problem: ChipProblem,
    seed: int,
    sa_config: dict[str, object],
) -> dict[str, object]:
    constraints = build_constraint_registry(problem)
    bounds = (
        problem.die_area.min_x,
        problem.die_area.max_x,
        problem.die_area.min_y,
        problem.die_area.max_y,
    )
    iterations = int(sa_config["iterations"])
    solver = SimulatedAnnealingSolver(seed=seed)
    started = time.perf_counter()
    result = solver.solve(
        initial_state=problem.placement(),
        objective_fn=lambda placement: _objective(problem, constraints, placement),
        neighbor_fn=lambda placement, rng: _neighbor(problem, placement, bounds, rng),
        iterations=iterations,
        initial_temperature=float(sa_config.get("initial_temperature", 10.0)),
        cooling_rate=float(sa_config.get("cooling_rate", 0.95)),
    )
    runtime_seconds = time.perf_counter() - started
    best_placement = result.best_state
    report = constraints.check_all(best_placement)
    return {
        "seed": seed,
        "final_hpwl": float(hpwl_cost(problem, best_placement)),
        "objective_hpwl_plus_penalty": float(result.best_score),
        "total_penalty": float(report.total_penalty),
        "is_legal": bool(report.is_valid),
        "runtime_seconds": runtime_seconds,
        "iterations": iterations,
        "placement": {name: [float(coord[0]), float(coord[1])] for name, coord in best_placement.items()},
    }


def _method_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    hpwl = summarize_numeric([float(row["final_hpwl"]) for row in rows])
    runtime = summarize_numeric([float(row["runtime_seconds"]) for row in rows])
    legal_count = sum(1 for row in rows if bool(row["is_legal"]))
    public_rows = [{key: value for key, value in row.items() if key != "placement"} for row in rows]
    return {
        "final_hpwl": hpwl,
        "final_hpwl_display": display_mean_std(hpwl),
        "runtime_seconds": runtime,
        "runtime_seconds_display": display_mean_std(runtime),
        "legal_count": legal_count,
        "seed_count": len(rows),
        "per_seed": public_rows,
    }


def _geometry(problem: ChipProblem, placement: dict[str, tuple[float, float]]) -> dict[str, object]:
    return {
        name: {
            "x": float(placement[name][0]),
            "y": float(placement[name][1]),
            "width": float(macro.width),
            "height": float(macro.height),
            "is_fixed": bool(macro.is_fixed),
            "kind": macro.kind,
        }
        for name, macro in problem.macros.items()
    }


def _placement_dict(value: object) -> dict[str, tuple[float, float]]:
    if not isinstance(value, dict):
        raise TypeError("placement must be a dict")
    return {str(name): (float(coord[0]), float(coord[1])) for name, coord in value.items()}


def _require_supported_benchmarks(config: dict[str, object]) -> list[str]:
    raw = config.get("benchmarks")
    if not isinstance(raw, list) or not raw:
        raise ValueError("config.benchmarks must be a non-empty list")
    names = [str(name) for name in raw]
    unsupported = [name for name in names if name not in SUPPORTED_BENCHMARKS]
    if unsupported:
        supported = ", ".join(SUPPORTED_BENCHMARKS)
        raise ValueError(
            "benchmark(s) "
            + ", ".join(unsupported)
            + f" are outside this runner (supported: {supported}). "
            + "Real-chip Ariane masked PPO vs SA was not executed."
        )
    return names


def _require_seeds(config: dict[str, object]) -> tuple[int, ...]:
    raw = config.get("seeds")
    if not isinstance(raw, list) or not raw:
        raise ValueError("config.seeds must be a non-empty list of integers")
    return tuple(int(seed) for seed in raw)


def _require_mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config.{key} must be an object")
    return value


def _qualifiers() -> list[str]:
    return [
        "Toy ladder only: tiny, small, and medium problems defined in chip/experiments/phase3_benchmarks.py.",
        "Not Ariane, not an ISPD mgc_* benchmark, and not an OpenROAD or ORFS routed-wirelength result.",
        "Masked PPO uses PPOPlacementPolicy, which fills illegal grid cells with -inf before sampling.",
        "PPO is trained from scratch for num_updates on each seed. The reported HPWL is a deterministic argmax rollout after training, not the best stochastic training episode.",
        "PPO training-episode HPWL values are stored separately and are not the comparison metric.",
        "Simulated annealing uses the Phase 1 neighbor in chip/experiments/baseline.py. Its search objective is analytic HPWL plus constraint penalty. The reported score is analytic HPWL of the incumbent.",
        "The two methods do not share an action space: PPO places macros sequentially onto a legality mask; SA perturbs a full placement.",
        "These numbers are not the Phase 2 Spearman results and not the Ariane reference-flow DRC count.",
    ]


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _print_display(report: dict[str, object]) -> None:
    benchmarks = report["benchmarks"]
    assert isinstance(benchmarks, dict)
    for name, row in benchmarks.items():
        methods = row["methods"]
        print(f"{name} masked_ppo analytic HPWL: {methods[METHOD_MASKED_PPO]['final_hpwl_display']}")
        print(f"{name} simulated_annealing analytic HPWL: {methods[METHOD_SA]['final_hpwl_display']}")
        print(f"{name} paired delta (PPO - SA): {row['paired_hpwl_delta_ppo_minus_sa_display']}")


if __name__ == "__main__":
    raise SystemExit(main())
