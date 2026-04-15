from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from chip.experiments.phase3_baselines import run_greedy_episode
from chip.experiments.phase3_baselines import run_masked_random_episode
from chip.experiments.phase3_baselines import summarize_episode_metrics
from chip.experiments.phase3_macroplacement_benchmarks import materialize_combined_lef
from chip.experiments.phase3_macroplacement_benchmarks import resolve_ariane_benchmark
from chip.macroplacement_wirelength import MacroPlacementWirelengthEvaluator
from chip.netlist_parser import load_chip_problem
from chip.phase3_reward import RewardWeights
from chip.phase3_reward import SeedCompletedRewardModel
from chip.phase3_reward import load_frozen_linear_proxy
from chip.rl_adapter import build_macro_order
from chip.rl_adapter import ChipRLAdapter


DEFAULT_MASKED_RANDOM_EPISODES = 100


def build_ariane_formulation_config() -> dict[str, object]:
    return {
        "grid_shape": (64, 64),
        "macro_order_strategy": "descending_area",
    }


def build_ariane_problem(*, artifact_dir: Path | None = None):
    spec = resolve_ariane_benchmark()
    resolved_artifact_dir = (artifact_dir or (_repo_root() / "experiments" / "chip" / "phase3_macroplacement_bringup")).resolve()
    resolved_artifact_dir.mkdir(parents=True, exist_ok=True)
    combined_lef_path = materialize_combined_lef(spec, output_dir=resolved_artifact_dir)
    _, problem = load_chip_problem(combined_lef_path, spec.reference_def_path)
    seed_placement = {macro.name: (macro.x, macro.y) for macro in problem.macros.values() if macro.kind != "PIN"}
    return spec, problem, seed_placement, combined_lef_path


def build_ariane_reward_model(
    *,
    artifact_dir: Path | None = None,
    weights: RewardWeights | None = None,
) -> tuple[SeedCompletedRewardModel, dict[str, object]]:
    spec, problem, seed_placement, combined_lef_path = build_ariane_problem(artifact_dir=artifact_dir)
    reward_model = SeedCompletedRewardModel(
        problem=problem,
        seed_placement=seed_placement,
        proxy=load_frozen_linear_proxy(target_name="total_routed_wirelength"),
        hpwl_evaluator=MacroPlacementWirelengthEvaluator(netlist_path=spec.grouped_netlist_pbtxt_path),
        # The Phase 2 frozen linear proxy is trained on standard-cell-dominant
        # perturbation data and is badly out-of-domain on macro-placement.
        # Ariane PPO bring-up therefore uses HPWL-driven rewards and keeps the
        # proxy available only for offline diagnostics.
        weights=weights or RewardWeights(w_proxy=0.0, w_hpwl=1.0, legality_penalty=10.0),
        normalize_rewards=False,
    )
    metadata = {
        "benchmark": spec.name,
        "reference_def_path": str(spec.reference_def_path),
        "combined_lef_path": str(combined_lef_path),
        "seed_macro_count": len(seed_placement),
        "reward_weights": {
            "w_proxy": reward_model.weights.w_proxy,
            "w_hpwl": reward_model.weights.w_hpwl,
            "legality_penalty": reward_model.weights.legality_penalty,
            "completion_bonus": reward_model.weights.completion_bonus,
        },
    }
    return reward_model, metadata


def summarize_formulation_sweep(
    *,
    runs: list[dict[str, object]],
    canonical_label: str,
) -> dict[str, object]:
    if not runs:
        raise ValueError("Need at least one formulation run to summarize")
    by_hpwl = min(runs, key=lambda run: float(run["episode"]["final_hpwl"]))
    by_steps = max(runs, key=lambda run: int(run["episode"]["steps"]))
    canonical = next(run for run in runs if run["label"] == canonical_label)
    return {
        "best_hpwl": {
            "label": by_hpwl["label"],
            "final_hpwl": float(by_hpwl["episode"]["final_hpwl"]),
            "steps": int(by_hpwl["episode"]["steps"]),
            "completed_episode": bool(by_hpwl["episode"]["completed_episode"]),
        },
        "best_steps": {
            "label": by_steps["label"],
            "final_hpwl": float(by_steps["episode"]["final_hpwl"]),
            "steps": int(by_steps["episode"]["steps"]),
            "completed_episode": bool(by_steps["episode"]["completed_episode"]),
        },
        "canonical_comparison": {
            "label": canonical_label,
            "hpwl_delta": float(by_hpwl["episode"]["final_hpwl"]) - float(canonical["episode"]["final_hpwl"]),
            "step_delta": int(by_steps["episode"]["steps"]) - int(canonical["episode"]["steps"]),
            "completion_improved": bool(by_hpwl["episode"]["completed_episode"]) and not bool(canonical["episode"]["completed_episode"]),
        },
    }


def measure_reward_runtime(
    reward_model: SeedCompletedRewardModel,
    *,
    grid_shape: tuple[int, int] = (32, 32),
    repeats: int = 25,
) -> dict[str, float]:
    adapter = ChipRLAdapter(
        problem=reward_model.problem,
        reward_model=reward_model,
        hpwl_evaluator=reward_model.hpwl_evaluator,
        grid_shape=grid_shape,
    )
    state = adapter.initial_state()
    action_mask = adapter.build_action_mask(state, step_index=0, grid_shape=grid_shape)
    chosen_index = None
    for row_index in range(grid_shape[0]):
        for col_index in range(grid_shape[1]):
            if not bool(action_mask[row_index, col_index]):
                chosen_index = (row_index, col_index)
                break
        if chosen_index is not None:
            break
    if chosen_index is None:
        raise ValueError("Ariane reward runtime probe found no legal actions")
    current_macro = adapter.current_macro_for_step(0)
    assert current_macro is not None
    target = adapter.grid_index_to_coordinate(chosen_index, grid_shape)
    next_state = adapter.apply_action(state, {"macro": current_macro, "target": target})
    is_legal = adapter.constraint_report(next_state).is_valid
    reward_model.reset_episode()
    started = time.perf_counter()
    for _ in range(repeats):
        reward_model.compute_step_reward(state, next_state, is_legal=is_legal)
    elapsed = time.perf_counter() - started
    return {
        "single_step_reward_seconds_mean": elapsed / repeats,
        "single_step_reward_seconds_total": elapsed,
        "single_step_reward_repeats": repeats,
    }


def run_masked_random_ariane(
    *,
    output_path: Path,
    episode_count: int = DEFAULT_MASKED_RANDOM_EPISODES,
    grid_shape: tuple[int, int] = (32, 32),
) -> dict[str, object]:
    reward_model, metadata = build_ariane_reward_model(artifact_dir=output_path.parent)
    runtime_metrics = measure_reward_runtime(reward_model, grid_shape=grid_shape)
    episodes = []
    for seed in range(episode_count):
        result = run_masked_random_episode(
            problem=reward_model.problem,
            grid_shape=grid_shape,
            reward_source="phase3_proxy_hpwl",
            reward_model=reward_model,
            hpwl_evaluator=reward_model.hpwl_evaluator,
            seed=seed,
        )
        episodes.append(
            {
                "seed": seed,
                "steps": result["steps"],
                "episode_return": result["episode_return"],
                "final_hpwl": result["final_hpwl"],
                "is_legal": result["is_legal"],
                "runtime_seconds": result["runtime_seconds"],
                "termination_reason": result["termination_reason"],
                "completed_episode": result["completed_episode"],
            }
        )
    summary = summarize_episode_metrics(episodes)
    artifact = {
        "benchmark": metadata["benchmark"],
        "method": "masked_random",
        "grid_shape": list(grid_shape),
        "reward_source": "phase3_proxy_hpwl",
        "episode_count": episode_count,
        "reward_runtime": runtime_metrics,
        "metadata": metadata,
        "summary": summary,
        "episodes": episodes,
        "milestone_a_gate": {
            "reward_pipeline_seconds_lt_50ms": runtime_metrics["single_step_reward_seconds_mean"] < 0.05,
            "episode_runtime_lt_2h_for_100_episodes": sum(episode["runtime_seconds"] for episode in episodes) < 7200.0,
        },
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def run_greedy_ariane(
    *,
    output_path: Path,
    random_floor_path: Path | None = None,
    grid_shape: tuple[int, int] = (32, 32),
) -> dict[str, object]:
    reward_model, metadata = build_ariane_reward_model(artifact_dir=output_path.parent)
    result = run_greedy_episode(
        problem=reward_model.problem,
        grid_shape=grid_shape,
        reward_source="phase3_proxy_hpwl",
        reward_model=reward_model,
        hpwl_evaluator=reward_model.hpwl_evaluator,
    )
    floor_mean_hpwl = None
    if random_floor_path is not None and random_floor_path.exists():
        floor_mean_hpwl = float(json.loads(random_floor_path.read_text(encoding="utf-8-sig"))["summary"]["final_hpwl"]["mean"])
    hpwl_improvement_fraction = None
    if floor_mean_hpwl is not None and floor_mean_hpwl > 0:
        hpwl_improvement_fraction = (floor_mean_hpwl - float(result["final_hpwl"])) / floor_mean_hpwl
    artifact = {
        "benchmark": metadata["benchmark"],
        "method": "greedy_one_step_analytic_hpwl",
        "grid_shape": list(grid_shape),
        "reward_source": "phase3_proxy_hpwl",
        "action_selection_source": "analytic_hpwl",
        "metadata": metadata,
        "episode": {
            "steps": result["steps"],
            "episode_return": result["episode_return"],
            "final_hpwl": result["final_hpwl"],
            "is_legal": result["is_legal"],
            "runtime_seconds": result["runtime_seconds"],
            "termination_reason": result["termination_reason"],
            "completed_episode": result["completed_episode"],
        },
        "comparison_to_masked_random": {
            "masked_random_mean_hpwl": floor_mean_hpwl,
            "hpwl_improvement_fraction": hpwl_improvement_fraction,
            "passes_20_percent_gate": hpwl_improvement_fraction is not None and hpwl_improvement_fraction >= 0.2,
        },
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def run_ariane_formulation_sweep(
    *,
    output_path: Path,
    grid_shapes: tuple[tuple[int, int], ...] = ((32, 32), (64, 64)),
    macro_order_strategies: tuple[str, ...] = ("descending_area", "seed_yx"),
) -> dict[str, object]:
    reward_model, metadata = build_ariane_reward_model(artifact_dir=output_path.parent)
    runs: list[dict[str, object]] = []
    canonical_label = f"g{grid_shapes[0][0]}x{grid_shapes[0][1]}_{macro_order_strategies[0]}"
    for grid_shape in grid_shapes:
        for strategy in macro_order_strategies:
            macro_order = build_macro_order(reward_model.problem, strategy)
            started = time.perf_counter()
            result = run_greedy_episode(
                problem=reward_model.problem,
                grid_shape=grid_shape,
                reward_source="phase3_proxy_hpwl",
                reward_model=reward_model,
                hpwl_evaluator=reward_model.hpwl_evaluator,
                macro_order=macro_order,
            )
            runtime = time.perf_counter() - started
            runs.append(
                {
                    "label": f"g{grid_shape[0]}x{grid_shape[1]}_{strategy}",
                    "grid_shape": list(grid_shape),
                    "macro_order_strategy": strategy,
                    "episode": {
                        "steps": int(result["steps"]),
                        "episode_return": float(result["episode_return"]),
                        "final_hpwl": float(result["final_hpwl"]),
                        "is_legal": bool(result["is_legal"]),
                        "runtime_seconds": float(runtime),
                        "termination_reason": str(result["termination_reason"]),
                        "completed_episode": bool(result["completed_episode"]),
                    },
                }
            )
    artifact = {
        "benchmark": metadata["benchmark"],
        "grid_shapes": [list(shape) for shape in grid_shapes],
        "macro_order_strategies": list(macro_order_strategies),
        "canonical_label": canonical_label,
        "runs": runs,
        "summary": summarize_formulation_sweep(runs=runs, canonical_label=canonical_label),
        "metadata": metadata,
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 Ariane bring-up baselines.")
    parser.add_argument(
        "--output-dir",
        default=str(_repo_root() / "experiments" / "chip"),
    )
    parser.add_argument(
        "--run",
        choices=("masked-random", "greedy", "all"),
        default="all",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_MASKED_RANDOM_EPISODES)
    parser.add_argument("--grid-rows", type=int, default=32)
    parser.add_argument("--grid-cols", type=int, default=32)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_shape = (args.grid_rows, args.grid_cols)
    random_output_path = output_dir / "phase3_masked_random_ariane.json"
    greedy_output_path = output_dir / "phase3_greedy_ariane.json"

    if args.run in {"masked-random", "all"}:
        run_masked_random_ariane(
            output_path=random_output_path,
            episode_count=args.episodes,
            grid_shape=grid_shape,
        )
    if args.run in {"greedy", "all"}:
        run_greedy_ariane(
            output_path=greedy_output_path,
            random_floor_path=random_output_path,
            grid_shape=grid_shape,
        )


if __name__ == "__main__":
    main()
