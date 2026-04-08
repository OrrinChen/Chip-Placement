from __future__ import annotations

import tempfile
from pathlib import Path
from statistics import mean
from statistics import pstdev

from chip.experiments.phase3_baselines import run_greedy_hpwl_baseline
from chip.experiments.phase3_baselines import run_masked_random_baseline
from chip.experiments.phase3_imitation import train_behavior_cloning_policy
from chip.experiments.phase3_ppo import collect_policy_rollout
from chip.experiments.phase3_ppo import initialize_policy_for_problem
from chip.experiments.phase3_ppo import load_policy_warm_start
from chip.experiments.phase3_ppo import train_ppo_policy
from chip.experiments.phase3_search import build_demonstrations_from_trace
from chip.experiments.phase3_search import run_beam_search_baseline
from chip.problem import ChipProblem


def run_phase3_comparison(
    *,
    problem: ChipProblem,
    grid_shape: tuple[int, int] = (32, 32),
    hidden_dim: int = 64,
    bc_epochs: int = 120,
    ppo_epochs: int = 1,
    reward_source: str = "analytic_hpwl",
    beam_width: int = 2,
    seed: int = 0,
    seeds: tuple[int, ...] | None = None,
) -> dict[str, object]:
    seed_values = tuple(seeds) if seeds is not None else (seed,)
    per_seed_reports = [
        _run_phase3_comparison_single_seed(
            problem=problem,
            grid_shape=grid_shape,
            hidden_dim=hidden_dim,
            bc_epochs=bc_epochs,
            ppo_epochs=ppo_epochs,
            reward_source=reward_source,
            beam_width=beam_width,
            seed=current_seed,
        )
        for current_seed in seed_values
    ]
    methods = {}
    method_names = per_seed_reports[0]["methods"].keys()
    for method_name in method_names:
        rows = [report["methods"][method_name] for report in per_seed_reports]
        aggregate = _aggregate_method_rows(rows)
        methods[method_name] = (
            {
                **rows[0],
                "per_seed": rows,
                "aggregate": aggregate,
            }
            if len(rows) == 1
            else {
                **aggregate,
                "per_seed": rows,
                "aggregate": aggregate,
            }
        )
    return {
        "seed_count": len(seed_values),
        "seeds": list(seed_values),
        "beam_width": int(beam_width),
        "methods": methods,
    }


def _run_phase3_comparison_single_seed(
    *,
    problem: ChipProblem,
    grid_shape: tuple[int, int],
    hidden_dim: int,
    bc_epochs: int,
    ppo_epochs: int,
    reward_source: str,
    beam_width: int,
    seed: int,
) -> dict[str, object]:
    random_result = run_masked_random_baseline(
        problem=problem,
        grid_shape=grid_shape,
        steps=len([macro for macro in problem.macros.values() if not macro.is_fixed]),
        seed=seed,
    )
    greedy_result = run_greedy_hpwl_baseline(problem=problem, grid_shape=grid_shape)
    beam_result = run_beam_search_baseline(problem=problem, grid_shape=grid_shape, beam_width=beam_width)
    demonstrations = build_demonstrations_from_trace(
        problem=problem,
        grid_shape=grid_shape,
        rollout_type="beam_search",
        trace=beam_result["trace"],
        final_score=beam_result["final_hpwl"],
    )

    with tempfile.TemporaryDirectory() as tempdir:
        checkpoint_path = Path(tempdir) / "bc_policy.pt"
        bc_result = train_behavior_cloning_policy(
            problem=problem,
            demonstrations=demonstrations,
            grid_shape=grid_shape,
            hidden_dim=hidden_dim,
            epochs=bc_epochs,
            learning_rate=1e-2,
            checkpoint_path=checkpoint_path,
        )

        behavior_cloning_rollout = collect_policy_rollout(
            problem=problem,
            policy=bc_result["policy"],
            grid_shape=grid_shape,
            reward_source=reward_source,
            max_steps=max(1, len(demonstrations)),
            seed=seed,
            deterministic=True,
        )

        scratch_policy = initialize_policy_for_problem(problem=problem, grid_shape=grid_shape, hidden_dim=hidden_dim)
        scratch_training = train_ppo_policy(
            problem=problem,
            policy=scratch_policy,
            grid_shape=grid_shape,
            reward_source=reward_source,
            max_steps=max(1, len(demonstrations)),
            num_updates=max(1, ppo_epochs),
            learning_rate=1e-3,
            seed=seed,
        )

        warm_policy = initialize_policy_for_problem(problem=problem, grid_shape=grid_shape, hidden_dim=hidden_dim)
        load_policy_warm_start(warm_policy, checkpoint_path)
        warm_training = train_ppo_policy(
            problem=problem,
            policy=warm_policy,
            grid_shape=grid_shape,
            reward_source=reward_source,
            max_steps=max(1, len(demonstrations)),
            num_updates=max(1, ppo_epochs),
            learning_rate=1e-3,
            seed=seed,
        )

    methods = {
        "random": _summarize_baseline(random_result),
        "greedy": _summarize_baseline(greedy_result),
        "beam": _summarize_baseline(beam_result),
        "behavior_cloning": _summarize_policy_rollout(
            behavior_cloning_rollout,
            extra={"masked_accuracy": float(bc_result["masked_accuracy"])},
        ),
        "ppo_scratch": _summarize_policy_rollout(
            scratch_training["final_rollout"],
            extra=_coerce_training_metrics(scratch_training),
        ),
        "ppo_warm_start": _summarize_policy_rollout(
            warm_training["final_rollout"],
            extra=_coerce_training_metrics(warm_training),
        ),
    }
    beam_level = float(methods["beam"]["final_hpwl"])
    bc_level = float(methods["behavior_cloning"]["final_hpwl"])
    for method_name in ("ppo_scratch", "ppo_warm_start"):
        curve = methods[method_name].get("curve", [])
        methods[method_name]["updates_to_beam_level"] = float(_updates_to_level(curve, beam_level))
        methods[method_name]["updates_to_bc_level"] = float(_updates_to_level(curve, bc_level))
    return {"seed": float(seed), "methods": methods}


def _summarize_baseline(result: dict[str, object]) -> dict[str, float]:
    return {
        "steps": float(result["steps"]),
        "invalid_action_count": float(result.get("invalid_action_count", 0)),
        "final_hpwl": float(result["final_hpwl"]),
    }


def _summarize_policy_rollout(result: dict[str, object], *, extra: dict[str, float] | None = None) -> dict[str, float]:
    summary = {
        "steps": float(result["steps"]),
        "invalid_action_count": float(result.get("invalid_action_count", 0)),
        "final_hpwl": float(result["final_hpwl"]),
    }
    if extra:
        summary.update(extra)
    return summary


def _coerce_training_metrics(training: dict[str, object] | None) -> dict[str, float]:
    if training is None:
        return {}
    history = training.get("history", [])
    if not history:
        return {"update_count": 0.0}
    last = history[-1]
    return {
        "update_count": float(len(history)),
        "policy_loss": float(last["policy_loss"]),
        "value_loss": float(last["value_loss"]),
        "entropy": float(last["entropy"]),
        "curve": [{key: float(value) for key, value in item.items()} for item in history],
    }


def _aggregate_method_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row.keys()})
    aggregate: dict[str, float] = {}
    for key in keys:
        if key == "curve":
            continue
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        aggregate[key] = float(mean(values))
        if len(values) > 1:
            aggregate[f"{key}_std"] = float(pstdev(values))
    return aggregate


def _updates_to_level(curve: list[dict[str, float]], target_hpwl: float) -> int:
    for index, point in enumerate(curve, start=1):
        if float(point["final_hpwl"]) <= target_hpwl:
            return index
    return len(curve) + 1
