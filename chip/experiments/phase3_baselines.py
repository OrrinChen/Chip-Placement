from __future__ import annotations

import random
import time
from statistics import mean

from chip.problem import ChipProblem
from chip.rl_adapter import ChipRLAdapter
from engine.rl_env import PlacementEnv


def run_masked_random_baseline(
    *,
    problem: ChipProblem,
    grid_shape: tuple[int, int] = (32, 32),
    steps: int,
    seed: int = 0,
) -> dict[str, object]:
    adapter = ChipRLAdapter(problem=problem, grid_shape=grid_shape)
    rng = random.Random(seed)
    state = adapter.initial_state()
    trace: list[dict[str, object]] = []
    invalid_action_count = 0

    while len(trace) < steps:
        step_index = len(state.placed_macros)
        current_macro = adapter.current_macro_for_step(step_index)
        if current_macro is None:
            break
        action = adapter.sample_random_action(state, rng)
        action_mask = adapter.build_action_mask(state, step_index=step_index, grid_shape=grid_shape)
        grid_index = adapter.coordinate_to_grid_index(action["target"], grid_shape)
        if bool(action_mask[grid_index[0], grid_index[1]]):
            invalid_action_count += 1
        state = adapter.apply_action(state, action)
        trace.append({"action": action, "step_index": step_index})

    return {
        "steps": len(trace),
        "trace": trace,
        "invalid_action_count": invalid_action_count,
        "final_hpwl": adapter.current_hpwl(state),
    }


def run_greedy_hpwl_baseline(
    *,
    problem: ChipProblem,
    grid_shape: tuple[int, int] = (32, 32),
) -> dict[str, object]:
    adapter = ChipRLAdapter(problem=problem, grid_shape=grid_shape)
    state = adapter.initial_state()
    trace: list[dict[str, object]] = []

    while True:
        step_index = len(state.placed_macros)
        current_macro = adapter.current_macro_for_step(step_index)
        if current_macro is None:
            break

        action_mask = adapter.build_action_mask(state, step_index=step_index, grid_shape=grid_shape)
        best_action = None
        best_hpwl = None
        for row_index in range(grid_shape[0]):
            for col_index in range(grid_shape[1]):
                if bool(action_mask[row_index, col_index]):
                    continue
                target = adapter.grid_index_to_coordinate((row_index, col_index), grid_shape)
                action = {"macro": current_macro, "target": target}
                candidate = adapter.apply_action(state, action)
                candidate_hpwl = adapter.current_hpwl(candidate)
                if best_hpwl is None or candidate_hpwl < best_hpwl:
                    best_hpwl = candidate_hpwl
                    best_action = action

        if best_action is None:
            break
        state = adapter.apply_action(state, best_action)
        trace.append({"action": best_action, "step_index": step_index, "hpwl": best_hpwl})

    return {
        "steps": len(trace),
        "trace": trace,
        "final_hpwl": adapter.current_hpwl(state),
    }


def run_masked_random_episode(
    *,
    problem: ChipProblem,
    grid_shape: tuple[int, int] = (32, 32),
    reward_source: str = "analytic_hpwl",
    reward_model=None,
    hpwl_evaluator=None,
    seed: int = 0,
) -> dict[str, object]:
    adapter = ChipRLAdapter(
        problem=problem,
        reward_model=reward_model,
        hpwl_evaluator=hpwl_evaluator,
        grid_shape=grid_shape,
    )
    env = PlacementEnv(adapter=adapter, max_steps=len(adapter.ordered_macros()), reward_source=reward_source)
    rng = random.Random(seed)
    start = time.perf_counter()
    observation, _ = env.reset()
    episode_return = 0.0
    trace: list[dict[str, object]] = []
    terminated = False
    truncated = False
    termination_reason = "completed"

    while not terminated and not truncated:
        current_state = env.state
        step_index = len(current_state.placed_macros)
        action_mask = adapter.build_action_mask(current_state, step_index=step_index, grid_shape=grid_shape)
        if bool(action_mask.all()):
            termination_reason = "no_valid_actions"
            break
        action = adapter.sample_random_action(current_state, rng)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        trace.append(
            {
                "action": action,
                "reward": reward,
                "step_index": step_index,
            }
        )

    runtime_seconds = time.perf_counter() - start
    final_state = env.state
    return {
        "steps": len(trace),
        "trace": trace,
        "episode_return": float(episode_return),
        "final_hpwl": adapter.current_hpwl(final_state),
        "is_legal": adapter.constraint_report(final_state).is_valid,
        "runtime_seconds": float(runtime_seconds),
        "final_observation": observation,
        "termination_reason": termination_reason,
        "completed_episode": termination_reason == "completed",
    }


def run_greedy_episode(
    *,
    problem: ChipProblem,
    grid_shape: tuple[int, int] = (32, 32),
    reward_source: str = "analytic_hpwl",
    reward_model=None,
    hpwl_evaluator=None,
    macro_order: tuple[str, ...] | None = None,
) -> dict[str, object]:
    adapter = ChipRLAdapter(
        problem=problem,
        reward_model=reward_model,
        hpwl_evaluator=hpwl_evaluator,
        grid_shape=grid_shape,
        macro_order=macro_order,
    )
    env = PlacementEnv(adapter=adapter, max_steps=len(adapter.ordered_macros()), reward_source=reward_source)
    start = time.perf_counter()
    observation, _ = env.reset()
    episode_return = 0.0
    trace: list[dict[str, object]] = []
    terminated = False
    truncated = False
    termination_reason = "completed"

    while not terminated and not truncated:
        step_index = len(env.state.placed_macros)
        current_macro = adapter.current_macro_for_step(step_index)
        if current_macro is None:
            break
        action_mask = adapter.build_action_mask(env.state, step_index=step_index, grid_shape=grid_shape)
        best_action = None
        best_hpwl = None
        for row_index in range(grid_shape[0]):
            for col_index in range(grid_shape[1]):
                if bool(action_mask[row_index, col_index]):
                    continue
                target = adapter.grid_index_to_coordinate((row_index, col_index), grid_shape)
                action = {"macro": current_macro, "target": target}
                candidate = adapter.apply_action(env.state, action)
                candidate_hpwl = adapter.current_hpwl(candidate)
                if best_hpwl is None or candidate_hpwl < best_hpwl:
                    best_hpwl = candidate_hpwl
                    best_action = action
        if best_action is None:
            termination_reason = "no_valid_actions"
            break
        observation, reward, terminated, truncated, info = env.step(best_action)
        episode_return += reward
        trace.append(
            {
                "action": best_action,
                "reward": reward,
                "step_index": step_index,
                "hpwl": best_hpwl,
            }
        )

    runtime_seconds = time.perf_counter() - start
    final_state = env.state
    return {
        "steps": len(trace),
        "trace": trace,
        "episode_return": float(episode_return),
        "final_hpwl": adapter.current_hpwl(final_state),
        "is_legal": adapter.constraint_report(final_state).is_valid,
        "runtime_seconds": float(runtime_seconds),
        "final_observation": observation,
        "termination_reason": termination_reason,
        "completed_episode": termination_reason == "completed",
    }


def summarize_episode_metrics(episodes: list[dict[str, object]]) -> dict[str, object]:
    if not episodes:
        return {
            "episode_count": 0,
            "legality_rate": 0.0,
            "episode_return": {"mean": 0.0},
            "final_hpwl": {"mean": 0.0},
            "runtime_seconds": {"mean": 0.0},
            "completed_episode_rate": 0.0,
        }
    legal_count = sum(1 for episode in episodes if bool(episode.get("is_legal", False)))
    episode_returns = [float(episode["episode_return"]) for episode in episodes]
    final_hpwls = [float(episode["final_hpwl"]) for episode in episodes]
    runtimes = [float(episode["runtime_seconds"]) for episode in episodes]
    return {
        "episode_count": len(episodes),
        "legality_rate": legal_count / len(episodes),
        "episode_return": {"mean": mean(episode_returns)},
        "final_hpwl": {"mean": mean(final_hpwls)},
        "runtime_seconds": {"mean": mean(runtimes)},
        "completed_episode_rate": sum(1 for episode in episodes if bool(episode.get("completed_episode", False))) / len(episodes),
    }
