from __future__ import annotations

import random

from chip.problem import ChipProblem
from chip.rl_adapter import ChipRLAdapter


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
