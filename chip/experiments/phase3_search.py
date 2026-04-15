from __future__ import annotations

from dataclasses import dataclass

from chip.imitation import DemonstrationStep
from chip.problem import ChipProblem
from chip.rl_adapter import ChipRLAdapter
from chip.rl_adapter import ChipRLState


@dataclass(frozen=True)
class SearchCandidate:
    state: ChipRLState
    trace: tuple[dict[str, object], ...]
    score: float


def run_beam_search_baseline(
    *,
    problem: ChipProblem,
    grid_shape: tuple[int, int] = (32, 32),
    beam_width: int = 4,
) -> dict[str, object]:
    adapter = ChipRLAdapter(problem=problem, grid_shape=grid_shape)
    initial_state = adapter.initial_state()
    frontier = [SearchCandidate(state=initial_state, trace=tuple(), score=adapter.current_hpwl(initial_state))]

    while True:
        next_frontier: list[SearchCandidate] = []
        expanded = False
        for candidate in frontier:
            step_index = len(candidate.state.placed_macros)
            current_macro = adapter.current_macro_for_step(step_index)
            if current_macro is None:
                next_frontier.append(candidate)
                continue

            action_mask = adapter.build_action_mask(candidate.state, step_index=step_index, grid_shape=grid_shape)
            for row_index in range(grid_shape[0]):
                for col_index in range(grid_shape[1]):
                    if bool(action_mask[row_index, col_index]):
                        continue
                    expanded = True
                    target = adapter.grid_index_to_coordinate((row_index, col_index), grid_shape)
                    action = {"macro": current_macro, "target": target}
                    next_state = adapter.apply_action(candidate.state, action)
                    next_trace = candidate.trace + ({"action": action, "step_index": step_index},)
                    next_frontier.append(
                        SearchCandidate(
                            state=next_state,
                            trace=next_trace,
                            score=adapter.current_hpwl(next_state),
                        )
                    )

        if not expanded:
            break

        next_frontier.sort(key=lambda candidate: candidate.score)
        frontier = next_frontier[: max(1, beam_width)]

    best = min(frontier, key=lambda candidate: candidate.score)
    return {
        "steps": len(best.trace),
        "trace": list(best.trace),
        "invalid_action_count": 0,
        "final_hpwl": adapter.current_hpwl(best.state),
    }


def build_demonstrations_from_trace(
    *,
    problem: ChipProblem,
    grid_shape: tuple[int, int],
    rollout_type: str,
    trace: list[dict[str, object]],
    final_score: float,
) -> list[DemonstrationStep]:
    adapter = ChipRLAdapter(problem=problem, grid_shape=grid_shape)
    state = adapter.initial_state()
    demonstrations: list[DemonstrationStep] = []

    for event in trace:
        step_index = len(state.placed_macros)
        action = dict(event["action"])
        observation = adapter.observe(state, step_index=step_index)
        action_mask = observation["action_mask"].reshape(-1).tolist()
        action_index = _action_to_flat_index(adapter, action["target"], grid_shape)
        demonstrations.append(
            DemonstrationStep(
                design="chip",
                rollout_type=rollout_type,
                step_index=step_index,
                current_macro=str(observation["current_macro"]),
                action_index=action_index,
                action_mask=[bool(value) for value in action_mask],
                final_score=float(final_score),
                placement=dict(state.placement),
            )
        )
        state = adapter.apply_action(state, action)

    return demonstrations


def _action_to_flat_index(adapter: ChipRLAdapter, target: tuple[float, float], grid_shape: tuple[int, int]) -> int:
    row_index, col_index = adapter.coordinate_to_grid_index(target, grid_shape)
    return row_index * grid_shape[1] + col_index
