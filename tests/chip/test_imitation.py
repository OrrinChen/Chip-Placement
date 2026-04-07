from __future__ import annotations

from pathlib import Path

from chip.imitation import DemonstrationStep
from chip.imitation import load_demonstrations
from chip.imitation import observation_from_demonstration
from chip.imitation import save_demonstrations
from chip.experiments.phase3_search import build_demonstrations_from_trace
from chip.experiments.phase3_search import run_beam_search_baseline
from tests.chip.test_rl_adapter import build_problem


def test_demonstration_round_trip(tmp_path: Path):
    steps = [
        DemonstrationStep(
            design="toy",
            rollout_type="beam_search",
            step_index=0,
            current_macro="u1",
            action_index=3,
            action_mask=[False, False, True, False],
            final_score=12.5,
            placement={"u1": (0.0, 0.0), "u2": (8.0, 0.0)},
        )
    ]

    path = tmp_path / "demo.json"
    save_demonstrations(path, steps)
    loaded = load_demonstrations(path)

    assert loaded == steps


def test_observation_from_demonstration_reconstructs_adapter_view():
    problem = build_problem()
    search_result = run_beam_search_baseline(problem=problem, grid_shape=(4, 4), beam_width=2)
    demonstration = build_demonstrations_from_trace(
        problem=problem,
        grid_shape=(4, 4),
        rollout_type="beam_search",
        trace=search_result["trace"],
        final_score=search_result["final_hpwl"],
    )[0]

    observation = observation_from_demonstration(problem=problem, demonstration=demonstration, grid_shape=(4, 4))

    assert observation["current_macro"] == "u1"
    assert observation["step_index"] == 0
    assert tuple(observation["action_mask"].shape) == (4, 4)
