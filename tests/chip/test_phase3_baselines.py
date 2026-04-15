from __future__ import annotations

import torch

from chip.experiments.phase3_baselines import run_greedy_episode
from chip.experiments.phase3_baselines import run_greedy_hpwl_baseline
from chip.experiments.phase3_baselines import run_masked_random_episode
from chip.experiments.phase3_baselines import run_masked_random_baseline
from chip.experiments.phase3_baselines import summarize_episode_metrics
from chip.problem import ChipProblem
from chip.problem import DieArea
from chip.problem import Macro
from chip.problem import Net
from chip.problem import PlacementRow
from chip.rl_adapter import ChipRLAdapter
from tests.chip.test_rl_adapter import build_problem
from tests.chip.test_rl_adapter import build_two_movable_problem


def test_masked_random_baseline_only_emits_valid_actions():
    result = run_masked_random_baseline(problem=build_problem(), grid_shape=(4, 4), steps=2, seed=7)

    assert result["invalid_action_count"] == 0
    assert result["steps"] == 1


def test_greedy_baseline_improves_over_random_on_tiny_problem():
    greedy = run_greedy_hpwl_baseline(problem=build_problem(), grid_shape=(4, 4))
    random_result = run_masked_random_baseline(problem=build_problem(), grid_shape=(4, 4), steps=1, seed=0)

    assert greedy["final_hpwl"] <= random_result["final_hpwl"]


def test_masked_random_episode_reports_return_legality_and_runtime():
    result = run_masked_random_episode(problem=build_problem(), grid_shape=(4, 4), seed=3)

    assert result["steps"] == 1
    assert "episode_return" in result
    assert result["is_legal"] is True
    assert result["runtime_seconds"] >= 0.0


def test_greedy_episode_beats_masked_random_episode_on_tiny_problem():
    greedy = run_greedy_episode(problem=build_problem(), grid_shape=(4, 4))
    random_result = run_masked_random_episode(problem=build_problem(), grid_shape=(4, 4), seed=0)

    assert greedy["final_hpwl"] <= random_result["final_hpwl"]


def test_summarize_episode_metrics_computes_legality_rate():
    summary = summarize_episode_metrics(
        [
            {"episode_return": 1.0, "final_hpwl": 10.0, "is_legal": True, "runtime_seconds": 2.0},
            {"episode_return": -2.0, "final_hpwl": 20.0, "is_legal": False, "runtime_seconds": 4.0},
        ]
    )

    assert summary["episode_count"] == 2
    assert summary["legality_rate"] == 0.5
    assert summary["episode_return"]["mean"] == -0.5


def test_masked_random_episode_handles_dead_end_without_raising(monkeypatch):
    problem = ChipProblem(
        die_area=DieArea(min_x=0.0, min_y=0.0, max_x=12.0, max_y=4.0),
        macros={
            "u1": Macro(name="u1", kind="CELL", width=1.0, height=1.0, x=0.0, y=0.0, is_fixed=False),
            "u2": Macro(name="u2", kind="CELL", width=1.0, height=1.0, x=8.0, y=0.0, is_fixed=False),
        },
        nets=[Net(name="n1", pins=("u1", "u2"))],
        rows=(
            PlacementRow(name="ROW_0", origin_x=0.0, origin_y=0.0, orientation="N", site_count=12, step_x=1.0, step_y=0.0),
            PlacementRow(name="ROW_1", origin_x=0.0, origin_y=2.0, orientation="FS", site_count=12, step_x=1.0, step_y=0.0),
        ),
    )
    original_build_action_mask = ChipRLAdapter.build_action_mask

    def dead_end_after_first_step(self, state, *, step_index, grid_shape):
        if step_index >= 1:
            return torch.ones(grid_shape, dtype=torch.bool)
        return original_build_action_mask(self, state, step_index=step_index, grid_shape=grid_shape)

    monkeypatch.setattr(ChipRLAdapter, "build_action_mask", dead_end_after_first_step)

    result = run_masked_random_episode(problem=problem, grid_shape=(4, 4), seed=0)

    assert result["steps"] == 1
    assert result["termination_reason"] == "no_valid_actions"


def test_greedy_episode_trace_uses_pre_action_step_indices():
    result = run_greedy_episode(problem=build_two_movable_problem(), grid_shape=(1, 6))

    assert [row["step_index"] for row in result["trace"]] == [0, 1]
