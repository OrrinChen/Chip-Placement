from __future__ import annotations

from chip.experiments.phase3_baselines import run_greedy_hpwl_baseline
from chip.experiments.phase3_baselines import run_masked_random_baseline
from tests.chip.test_rl_adapter import build_problem


def test_masked_random_baseline_only_emits_valid_actions():
    result = run_masked_random_baseline(problem=build_problem(), grid_shape=(4, 4), steps=2, seed=7)

    assert result["invalid_action_count"] == 0
    assert result["steps"] == 1


def test_greedy_baseline_improves_over_random_on_tiny_problem():
    greedy = run_greedy_hpwl_baseline(problem=build_problem(), grid_shape=(4, 4))
    random_result = run_masked_random_baseline(problem=build_problem(), grid_shape=(4, 4), steps=1, seed=0)

    assert greedy["final_hpwl"] <= random_result["final_hpwl"]
