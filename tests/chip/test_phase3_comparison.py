from __future__ import annotations

from chip.experiments.phase3_comparison import run_phase3_comparison
from tests.chip.test_rl_adapter import build_problem


def test_phase3_comparison_returns_all_expected_methods():
    report = run_phase3_comparison(problem=build_problem(), grid_shape=(4, 4), hidden_dim=16, ppo_epochs=1)

    assert set(report["methods"]) == {
        "random",
        "greedy",
        "beam",
        "behavior_cloning",
        "ppo_scratch",
        "ppo_warm_start",
    }


def test_phase3_comparison_preserves_basic_ordering_sanity():
    report = run_phase3_comparison(problem=build_problem(), grid_shape=(4, 4), hidden_dim=16, ppo_epochs=1)
    methods = report["methods"]

    assert methods["beam"]["final_hpwl"] <= methods["greedy"]["final_hpwl"]
    assert methods["behavior_cloning"]["final_hpwl"] <= methods["random"]["final_hpwl"]
    assert methods["ppo_scratch"]["invalid_action_count"] == 0
    assert methods["ppo_warm_start"]["invalid_action_count"] == 0


def test_phase3_comparison_reports_ppo_update_count():
    report = run_phase3_comparison(problem=build_problem(), grid_shape=(4, 4), hidden_dim=16, ppo_epochs=3)
    methods = report["methods"]

    assert methods["ppo_scratch"]["update_count"] == 3.0
    assert methods["ppo_warm_start"]["update_count"] == 3.0
    assert len(methods["ppo_scratch"]["curve"]) == 3
    assert "updates_to_beam_level" in methods["ppo_warm_start"]


def test_phase3_comparison_supports_multiple_seeds():
    report = run_phase3_comparison(
        problem=build_problem(),
        grid_shape=(4, 4),
        hidden_dim=16,
        ppo_epochs=2,
        seeds=(0, 1),
    )
    methods = report["methods"]

    assert report["seed_count"] == 2
    assert len(methods["ppo_warm_start"]["per_seed"]) == 2
    assert "aggregate" in methods["ppo_warm_start"]
