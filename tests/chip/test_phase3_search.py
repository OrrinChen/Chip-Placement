from __future__ import annotations

from chip.experiments.phase3_baselines import run_greedy_hpwl_baseline
from chip.experiments.phase3_search import build_demonstrations_from_trace
from chip.experiments.phase3_search import run_beam_search_baseline
from tests.chip.test_rl_adapter import build_problem


def test_beam_search_baseline_only_emits_valid_actions():
    result = run_beam_search_baseline(problem=build_problem(), grid_shape=(4, 4), beam_width=2)

    assert result["invalid_action_count"] == 0
    assert result["steps"] == 1


def test_beam_search_matches_or_beats_greedy_on_tiny_problem():
    beam = run_beam_search_baseline(problem=build_problem(), grid_shape=(4, 4), beam_width=2)
    greedy = run_greedy_hpwl_baseline(problem=build_problem(), grid_shape=(4, 4))

    assert beam["final_hpwl"] <= greedy["final_hpwl"]


def test_beam_search_trace_can_be_exported_as_demonstrations():
    result = run_beam_search_baseline(problem=build_problem(), grid_shape=(4, 4), beam_width=2)

    demonstrations = build_demonstrations_from_trace(
        problem=build_problem(),
        grid_shape=(4, 4),
        rollout_type="beam_search",
        trace=result["trace"],
        final_score=result["final_hpwl"],
    )

    assert len(demonstrations) == 1
    assert demonstrations[0].current_macro == "u1"
    assert len(demonstrations[0].action_mask) == 16
