from __future__ import annotations

from chip.experiments.phase3_random_agent import run_random_agent_smoke
from tests.chip.test_rl_adapter import build_problem


def test_random_agent_smoke_runs_episode_and_returns_trace():
    result = run_random_agent_smoke(problem=build_problem(), steps=3, reward_source="analytic_hpwl", seed=7)

    assert result["steps"] == 1
    assert len(result["trace"]) == 1
    assert "final_hpwl" in result
    assert result["trace"][-1]["truncated"] is False
