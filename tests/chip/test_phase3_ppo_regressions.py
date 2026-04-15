from __future__ import annotations

from chip.experiments.phase3_ppo import collect_rollout_for_action_trace
from chip.experiments.phase3_ppo import initialize_policy_for_problem
from tests.chip.test_rl_adapter import build_problem
from tests.chip.test_rl_adapter import build_two_movable_problem


def test_collect_rollout_for_action_trace_does_not_mark_partial_trace_as_completed():
    problem = build_two_movable_problem()
    policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)

    rollout = collect_rollout_for_action_trace(
        problem=problem,
        policy=policy,
        action_trace=[{"action": {"macro": "u1", "target": (1.0, 0.0)}}],
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=2,
    )

    assert rollout.completed_episode is False
    assert rollout.termination_reason != "completed"
