from __future__ import annotations

from pathlib import Path

import torch

from chip.experiments.phase3_imitation import train_behavior_cloning_policy
from chip.experiments.phase3_ppo import collect_policy_rollout
from chip.experiments.phase3_ppo import initialize_policy_for_problem
from chip.experiments.phase3_ppo import load_policy_warm_start
from chip.experiments.phase3_ppo import run_single_ppo_update
from chip.experiments.phase3_ppo import train_ppo_policy
from chip.experiments.phase3_search import build_demonstrations_from_trace
from chip.experiments.phase3_search import run_beam_search_baseline
from engine.ppo_policy import PPOPlacementPolicy
from tests.chip.test_rl_adapter import build_problem


def test_load_policy_warm_start_ignores_missing_value_head(tmp_path: Path):
    policy = PPOPlacementPolicy(
        node_feature_dim=4,
        graph_feature_dim=0,
        grid_channels=6,
        grid_shape=(4, 4),
    )
    checkpoint = {
        "policy_state_dict": {
            key: value
            for key, value in policy.state_dict().items()
            if not key.startswith("value_head.")
        }
    }
    path = tmp_path / "bc.pt"
    torch.save(checkpoint, path)

    loaded = load_policy_warm_start(policy, path)

    assert loaded is True


def test_load_policy_warm_start_does_not_overwrite_value_head(tmp_path: Path):
    policy = PPOPlacementPolicy(
        node_feature_dim=4,
        graph_feature_dim=0,
        grid_channels=6,
        grid_shape=(4, 4),
    )
    original_value_head = {
        key: value.clone()
        for key, value in policy.state_dict().items()
        if key.startswith("value_head.")
    }
    checkpoint = {"policy_state_dict": policy.state_dict()}
    for key in list(checkpoint["policy_state_dict"].keys()):
        if key.startswith("value_head."):
            checkpoint["policy_state_dict"][key] = torch.zeros_like(checkpoint["policy_state_dict"][key])
    path = tmp_path / "bc_full.pt"
    torch.save(checkpoint, path)

    loaded = load_policy_warm_start(policy, path)

    assert loaded is True
    for key, value in original_value_head.items():
        assert torch.equal(policy.state_dict()[key], value)


def test_collect_policy_rollout_only_emits_valid_actions():
    policy = initialize_policy_for_problem(problem=build_problem(), grid_shape=(4, 4), hidden_dim=16)

    rollout = collect_policy_rollout(
        problem=build_problem(),
        policy=policy,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=1,
        seed=0,
    )

    assert rollout["invalid_action_count"] == 0
    assert rollout["steps"] == 1


def test_run_single_ppo_update_returns_finite_metrics():
    policy = initialize_policy_for_problem(problem=build_problem(), grid_shape=(4, 4), hidden_dim=16)

    result = run_single_ppo_update(
        problem=build_problem(),
        policy=policy,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=1,
        learning_rate=1e-3,
        seed=0,
    )

    assert torch.isfinite(torch.tensor(result["policy_loss"]))
    assert torch.isfinite(torch.tensor(result["value_loss"]))
    assert result["steps"] == 1


def test_behavior_cloning_checkpoint_can_warm_start_ppo_policy(tmp_path: Path):
    problem = build_problem()
    search_result = run_beam_search_baseline(problem=problem, grid_shape=(4, 4), beam_width=2)
    demonstrations = build_demonstrations_from_trace(
        problem=problem,
        grid_shape=(4, 4),
        rollout_type="beam_search",
        trace=search_result["trace"],
        final_score=search_result["final_hpwl"],
    )
    checkpoint_path = tmp_path / "bc_policy.pt"
    train_behavior_cloning_policy(
        problem=problem,
        demonstrations=demonstrations,
        grid_shape=(4, 4),
        hidden_dim=16,
        epochs=120,
        learning_rate=1e-2,
        checkpoint_path=checkpoint_path,
    )
    policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)

    loaded = load_policy_warm_start(policy, checkpoint_path)
    rollout = collect_policy_rollout(
        problem=problem,
        policy=policy,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=1,
        seed=0,
    )

    assert loaded is True
    assert rollout["invalid_action_count"] == 0


def test_train_ppo_policy_records_multi_update_history():
    policy = initialize_policy_for_problem(problem=build_problem(), grid_shape=(4, 4), hidden_dim=16)

    result = train_ppo_policy(
        problem=build_problem(),
        policy=policy,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=1,
        num_updates=3,
        learning_rate=1e-3,
        seed=0,
    )

    assert len(result["history"]) == 3
    assert result["final_rollout"]["invalid_action_count"] == 0
    assert all(torch.isfinite(torch.tensor(item["policy_loss"])) for item in result["history"])
