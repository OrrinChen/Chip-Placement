from __future__ import annotations

import copy
from pathlib import Path

import torch

from chip.experiments.phase3_imitation import train_behavior_cloning_policy
from chip.experiments.phase3_ppo import collect_policy_rollout
from chip.experiments.phase3_ppo import collect_rollout_for_action_trace
from chip.experiments.phase3_ppo import compute_discounted_returns
from chip.experiments.phase3_ppo import FixedRolloutBatch
from chip.experiments.phase3_ppo import initialize_policy_for_problem
from chip.experiments.phase3_ppo import load_policy_warm_start
from chip.experiments.phase3_ppo import normalize_advantages
from chip.experiments.phase3_ppo import prefit_value_head_from_rollout
from chip.experiments.phase3_ppo import prefit_value_head_from_rollouts
from chip.experiments.phase3_ppo import overfit_fixed_rollout
from chip.experiments.phase3_ppo import run_single_ppo_update
from chip.experiments.phase3_ppo import summarize_rollout_return_signal
from chip.experiments.phase3_ppo import train_ppo_policy
from chip.experiments.phase3_search import build_demonstrations_from_trace
from chip.experiments.phase3_search import run_beam_search_baseline
from chip.rl_adapter import ChipRLAdapter
from engine.ppo_policy import PPOPlacementPolicy
from tests.chip.test_rl_adapter import build_problem
from tests.chip.test_rl_adapter import build_two_movable_problem


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
    assert rollout["completed_episode"] is True


def test_collect_policy_rollout_supports_uniform_legal_policy():
    policy = initialize_policy_for_problem(problem=build_problem(), grid_shape=(4, 4), hidden_dim=16)

    rollout = collect_policy_rollout(
        problem=build_problem(),
        policy=policy,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=1,
        seed=0,
        uniform_legal_policy=True,
    )

    assert rollout["invalid_action_count"] == 0
    assert rollout["completed_episode"] is True


def test_collect_policy_rollout_supports_initial_prefix_state():
    problem = build_two_movable_problem()
    policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)
    adapter = ChipRLAdapter(problem=problem, grid_shape=(4, 4))
    initial_state = adapter.apply_action(adapter.initial_state(), {"macro": "u1", "target": (4.0, 0.0)})

    rollout = collect_policy_rollout(
        problem=problem,
        policy=policy,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=2,
        seed=0,
        initial_state=initial_state,
    )

    assert rollout["steps"] == 1
    assert rollout["completed_episode"] is True


def test_collect_policy_rollout_treats_completed_prefix_as_zero_step_terminal():
    problem = build_problem()
    policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)
    adapter = ChipRLAdapter(problem=problem, grid_shape=(4, 4))
    initial_state = adapter.apply_action(adapter.initial_state(), {"macro": "u1", "target": (4.0, 0.0)})

    rollout = collect_policy_rollout(
        problem=problem,
        policy=policy,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=2,
        seed=0,
        initial_state=initial_state,
    )

    assert rollout["steps"] == 0
    assert rollout["completed_episode"] is True


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
    assert torch.isfinite(torch.tensor(result["approx_kl"]))
    assert result["kl_anchor_loss"] == 0.0
    assert result["kl_anchor_beta"] == 0.0
    assert result["steps"] == 1
    assert result["completed_episode"] == 1.0


def test_run_single_ppo_update_reports_positive_kl_anchor_loss_with_reference_policy():
    problem = build_two_movable_problem()
    reference_policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)
    policy = copy.deepcopy(reference_policy)
    with torch.no_grad():
        policy.policy_head.bias[0] += 0.5

    result = run_single_ppo_update(
        problem=problem,
        policy=policy,
        reference_policy=reference_policy,
        kl_anchor_beta=5.0,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=2,
        learning_rate=1e-3,
        seed=0,
    )

    assert result["kl_anchor_beta"] == 5.0
    assert result["kl_anchor_loss"] > 0.0
    assert torch.isfinite(torch.tensor(result["kl_anchor_loss"]))


def test_run_single_ppo_update_records_pre_update_and_substep_diagnostics():
    problem = build_two_movable_problem()
    reference_policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)
    policy = copy.deepcopy(reference_policy)

    result = run_single_ppo_update(
        problem=problem,
        policy=policy,
        reference_policy=reference_policy,
        kl_anchor_beta=5.0,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=2,
        learning_rate=1e-3,
        seed=0,
        policy_epochs=3,
        diagnostic_eval_seeds=(10, 11),
    )

    assert result["pre_update_diagnostic"]["heldout_rollout_count"] == 2
    assert torch.isfinite(torch.tensor(result["pre_update_diagnostic"]["heldout_mean_hpwl"]))
    assert result["batch_diagnostic"]["return_summary"]["count"] == 2
    assert result["batch_diagnostic"]["raw_advantage_summary"]["count"] == 2
    assert result["batch_diagnostic"]["normalized_advantage_summary"]["count"] == 2
    assert result["batch_diagnostic"]["top_abs_advantages"][0]["transition_index"] in {0, 1}
    assert "value_return_correlation" in result["batch_diagnostic"]
    assert len(result["substep_history"]) == 3
    assert [row["substep_index"] for row in result["substep_history"]] == [0, 1, 2]
    assert all("update_batch_kl_to_reference" in row for row in result["substep_history"])
    assert all("heldout_completion_rate" in row for row in result["substep_history"])
    assert all("heldout_log_prob_delta_mean" in row for row in result["substep_history"])
    assert all("global_grad_norm" in row for row in result["substep_history"])
    assert all("policy_head_grad_norm" in row for row in result["substep_history"])
    assert all("value_head_grad_norm" in row for row in result["substep_history"])
    assert all(torch.isfinite(torch.tensor(row["update_batch_kl_to_reference"])) for row in result["substep_history"])


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
    assert "final_hpwl" in result["history"][0]
    assert "best_hpwl_so_far" in result["history"][0]
    assert "invalid_action_count" in result["history"][0]
    assert result["final_rollout"]["invalid_action_count"] == 0
    assert all(torch.isfinite(torch.tensor(item["policy_loss"])) for item in result["history"])


def test_collect_rollout_for_action_trace_replays_teacher_actions():
    problem = build_problem()
    policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)

    rollout = collect_rollout_for_action_trace(
        problem=problem,
        policy=policy,
        action_trace=[{"action": {"macro": "u1", "target": (4.0, 0.0)}}],
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=1,
    )

    assert len(rollout.transitions) == 1
    assert rollout.completed_episode is True


def test_overfit_fixed_rollout_improves_return_on_tiny_problem():
    problem = build_problem()
    teacher_policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)
    fixed_rollout = collect_rollout_for_action_trace(
        problem=problem,
        policy=teacher_policy,
        action_trace=[{"action": {"macro": "u1", "target": (4.0, 0.0)}}],
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=1,
    )
    policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)

    result = overfit_fixed_rollout(
        problem=problem,
        policy=policy,
        fixed_rollout=fixed_rollout,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=1,
        num_updates=25,
        eval_every=5,
        learning_rate=1e-2,
        seed=0,
    )

    assert result["final_eval"]["episode_return"] >= result["initial_eval"]["episode_return"]


def test_normalize_advantages_zero_centers_non_constant_vector():
    advantages = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)

    normalized = normalize_advantages(advantages)

    assert torch.isfinite(normalized).all()
    assert torch.allclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(normalized.std(unbiased=False), torch.tensor(1.0), atol=1e-5)


def test_normalize_advantages_returns_zero_vector_for_constant_input():
    advantages = torch.tensor([3.5, 3.5, 3.5], dtype=torch.float32)

    normalized = normalize_advantages(advantages)

    assert torch.equal(normalized, torch.zeros_like(advantages))


def test_discounted_returns_feed_finite_normalized_advantages():
    returns = compute_discounted_returns([1.0, 0.5, -0.5], gamma=0.99)
    baseline = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    normalized = normalize_advantages(returns - baseline)

    assert torch.isfinite(normalized).all()


def test_prefit_value_head_only_updates_value_parameters():
    problem = build_problem()
    policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)
    fixed_rollout = collect_rollout_for_action_trace(
        problem=problem,
        policy=policy,
        action_trace=[{"action": {"macro": "u1", "target": (4.0, 0.0)}}],
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=1,
    )
    before = {name: tensor.clone() for name, tensor in policy.state_dict().items()}

    metrics = prefit_value_head_from_rollout(
        policy=policy,
        fixed_rollout=fixed_rollout,
        epochs=20,
        learning_rate=1e-2,
    )

    assert metrics["final_loss"] <= metrics["initial_loss"]
    changed_value_head = False
    for name, tensor in policy.state_dict().items():
        if name.startswith("value_head."):
            if not torch.equal(tensor, before[name]):
                changed_value_head = True
        else:
            assert torch.equal(tensor, before[name])
    assert changed_value_head is True


def test_prefit_value_head_from_rollouts_supports_zscore_targets_and_warmup():
    problem = build_two_movable_problem()
    policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)
    rollouts = [
        collect_policy_rollout(
            problem=problem,
            policy=policy,
            grid_shape=(4, 4),
            reward_source="analytic_hpwl",
            max_steps=2,
            seed=seed,
        )
        for seed in (0, 1)
    ]
    fixed_rollouts = [
        FixedRolloutBatch(
            transitions=rollout["transitions"],
            schema=rollout["schema"],
            final_hpwl=float(rollout["final_hpwl"]),
            episode_return=float(rollout["episode_return"]),
            termination_reason=str(rollout["termination_reason"]),
            completed_episode=bool(rollout["completed_episode"]),
        )
        for rollout in rollouts
    ]

    metrics = prefit_value_head_from_rollouts(
        policy=policy,
        fixed_rollouts=fixed_rollouts,
        learning_rate=1e-3,
        epochs=5,
        lr_warmup_epochs=2,
        target_transform="zscore",
    )

    assert metrics["target_transform"] == "zscore"
    assert metrics["lr_warmup_epochs"] == 2.0
    assert metrics["rollout_count"] == 2.0
    assert metrics["transition_count"] == 4.0
    assert metrics["raw_return_summary"]["count"] == 4.0
    assert metrics["target_summary"]["count"] == 4.0
    assert torch.isfinite(torch.tensor(metrics["initial_value_return_correlation"]))
    assert torch.isfinite(torch.tensor(metrics["final_value_return_correlation"]))


def test_summarize_rollout_return_signal_reports_simple_state_correlations():
    problem = build_two_movable_problem()
    policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)
    rollouts = [
        collect_policy_rollout(
            problem=problem,
            policy=policy,
            grid_shape=(4, 4),
            reward_source="analytic_hpwl",
            max_steps=2,
            seed=seed,
        )
        for seed in (0, 1)
    ]
    fixed_rollouts = [
        FixedRolloutBatch(
            transitions=rollout["transitions"],
            schema=rollout["schema"],
            final_hpwl=float(rollout["final_hpwl"]),
            episode_return=float(rollout["episode_return"]),
            termination_reason=str(rollout["termination_reason"]),
            completed_episode=bool(rollout["completed_episode"]),
        )
        for rollout in rollouts
    ]

    summary = summarize_rollout_return_signal(
        fixed_rollouts=fixed_rollouts,
        total_macro_count=2,
    )

    assert summary["rollout_count"] == 2.0
    assert summary["transition_count"] == 4.0
    assert "return_to_go_vs_step_index" in summary
    assert "return_to_go_vs_remaining_macros" in summary
    assert "return_to_go_vs_current_hpwl" in summary
    assert "episode_return_vs_final_hpwl" in summary
    assert torch.isfinite(torch.tensor(summary["return_to_go_vs_step_index"]))
