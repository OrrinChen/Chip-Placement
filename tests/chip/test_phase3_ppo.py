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
from chip.experiments.phase3_ppo import summarize_argmax_agreement_for_fixed_rollout
from chip.experiments.phase3_ppo import summarize_rollout_return_signal
from chip.experiments.phase3_ppo import train_ppo_policy
from chip.experiments.phase3_search import build_demonstrations_from_trace
from chip.experiments.phase3_search import run_beam_search_baseline
from chip.problem import ChipProblem
from chip.problem import DieArea
from chip.problem import Macro
from chip.problem import Net
from chip.problem import PlacementRow
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


def test_summarize_argmax_agreement_for_fixed_rollout_reports_flips_entropy_and_buckets():
    problem = ChipProblem(
        die_area=DieArea(min_x=0.0, min_y=0.0, max_x=16.0, max_y=4.0),
        macros={
            "fixed": Macro(name="fixed", kind="PIN", width=0.0, height=0.0, x=15.0, y=2.0, is_fixed=True),
            "large": Macro(name="large", kind="CELL", width=3.0, height=1.0, x=0.0, y=0.0),
            "medium": Macro(name="medium", kind="CELL", width=2.0, height=1.0, x=4.0, y=0.0),
            "small": Macro(name="small", kind="CELL", width=1.0, height=1.0, x=8.0, y=0.0),
        },
        nets=[Net(name="n1", pins=("fixed", "large", "medium", "small"))],
        rows=(
            PlacementRow(name="ROW_0", origin_x=0.0, origin_y=0.0, orientation="N", site_count=16, step_x=1.0, step_y=0.0),
            PlacementRow(name="ROW_1", origin_x=0.0, origin_y=2.0, orientation="FS", site_count=16, step_x=1.0, step_y=0.0),
        ),
    )
    policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)
    reference_rollout = collect_policy_rollout(
        problem=problem,
        policy=policy,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=3,
        seed=0,
        deterministic=True,
    )
    fixed_rollout = FixedRolloutBatch(
        transitions=reference_rollout["transitions"],
        schema=reference_rollout["schema"],
        final_hpwl=float(reference_rollout["final_hpwl"]),
        episode_return=float(reference_rollout["episode_return"]),
        termination_reason=str(reference_rollout["termination_reason"]),
        completed_episode=bool(reference_rollout["completed_episode"]),
    )

    class _FixedLogitPolicy:
        def __init__(self, rows):
            self.rows = rows

        def __call__(self, *, graph_batch, canvas, current_macro_index, action_mask):
            del graph_batch, canvas, current_macro_index
            flat_mask = action_mask.view(action_mask.shape[0], -1)
            logits = torch.full((flat_mask.shape[0], flat_mask.shape[1]), -5.0, dtype=torch.float32)
            logits = logits.masked_fill(flat_mask, float("-inf"))
            for row_index, row_spec in enumerate(self.rows):
                logits[row_index, row_spec["top"]] = row_spec["top_logit"]
                logits[row_index, row_spec["second"]] = row_spec["second_logit"]
            return logits, torch.zeros((flat_mask.shape[0], 1), dtype=torch.float32)

    masks = [step.observation["action_mask"].reshape(-1) for step in fixed_rollout.transitions]
    legal_indices = [(~mask).nonzero(as_tuple=False).view(-1).tolist() for mask in masks]
    reference_policy = _FixedLogitPolicy(
        [
            {"top": legal_indices[0][0], "second": legal_indices[0][1], "top_logit": 4.0, "second_logit": 0.0},
            {"top": legal_indices[1][0], "second": legal_indices[1][1], "top_logit": 1.01, "second_logit": 1.0},
            {"top": legal_indices[2][0], "second": legal_indices[2][1], "top_logit": 3.5, "second_logit": 0.0},
        ]
    )
    updated_policy = _FixedLogitPolicy(
        [
            {"top": legal_indices[0][0], "second": legal_indices[0][1], "top_logit": 4.0, "second_logit": 0.0},
            {"top": legal_indices[1][1], "second": legal_indices[1][0], "top_logit": 1.01, "second_logit": 1.0},
            {"top": legal_indices[2][0], "second": legal_indices[2][1], "top_logit": 3.5, "second_logit": 0.0},
        ]
    )

    summary = summarize_argmax_agreement_for_fixed_rollout(
        reference_policy=reference_policy,
        candidate_policy=updated_policy,
        fixed_rollout=fixed_rollout,
    )

    assert summary["step_count"] == 3
    assert summary["agreement_count"] == 2
    assert summary["flip_count"] == 1
    assert torch.isclose(torch.tensor(summary["overall_agreement_rate"]), torch.tensor(2.0 / 3.0))
    assert summary["flipped_step_indices"] == [1]
    assert summary["position_buckets"]["early"]["step_count"] == 1
    assert summary["position_buckets"]["mid"]["step_count"] == 1
    assert summary["position_buckets"]["late"]["step_count"] == 1
    assert summary["flipped_entropy_summary"]["count"] == 1.0
    assert summary["non_flipped_entropy_summary"]["count"] == 2.0
    assert summary["flipped_entropy_summary"]["mean"] > summary["non_flipped_entropy_summary"]["mean"]
    assert summary["flipped_margin_summary"]["mean"] < summary["non_flipped_margin_summary"]["mean"]
    assert summary["flipped_margin_histogram"]["lt_0_05"]["count"] == 1
    assert summary["non_flipped_margin_histogram"]["gte_0_50"]["count"] == 2
    assert summary["macro_area_buckets"]["large"]["step_count"] == 1
    assert summary["macro_area_buckets"]["large"]["agreement_count"] == 1
    assert summary["macro_area_buckets"]["medium"]["step_count"] == 1
    assert summary["macro_area_buckets"]["medium"]["flip_count"] == 1
    assert summary["macro_area_buckets"]["small"]["step_count"] == 1
    assert summary["macro_area_buckets"]["small"]["agreement_count"] == 1


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
    assert torch.isfinite(torch.tensor(result["value_loss_train_space"]))
    assert torch.isfinite(torch.tensor(result["value_loss_raw_space"]))
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
    assert result["batch_diagnostic"]["old_value_summary"]["count"] == 2
    assert result["batch_diagnostic"]["raw_advantage_summary"]["count"] == 2
    assert result["batch_diagnostic"]["normalized_advantage_summary"]["count"] == 2
    assert result["batch_diagnostic"]["top_abs_advantages"][0]["transition_index"] in {0, 1}
    assert "value_return_correlation" in result["batch_diagnostic"]
    assert result["batch_diagnostic"]["advantage_gate"]["std_gt_epsilon"] is True
    assert result["batch_diagnostic"]["advantage_gate"]["pass"] is True
    assert result["value_target_stats"]["mean"] == 0.0
    assert result["value_target_stats"]["std"] == 1.0
    assert "pre_update_metrics" in result
    assert torch.isfinite(torch.tensor(result["pre_update_metrics"]["entropy"]))
    assert len(result["substep_history"]) == 3
    assert [row["substep_index"] for row in result["substep_history"]] == [0, 1, 2]
    assert all("update_batch_kl_to_reference" in row for row in result["substep_history"])
    assert all("heldout_completion_rate" in row for row in result["substep_history"])
    assert all("heldout_log_prob_delta_mean" in row for row in result["substep_history"])
    assert all("global_grad_norm" in row for row in result["substep_history"])
    assert all("policy_head_grad_norm" in row for row in result["substep_history"])
    assert all("value_head_grad_norm" in row for row in result["substep_history"])
    assert all("value_loss_train_space" in row for row in result["substep_history"])
    assert all("value_loss_raw_space" in row for row in result["substep_history"])
    assert all("value_target_mean" in row for row in result["substep_history"])
    assert all("value_target_std" in row for row in result["substep_history"])
    assert all("grad_clip_norm" in row for row in result["substep_history"])
    assert all("global_grad_norm_pre_clip" in row for row in result["substep_history"])
    assert all("global_grad_norm_post_clip" in row for row in result["substep_history"])
    assert all("grad_clip_applied" in row for row in result["substep_history"])
    assert all("effective_learning_rate" in row for row in result["substep_history"])
    assert all("first_substep_lr_scale" in row for row in result["substep_history"])
    assert all("global_param_delta_norm_from_pre_update" in row for row in result["substep_history"])
    assert all("policy_head_param_delta_norm_from_pre_update" in row for row in result["substep_history"])
    assert all("ratio_fraction_gt_1_5" in row for row in result["substep_history"])
    assert all("ratio_fraction_lt_0_67" in row for row in result["substep_history"])
    assert all("ratio_max" in row for row in result["substep_history"])
    assert all("approx_kl_abs_top10pct_fraction" in row for row in result["substep_history"])
    assert all("entropy_delta_from_pre_update" in row for row in result["substep_history"])
    assert all(torch.isfinite(torch.tensor(row["update_batch_kl_to_reference"])) for row in result["substep_history"])
    assert all(row["grad_clip_applied"] is False for row in result["substep_history"])
    assert all(torch.isclose(torch.tensor(row["effective_learning_rate"]), torch.tensor(1e-3)) for row in result["substep_history"])
    assert all(row["first_substep_lr_scale"] is None for row in result["substep_history"])


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
    assert torch.allclose(policy.value_target_mean, torch.tensor(metrics["target_mean"]))
    assert torch.allclose(policy.value_target_std, torch.tensor(metrics["target_std"]))


def test_prefit_value_head_from_rollouts_keeps_identity_scale_for_raw_targets():
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
        target_transform="raw",
    )

    assert metrics["target_transform"] == "raw"
    assert torch.allclose(policy.value_target_mean, torch.tensor(0.0))
    assert torch.allclose(policy.value_target_std, torch.tensor(1.0))


def test_run_single_ppo_update_is_invariant_under_identity_value_scale():
    problem = build_two_movable_problem()
    base_policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)
    identity_policy = copy.deepcopy(base_policy)
    identity_policy.set_value_target_stats(mean=0.0, std=1.0)

    base_result = run_single_ppo_update(
        problem=problem,
        policy=base_policy,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=2,
        learning_rate=1e-3,
        seed=0,
    )
    identity_result = run_single_ppo_update(
        problem=problem,
        policy=identity_policy,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=2,
        learning_rate=1e-3,
        seed=0,
    )

    assert torch.allclose(torch.tensor(base_result["policy_loss"]), torch.tensor(identity_result["policy_loss"]))
    assert torch.allclose(torch.tensor(base_result["value_loss"]), torch.tensor(identity_result["value_loss"]))
    assert torch.allclose(
        torch.tensor(base_result["value_loss_train_space"]),
        torch.tensor(identity_result["value_loss_train_space"]),
    )
    assert torch.allclose(
        torch.tensor(base_result["value_loss_raw_space"]),
        torch.tensor(identity_result["value_loss_raw_space"]),
    )
    assert torch.allclose(torch.tensor(base_result["approx_kl"]), torch.tensor(identity_result["approx_kl"]))
    assert torch.allclose(torch.tensor(base_result["final_hpwl"]), torch.tensor(identity_result["final_hpwl"]))
    assert torch.allclose(
        torch.tensor(base_result["batch_diagnostic"]["raw_advantage_summary"]["mean"]),
        torch.tensor(identity_result["batch_diagnostic"]["raw_advantage_summary"]["mean"]),
    )
    assert base_result["value_target_stats"] == identity_result["value_target_stats"]


def test_run_single_ppo_update_records_grad_clip_when_enabled():
    problem = build_two_movable_problem()
    policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)

    result = run_single_ppo_update(
        problem=problem,
        policy=policy,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=2,
        learning_rate=1e-3,
        seed=0,
        grad_clip_norm=0.1,
    )

    substep0 = result["substep_history"][0]
    assert substep0["grad_clip_norm"] == 0.1
    assert substep0["global_grad_norm_pre_clip"] >= substep0["global_grad_norm_post_clip"]
    assert substep0["global_grad_norm_post_clip"] <= 0.1001
    assert substep0["grad_clip_applied"] is True


def test_run_single_ppo_update_scales_first_substep_learning_rate_and_param_delta_norms():
    problem = build_two_movable_problem()
    base_policy = initialize_policy_for_problem(problem=problem, grid_shape=(4, 4), hidden_dim=16)
    scaled_policy = copy.deepcopy(base_policy)

    baseline = run_single_ppo_update(
        problem=problem,
        policy=base_policy,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=2,
        learning_rate=1e-3,
        seed=0,
        policy_epochs=1,
    )
    scaled = run_single_ppo_update(
        problem=problem,
        policy=scaled_policy,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=2,
        learning_rate=1e-3,
        seed=0,
        policy_epochs=1,
        first_substep_lr_scale=0.1,
    )

    baseline_row = baseline["substep_history"][0]
    scaled_row = scaled["substep_history"][0]
    assert torch.isclose(torch.tensor(scaled_row["effective_learning_rate"]), torch.tensor(1e-4))
    assert scaled_row["first_substep_lr_scale"] == 0.1
    assert scaled_row["global_param_delta_norm_from_pre_update"] < baseline_row["global_param_delta_norm_from_pre_update"] * 0.2
    assert scaled_row["policy_head_param_delta_norm_from_pre_update"] < baseline_row["policy_head_param_delta_norm_from_pre_update"] * 0.2


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
