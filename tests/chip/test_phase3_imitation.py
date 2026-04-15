from __future__ import annotations

from pathlib import Path

import torch

from chip.experiments.phase3_baselines import run_greedy_episode
from chip.experiments.phase3_imitation import evaluate_behavior_cloning_policy
from chip.experiments.phase3_imitation import build_behavior_cloning_batch
from chip.experiments.phase3_imitation import masked_behavior_cloning_loss
from chip.experiments.phase3_imitation import train_behavior_cloning_policy
from chip.experiments.phase3_search import build_demonstrations_from_trace
from chip.experiments.phase3_search import run_beam_search_baseline
from tests.chip.test_rl_adapter import build_problem
from tests.chip.test_rl_adapter import build_two_movable_problem


def test_masked_behavior_cloning_ignores_invalid_actions():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    action_index = torch.tensor([1], dtype=torch.int64)
    invalid_action_mask = torch.tensor([[False, False, True, True]])

    loss = masked_behavior_cloning_loss(logits, action_index, invalid_action_mask)

    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_behavior_cloning_trainer_fits_tiny_search_demonstration(tmp_path: Path):
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

    result = train_behavior_cloning_policy(
        problem=problem,
        demonstrations=demonstrations,
        grid_shape=(4, 4),
        hidden_dim=16,
        epochs=120,
        learning_rate=1e-2,
        checkpoint_path=checkpoint_path,
    )
    evaluation = evaluate_behavior_cloning_policy(
        policy=result["policy"],
        problem=problem,
        demonstrations=demonstrations,
        grid_shape=(4, 4),
    )

    assert result["masked_accuracy"] >= 0.99
    assert evaluation["masked_accuracy"] >= 0.99
    assert checkpoint_path.exists()


def test_behavior_cloning_trainer_can_reserve_larger_macro_embedding_table(tmp_path: Path):
    problem = build_problem()
    search_result = run_beam_search_baseline(problem=problem, grid_shape=(4, 4), beam_width=2)
    demonstrations = build_demonstrations_from_trace(
        problem=problem,
        grid_shape=(4, 4),
        rollout_type="beam_search",
        trace=search_result["trace"],
        final_score=search_result["final_hpwl"],
    )

    result = train_behavior_cloning_policy(
        problem=problem,
        demonstrations=demonstrations,
        grid_shape=(4, 4),
        hidden_dim=16,
        epochs=5,
        learning_rate=1e-2,
        checkpoint_path=tmp_path / "bc_policy_full_index.pt",
        max_macro_index=8,
    )

    assert result["policy"].current_macro_embedding.weight.shape[0] == 8


def test_demonstrations_from_greedy_episode_preserve_valid_teacher_actions():
    problem = build_two_movable_problem()
    greedy = run_greedy_episode(problem=problem, grid_shape=(1, 6))
    demonstrations = build_demonstrations_from_trace(
        problem=problem,
        grid_shape=(1, 6),
        rollout_type="greedy",
        trace=greedy["trace"],
        final_score=greedy["final_hpwl"],
    )

    batch = build_behavior_cloning_batch(problem=problem, demonstrations=demonstrations, grid_shape=(1, 6))

    for row_index in range(batch.action_index.shape[0]):
        flat_mask = batch.action_mask[row_index].reshape(-1)
        action_index = int(batch.action_index[row_index].item())
        assert bool(flat_mask[action_index]) is False
        assert bool(flat_mask.all()) is False
