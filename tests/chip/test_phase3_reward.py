from __future__ import annotations

import math

from chip.phase3_reward import EpisodeRewardNormalizer
from chip.phase3_reward import RewardWeights
from chip.phase3_reward import SeedCompletedRewardModel
from chip.rl_adapter import ChipRLState
from tests.chip.test_rl_adapter import build_problem


class HPWLProxy:
    def predict_cost(self, *, problem, placement, seed_placement):
        del seed_placement
        from chip.cost import hpwl_cost

        return hpwl_cost(problem, placement=placement)


class DummyHPWLEvaluator:
    def evaluate(self, placement):
        return float(sum(x + y for x, y in placement.values()))


def test_seed_complete_fills_unplaced_macros_from_seed_reference():
    problem = build_problem()
    reward_model = SeedCompletedRewardModel(
        problem=problem,
        seed_placement={"u1": (0.0, 0.0), "u2": (8.0, 0.0)},
        proxy=HPWLProxy(),
        normalize_rewards=False,
    )

    completed = reward_model.seed_complete({"u1": (4.0, 0.0)}, placed_macros=frozenset({"u1"}))

    assert completed == {"u1": (4.0, 0.0), "u2": (8.0, 0.0)}


def test_compute_raw_step_reward_is_positive_when_proxy_and_hpwl_improve():
    problem = build_problem()
    reward_model = SeedCompletedRewardModel(
        problem=problem,
        seed_placement={"u1": (10.0, 0.0), "u2": (8.0, 0.0)},
        proxy=HPWLProxy(),
        weights=RewardWeights(w_proxy=1.0, w_hpwl=0.1, legality_penalty=10.0),
        normalize_rewards=False,
    )
    previous_state = ChipRLState(
        problem=problem,
        placement={"u1": (10.0, 0.0), "u2": (8.0, 0.0)},
        placed_macros=frozenset({"u1"}),
    )
    next_state = ChipRLState(
        problem=problem,
        placement={"u1": (9.0, 0.0), "u2": (8.0, 0.0)},
        placed_macros=frozenset({"u1"}),
    )

    reward = reward_model.compute_raw_step_reward(previous_state, next_state, is_legal=True)

    assert reward > 0.0


def test_compute_raw_step_reward_applies_legality_penalty():
    problem = build_problem()
    reward_model = SeedCompletedRewardModel(
        problem=problem,
        seed_placement={"u1": (10.0, 0.0), "u2": (8.0, 0.0)},
        proxy=HPWLProxy(),
        weights=RewardWeights(w_proxy=1.0, w_hpwl=0.1, legality_penalty=10.0),
        normalize_rewards=False,
    )
    previous_state = ChipRLState(
        problem=problem,
        placement={"u1": (10.0, 0.0), "u2": (8.0, 0.0)},
        placed_macros=frozenset({"u1"}),
    )
    next_state = ChipRLState(
        problem=problem,
        placement={"u1": (9.0, 0.0), "u2": (8.0, 0.0)},
        placed_macros=frozenset({"u1"}),
    )

    legal_reward = reward_model.compute_raw_step_reward(previous_state, next_state, is_legal=True)
    illegal_reward = reward_model.compute_raw_step_reward(previous_state, next_state, is_legal=False)

    assert illegal_reward == legal_reward - 10.0


def test_compute_raw_step_reward_adds_completion_bonus_when_last_macro_is_placed():
    problem = build_problem()
    reward_model = SeedCompletedRewardModel(
        problem=problem,
        seed_placement={"u1": (10.0, 0.0), "u2": (8.0, 0.0)},
        proxy=HPWLProxy(),
        weights=RewardWeights(w_proxy=0.0, w_hpwl=0.0, legality_penalty=10.0, completion_bonus=123.0),
        normalize_rewards=False,
    )
    previous_state = ChipRLState(
        problem=problem,
        placement={"u1": (10.0, 0.0), "u2": (8.0, 0.0)},
        placed_macros=frozenset({"u1"}),
    )
    next_state = ChipRLState(
        problem=problem,
        placement={"u1": (10.0, 0.0), "u2": (8.0, 0.0)},
        placed_macros=frozenset({"u1", "u2"}),
    )

    reward = reward_model.compute_raw_step_reward(previous_state, next_state, is_legal=True)

    assert reward == 123.0


def test_episode_reward_normalizer_is_finite_and_resettable():
    normalizer = EpisodeRewardNormalizer()

    values = [normalizer.normalize(value) for value in (1.0, 2.0, -1.5)]
    normalizer.reset()
    reset_value = normalizer.normalize(1.0)

    assert all(math.isfinite(value) for value in values)
    assert math.isfinite(reset_value)
    assert normalizer.count == 1


def test_reward_model_uses_custom_hpwl_evaluator_when_provided():
    problem = build_problem()
    reward_model = SeedCompletedRewardModel(
        problem=problem,
        seed_placement={"u1": (10.0, 0.0), "u2": (8.0, 0.0)},
        proxy=HPWLProxy(),
        hpwl_evaluator=DummyHPWLEvaluator(),
        weights=RewardWeights(w_proxy=0.0, w_hpwl=1.0, legality_penalty=10.0),
        normalize_rewards=False,
    )
    previous_state = ChipRLState(
        problem=problem,
        placement={"u1": (10.0, 0.0), "u2": (8.0, 0.0)},
        placed_macros=frozenset({"u1"}),
    )
    next_state = ChipRLState(
        problem=problem,
        placement={"u1": (9.0, 0.0), "u2": (8.0, 0.0)},
        placed_macros=frozenset({"u1"}),
    )

    reward = reward_model.compute_raw_step_reward(previous_state, next_state, is_legal=True)

    assert reward == 1.0
