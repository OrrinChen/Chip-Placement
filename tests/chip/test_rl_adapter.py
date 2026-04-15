from __future__ import annotations

from dataclasses import dataclass

from chip.problem import ChipProblem
from chip.problem import DieArea
from chip.problem import Macro
from chip.problem import Net
from chip.problem import PlacementRow
from chip.rl_adapter import build_macro_order
from chip.rl_adapter import ChipRLAdapter
from chip.rl_adapter import ChipRLState


@dataclass(frozen=True)
class StubProxy:
    prediction: float

    def predict(self, observation):
        return self.prediction


class StepProxy:
    def predict(self, observation):
        return float(observation["placed_count"])


class DummyHPWLEvaluator:
    def evaluate(self, placement):
        return float(sum(x + y for x, y in placement.values()))


class StubRewardModel:
    def __init__(self, reward: float) -> None:
        self.reward = reward
        self.reset_count = 0

    def compute_step_reward(self, previous_state, next_state, *, is_legal: bool) -> float:
        del previous_state, next_state, is_legal
        return self.reward

    def reset_episode(self) -> None:
        self.reset_count += 1


@dataclass(frozen=True)
class StubOracle:
    value: float

    def evaluate(self, candidate):
        return self.value


def build_problem() -> ChipProblem:
    return ChipProblem(
        die_area=DieArea(min_x=0.0, min_y=0.0, max_x=12.0, max_y=4.0),
        macros={
            "u1": Macro(name="u1", kind="CELL", width=1.0, height=1.0, x=0.0, y=0.0, is_fixed=False),
            "u2": Macro(name="u2", kind="CELL", width=1.0, height=1.0, x=8.0, y=0.0, is_fixed=True),
        },
        nets=[Net(name="n1", pins=("u1", "u2"))],
        rows=(
            PlacementRow(name="ROW_0", origin_x=0.0, origin_y=0.0, orientation="N", site_count=12, step_x=1.0, step_y=0.0),
            PlacementRow(name="ROW_1", origin_x=0.0, origin_y=2.0, orientation="FS", site_count=12, step_x=1.0, step_y=0.0),
        ),
    )


def build_two_movable_problem() -> ChipProblem:
    return ChipProblem(
        die_area=DieArea(min_x=0.0, min_y=0.0, max_x=6.0, max_y=2.0),
        macros={
            "u1": Macro(name="u1", kind="MACRO", width=1.0, height=1.0, x=0.0, y=0.0, is_fixed=False),
            "u2": Macro(name="u2", kind="MACRO", width=1.0, height=1.0, x=4.0, y=0.0, is_fixed=False),
        },
        nets=[Net(name="n1", pins=("u1", "u2"))],
        rows=(
            PlacementRow(name="ROW_0", origin_x=0.0, origin_y=0.0, orientation="N", site_count=6, step_x=1.0, step_y=0.0),
        ),
    )


def test_chip_rl_adapter_orders_macros_by_descending_area_then_name():
    problem = ChipProblem(
        die_area=DieArea(min_x=0.0, min_y=0.0, max_x=12.0, max_y=4.0),
        macros={
            "u2": Macro(name="u2", kind="CELL", width=1.0, height=1.0, x=8.0, y=0.0, is_fixed=False),
            "u1": Macro(name="u1", kind="CELL", width=2.0, height=1.0, x=0.0, y=0.0, is_fixed=False),
            "u3": Macro(name="u3", kind="CELL", width=1.0, height=1.0, x=4.0, y=0.0, is_fixed=False),
        },
        nets=[],
    )
    adapter = ChipRLAdapter(problem=problem, grid_shape=(4, 4))

    assert adapter.ordered_macros() == ("u1", "u2", "u3")


def test_build_macro_order_supports_seed_yx_strategy():
    problem = ChipProblem(
        die_area=DieArea(min_x=0.0, min_y=0.0, max_x=12.0, max_y=12.0),
        macros={
            "b": Macro(name="b", kind="MACRO", width=1.0, height=1.0, x=8.0, y=2.0, is_fixed=False),
            "a": Macro(name="a", kind="MACRO", width=2.0, height=2.0, x=4.0, y=2.0, is_fixed=False),
            "c": Macro(name="c", kind="MACRO", width=1.0, height=1.0, x=1.0, y=7.0, is_fixed=False),
        },
        nets=[],
    )

    assert build_macro_order(problem, "seed_yx") == ("a", "b", "c")


def test_chip_rl_adapter_respects_explicit_macro_order():
    problem = build_two_movable_problem()
    adapter = ChipRLAdapter(problem=problem, grid_shape=(1, 6), macro_order=("u2", "u1"))

    assert adapter.ordered_macros() == ("u2", "u1")
    assert adapter.current_macro_for_step(0) == "u2"


def test_chip_rl_adapter_observe_exposes_graph_and_step_index():
    adapter = ChipRLAdapter(problem=build_problem(), grid_shape=(4, 4))

    observation = adapter.observe(adapter.initial_state(), step_index=0)

    assert observation["step_index"] == 0
    assert observation["current_macro"] == "u1"
    assert observation["current_macro_index"] == 0
    assert observation["placed_count"] == 0
    assert observation["movable_macros"] == ("u1",)
    assert tuple(observation["canvas"].shape) == (6, 4, 4)
    assert tuple(observation["action_mask"].shape) == (4, 4)
    assert observation["graph"].nodes["u1"]["x"] == 0.0
    assert observation["graph"].nodes["u1"]["y"] == 0.0
    assert observation["graph"].nodes["u1"]["placed_flag"] == 0.0
    assert observation["graph"].nodes["u1"]["current_macro_flag"] == 1.0
    assert observation["graph"].nodes["u2"]["current_macro_flag"] == 0.0


def test_chip_rl_adapter_apply_action_snaps_y_to_row():
    adapter = ChipRLAdapter(problem=build_problem(), grid_shape=(4, 4))

    next_state = adapter.apply_action(adapter.initial_state(), {"macro": "u1", "target": (3.2, 1.6)})

    assert next_state.placement["u1"] == (3.0, 2.0)
    assert "u1" in next_state.placed_macros


def test_chip_rl_adapter_analytic_hpwl_reward_improves_when_hpwl_drops():
    adapter = ChipRLAdapter(problem=build_problem(), grid_shape=(4, 4))
    state = adapter.initial_state()
    better_state = adapter.apply_action(state, {"macro": "u1", "target": (4.0, 0.0)})

    reward = adapter.compute_reward(state, better_state, "analytic_hpwl")

    assert reward > 0.0


def test_chip_rl_adapter_proxy_reward_negates_predicted_cost():
    adapter = ChipRLAdapter(problem=build_problem(), proxy=StubProxy(prediction=42.0), grid_shape=(4, 4))

    reward = adapter.compute_reward(adapter.initial_state(), adapter.initial_state(), "proxy")

    assert reward == -42.0


def test_chip_rl_adapter_oracle_reward_negates_evaluator_cost():
    adapter = ChipRLAdapter(problem=build_problem(), oracle=StubOracle(value=17.5), grid_shape=(4, 4))

    reward = adapter.compute_reward(adapter.initial_state(), adapter.initial_state(), "oracle")

    assert reward == -17.5


def test_chip_rl_adapter_proxy_with_legality_penalty_subtracts_penalty_for_invalid_state():
    adapter = ChipRLAdapter(problem=build_problem(), proxy=StubProxy(prediction=10.0), legality_penalty=25.0, grid_shape=(4, 4))
    illegal_state = ChipRLState(
        problem=build_problem(),
        placement={"u1": (20.0, 20.0), "u2": (8.0, 0.0)},
        placed_macros=frozenset({"u1"}),
    )

    reward = adapter.compute_reward(adapter.initial_state(), illegal_state, "proxy_with_legality_penalty")

    assert reward == -35.0


def test_chip_rl_adapter_proxy_delta_uses_seed_completed_delta():
    adapter = ChipRLAdapter(problem=build_problem(), proxy=StepProxy(), grid_shape=(4, 4))
    state = adapter.initial_state()
    next_state = adapter.apply_action(state, {"macro": "u1", "target": (4.0, 0.0)})

    reward = adapter.compute_reward(state, next_state, "proxy_delta")

    assert reward == -1.0


def test_chip_rl_adapter_phase3_reward_source_delegates_to_reward_model():
    reward_model = StubRewardModel(reward=7.5)
    adapter = ChipRLAdapter(problem=build_problem(), reward_model=reward_model, grid_shape=(4, 4))
    state = adapter.initial_state()
    next_state = adapter.apply_action(state, {"macro": "u1", "target": (4.0, 0.0)})

    reward = adapter.compute_reward(state, next_state, "phase3_proxy_hpwl")

    assert reward == 7.5


def test_chip_rl_adapter_can_reset_reward_model_state():
    reward_model = StubRewardModel(reward=0.0)
    adapter = ChipRLAdapter(problem=build_problem(), reward_model=reward_model, grid_shape=(4, 4))

    adapter.reset_reward_state()
    adapter.reset_reward_state()

    assert reward_model.reset_count == 2


def test_chip_rl_adapter_current_hpwl_uses_custom_evaluator():
    adapter = ChipRLAdapter(problem=build_problem(), hpwl_evaluator=DummyHPWLEvaluator(), grid_shape=(4, 4))

    hpwl = adapter.current_hpwl(adapter.initial_state())

    assert hpwl == 8.0


def test_chip_rl_adapter_action_mask_ignores_future_unplaced_macros():
    adapter = ChipRLAdapter(problem=build_two_movable_problem(), grid_shape=(1, 6))

    observation = adapter.observe(adapter.initial_state(), step_index=0)

    # The future movable macro u2 sits at x=4.0 in the seed placement, but it should not
    # reserve that slot before it becomes the current macro.
    assert bool(observation["action_mask"][0, 4]) is False


def test_chip_rl_adapter_canvas_occupancy_only_marks_fixed_and_placed_macros():
    adapter = ChipRLAdapter(problem=build_two_movable_problem(), grid_shape=(1, 6))

    initial_observation = adapter.observe(adapter.initial_state(), step_index=0)
    assert float(initial_observation["canvas"][0].sum().item()) == 0.0

    next_state = adapter.apply_action(adapter.initial_state(), {"macro": "u1", "target": (1.0, 0.0)})
    next_observation = adapter.observe(next_state, step_index=1)

    assert float(next_observation["canvas"][0].sum().item()) == 1.0
