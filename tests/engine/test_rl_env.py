from __future__ import annotations

from dataclasses import dataclass

from engine.rl_env import PlacementEnv


@dataclass(frozen=True)
class StubState:
    value: float


class StubAdapter:
    def initial_state(self) -> StubState:
        return StubState(value=0.0)

    def observe(self, state: StubState, step_index: int) -> dict[str, object]:
        return {"step_index": step_index, "value": state.value}

    def apply_action(self, state: StubState, action: dict[str, float]) -> StubState:
        return StubState(value=state.value + float(action["delta"]))

    def compute_reward(self, previous_state: StubState, next_state: StubState, reward_source: str) -> float:
        assert reward_source == "analytic_hpwl"
        return previous_state.value - next_state.value

    def is_terminal(self, state: StubState, step_index: int, max_steps: int) -> bool:
        return False

    def render(self, state: StubState) -> dict[str, object]:
        return {"value": state.value}


def test_rl_env_reset_and_step_follow_gymnasium_shape():
    env = PlacementEnv(adapter=StubAdapter(), max_steps=2, reward_source="analytic_hpwl")

    observation, info = env.reset()
    next_observation, reward, terminated, truncated, step_info = env.step({"delta": 1.5})

    assert observation["step_index"] == 0
    assert info["reward_source"] == "analytic_hpwl"
    assert next_observation["step_index"] == 1
    assert reward == -1.5
    assert terminated is False
    assert truncated is False
    assert step_info["reward_source"] == "analytic_hpwl"


def test_rl_env_truncates_after_max_steps():
    env = PlacementEnv(adapter=StubAdapter(), max_steps=1, reward_source="analytic_hpwl")
    env.reset()

    _, _, terminated, truncated, _ = env.step({"delta": 1.0})

    assert terminated is False
    assert truncated is True


def test_rl_env_render_returns_adapter_snapshot():
    env = PlacementEnv(adapter=StubAdapter(), max_steps=2, reward_source="analytic_hpwl")
    env.reset()
    env.step({"delta": 2.0})

    assert env.render()["value"] == 2.0
