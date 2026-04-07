from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.contracts import RLAdapter


@dataclass
class PlacementEnv:
    adapter: RLAdapter
    max_steps: int
    reward_source: str = "analytic_hpwl"

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._state: Any | None = None
        self._step_index = 0

    @property
    def state(self) -> Any:
        if self._state is None:
            raise RuntimeError("Environment has not been reset")
        return self._state

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        del seed, options
        self._state = self.adapter.initial_state()
        self._step_index = 0
        return self.adapter.observe(self.state, self._step_index), {"reward_source": self.reward_source}

    def step(
        self,
        action: Any,
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        previous_state = self.state
        next_state = self.adapter.apply_action(previous_state, action)
        self._step_index += 1
        reward = float(self.adapter.compute_reward(previous_state, next_state, self.reward_source))
        terminated = bool(self.adapter.is_terminal(next_state, self._step_index, self.max_steps))
        truncated = bool(self._step_index >= self.max_steps and not terminated)
        self._state = next_state
        info = {
            "reward_source": self.reward_source,
            "step_index": self._step_index,
        }
        return self.adapter.observe(self.state, self._step_index), reward, terminated, truncated, info

    def render(self) -> dict[str, object]:
        return self.adapter.render(self.state)
