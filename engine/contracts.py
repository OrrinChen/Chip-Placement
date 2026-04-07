from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DecisionEnv(Protocol):
    def transition(self, state: Any, action: Any) -> Any: ...


# Protocol for RL-friendly domain adapters that own observation, action,
# reward, and termination semantics while the engine owns the env loop.
@runtime_checkable
class RLAdapter(Protocol):
    def initial_state(self) -> Any: ...

    def observe(self, state: Any, step_index: int) -> dict[str, Any]: ...

    def apply_action(self, state: Any, action: Any) -> Any: ...

    def compute_reward(self, previous_state: Any, next_state: Any, reward_source: str) -> float: ...

    def is_terminal(self, state: Any, step_index: int, max_steps: int) -> bool: ...

    def render(self, state: Any) -> dict[str, Any]: ...


# Deprecated compatibility alias. Remove when Phase 2 foundation merges to main
# and downstream callers have switched to DecisionEnv.
TransitionModel = DecisionEnv


@runtime_checkable
class ConstraintChecker(Protocol):
    def check(self, state: Any) -> Any: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, candidate: Any) -> Any: ...


@runtime_checkable
class SurrogateModel(Protocol):
    def predict(self, observation: Any) -> Any: ...


@runtime_checkable
class SearchStrategy(Protocol):
    def solve(self, problem: Any, objective: Any) -> Any: ...


@runtime_checkable
class TrainingStrategy(Protocol):
    def fit(self, samples: Any) -> Any: ...


__all__ = [
    "ConstraintChecker",
    "DecisionEnv",
    "Evaluator",
    "RLAdapter",
    "SearchStrategy",
    "SurrogateModel",
    "TrainingStrategy",
    "TransitionModel",
]
