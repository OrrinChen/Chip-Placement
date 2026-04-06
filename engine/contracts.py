from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DecisionEnv(Protocol):
    def transition(self, state: Any, action: Any) -> Any: ...


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
    "SearchStrategy",
    "SurrogateModel",
    "TrainingStrategy",
    "TransitionModel",
]
