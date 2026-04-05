from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TransitionModel(Protocol):
    def transition(self, state: Any, action: Any) -> Any: ...


@runtime_checkable
class ConstraintChecker(Protocol):
    def check(self, state: Any) -> Any: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, candidate: Any) -> Any: ...


@runtime_checkable
class SurrogateModel(Protocol):
    def predict(self, observation: Any) -> Any: ...
