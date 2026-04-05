from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ConstraintViolation:
    name: str
    magnitude: float
    penalty_weight: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstraintReport:
    violations: list[ConstraintViolation] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.violations

    @property
    def total_penalty(self) -> float:
        return sum(violation.magnitude * violation.penalty_weight for violation in self.violations)


@dataclass(frozen=True)
class RegisteredConstraint:
    name: str
    check_fn: Callable[[Any], list[dict[str, Any]]]
    penalty_weight: float


class ConstraintRegistry:
    def __init__(self) -> None:
        self._constraints: list[RegisteredConstraint] = []

    def register(
        self,
        name: str,
        check_fn: Callable[[Any], list[dict[str, Any]]],
        penalty_weight: float = 1.0,
    ) -> None:
        self._constraints.append(RegisteredConstraint(name, check_fn, penalty_weight))

    def check_all(self, candidate: Any) -> ConstraintReport:
        violations: list[ConstraintViolation] = []
        for constraint in self._constraints:
            for raw_violation in constraint.check_fn(candidate):
                violations.append(
                    ConstraintViolation(
                        name=constraint.name,
                        magnitude=float(raw_violation.get("magnitude", 0.0)),
                        penalty_weight=constraint.penalty_weight,
                        details={k: v for k, v in raw_violation.items() if k != "magnitude"},
                    )
                )
        return ConstraintReport(violations=violations)
