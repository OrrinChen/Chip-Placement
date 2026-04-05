from __future__ import annotations

from chip.problem import ChipProblem
from engine.constraints import ConstraintRegistry


def build_constraint_registry(problem: ChipProblem, min_spacing: float = 0.0) -> ConstraintRegistry:
    registry = ConstraintRegistry()
    registry.register("boundary", lambda placement: _boundary_violations(problem, placement), penalty_weight=5.0)
    registry.register("non_overlap", lambda placement: _non_overlap_violations(problem, placement), penalty_weight=10.0)
    if min_spacing > 0.0:
        registry.register(
            "min_spacing",
            lambda placement: _min_spacing_violations(problem, placement, min_spacing),
            penalty_weight=3.0,
        )
    return registry


def _boundary_violations(problem: ChipProblem, placement: dict[str, tuple[float, float]]) -> list[dict[str, float | str]]:
    violations: list[dict[str, float | str]] = []
    die = problem.die_area
    for macro in problem.macros.values():
        x, y = placement[macro.name]
        overflow = max(die.min_x - x, 0.0) + max(die.min_y - y, 0.0)
        overflow += max((x + macro.width) - die.max_x, 0.0)
        overflow += max((y + macro.height) - die.max_y, 0.0)
        if overflow > 0:
            violations.append({"macro": macro.name, "magnitude": overflow})
    return violations


def _non_overlap_violations(problem: ChipProblem, placement: dict[str, tuple[float, float]]) -> list[dict[str, float | str]]:
    violations: list[dict[str, float | str]] = []
    macros = list(problem.macros.values())
    for index, first in enumerate(macros):
        x1, y1 = placement[first.name]
        for second in macros[index + 1 :]:
            x2, y2 = placement[second.name]
            overlap_x = min(x1 + first.width, x2 + second.width) - max(x1, x2)
            overlap_y = min(y1 + first.height, y2 + second.height) - max(y1, y2)
            if overlap_x > 0 and overlap_y > 0:
                violations.append(
                    {
                        "first": first.name,
                        "second": second.name,
                        "magnitude": overlap_x * overlap_y,
                    }
                )
    return violations


def _min_spacing_violations(
    problem: ChipProblem,
    placement: dict[str, tuple[float, float]],
    min_spacing: float,
) -> list[dict[str, float | str]]:
    violations: list[dict[str, float | str]] = []
    macros = list(problem.macros.values())
    for index, first in enumerate(macros):
        x1, y1 = placement[first.name]
        for second in macros[index + 1 :]:
            x2, y2 = placement[second.name]
            gap_x = max(x2 - (x1 + first.width), x1 - (x2 + second.width), 0.0)
            gap_y = max(y2 - (y1 + first.height), y1 - (y2 + second.height), 0.0)
            if gap_x == 0.0 and gap_y == 0.0:
                continue
            if gap_x < min_spacing and gap_y < min_spacing:
                violations.append(
                    {
                        "first": first.name,
                        "second": second.name,
                        "magnitude": min_spacing - min(gap_x, gap_y),
                    }
                )
    return violations
