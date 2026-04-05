from __future__ import annotations

from chip.problem import ChipProblem


def hpwl_cost(problem: ChipProblem, placement: dict[str, tuple[float, float]] | None = None) -> float:
    placement = placement or problem.placement()
    total_hpwl = 0.0
    for net in problem.nets:
        xs = []
        ys = []
        for pin in net.pins:
            macro = problem.macros[pin]
            x, y = placement[pin]
            xs.append(x + macro.width / 2.0)
            ys.append(y + macro.height / 2.0)
        total_hpwl += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total_hpwl
