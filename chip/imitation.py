from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from chip.problem import ChipProblem
from chip.rl_adapter import ChipRLAdapter
from chip.rl_adapter import ChipRLState


@dataclass(frozen=True)
class DemonstrationStep:
    design: str
    rollout_type: str
    step_index: int
    current_macro: str
    action_index: int
    action_mask: list[bool]
    final_score: float
    placement: dict[str, tuple[float, float]]


def save_demonstrations(path: Path, steps: list[DemonstrationStep]) -> None:
    payload = [asdict(step) for step in steps]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_demonstrations(path: Path) -> list[DemonstrationStep]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    demonstrations: list[DemonstrationStep] = []
    for row in payload:
        placement = {name: (float(coords[0]), float(coords[1])) for name, coords in row["placement"].items()}
        demonstrations.append(DemonstrationStep(**{**row, "placement": placement}))
    return demonstrations


def observation_from_demonstration(
    *,
    problem: ChipProblem,
    demonstration: DemonstrationStep,
    grid_shape: tuple[int, int] = (32, 32),
) -> dict[str, object]:
    adapter = ChipRLAdapter(problem=problem, grid_shape=grid_shape)
    ordered = adapter.ordered_macros()
    placed_macros = frozenset(ordered[: demonstration.step_index])
    state = ChipRLState(problem=problem, placement=dict(demonstration.placement), placed_macros=placed_macros)
    return adapter.observe(state, step_index=demonstration.step_index)
