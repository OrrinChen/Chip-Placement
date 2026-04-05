from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DieArea:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


@dataclass(frozen=True)
class Macro:
    name: str
    kind: str
    width: float
    height: float
    x: float
    y: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)


@dataclass(frozen=True)
class Net:
    name: str
    pins: tuple[str, ...]


@dataclass(frozen=True)
class ChipProblem:
    die_area: DieArea
    macros: dict[str, Macro]
    nets: list[Net]

    def placement(self) -> dict[str, tuple[float, float]]:
        return {macro.name: (macro.x, macro.y) for macro in self.macros.values()}
