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
    orientation: str = "N"
    is_fixed: bool = False

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)


@dataclass(frozen=True)
class Net:
    name: str
    pins: tuple[str, ...]


@dataclass(frozen=True)
class PlacementRow:
    name: str
    origin_x: float
    origin_y: float
    orientation: str
    site_count: int
    step_x: float
    step_y: float

    @property
    def max_x(self) -> float:
        return self.origin_x + (self.site_count * self.step_x)


@dataclass(frozen=True)
class ChipProblem:
    die_area: DieArea
    macros: dict[str, Macro]
    nets: list[Net]
    rows: tuple[PlacementRow, ...] = ()

    def placement(self) -> dict[str, tuple[float, float]]:
        return {macro.name: (macro.x, macro.y) for macro in self.macros.values()}

    def site_pitch(self, default: float = 0.2) -> float:
        pitches = [row.step_x for row in self.rows if row.step_x > 0]
        return min(pitches) if pitches else default

    def nearest_row(self, target_y: float) -> PlacementRow | None:
        if not self.rows:
            return None
        return min(self.rows, key=lambda row: (abs(row.origin_y - target_y), row.origin_y, row.origin_x))

    def row_index(self, target_y: float) -> int | None:
        nearest = self.nearest_row(target_y)
        if nearest is None:
            return None
        return self.rows.index(nearest)

    def snap_movable_macro(
        self,
        macro_name: str,
        target_x: float,
        target_y: float,
        default_site_pitch: float = 0.2,
    ) -> tuple[float, float, str]:
        macro = self.macros[macro_name]
        row = self.nearest_row(target_y)
        if row is None:
            site_pitch = default_site_pitch
            snapped_x = round(target_x / site_pitch) * site_pitch
            clamped_x = min(max(snapped_x, self.die_area.min_x), self.die_area.max_x - macro.width)
            return (round(clamped_x, 6), round(macro.y, 6), macro.orientation)

        site_pitch = row.step_x if row.step_x > 0 else default_site_pitch
        snapped_x = round(target_x / site_pitch) * site_pitch
        min_x = row.origin_x
        max_x = row.max_x - macro.width
        clamped_x = min(max(snapped_x, min_x), max_x)
        return (round(clamped_x, 6), round(row.origin_y, 6), row.orientation)
