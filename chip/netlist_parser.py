from __future__ import annotations

import re
import math
from dataclasses import dataclass
from pathlib import Path

from chip.problem import ChipProblem, DieArea, Macro, Net, PlacementRow
from engine.graph import ProblemGraph


LEF_MACRO_RE = re.compile(r"MACRO\s+(?P<name>\S+)(?P<body>.*?)END\s+(?P=name)", re.DOTALL)
LEF_SIZE_RE = re.compile(r"SIZE\s+(?P<width>\d+(?:\.\d+)?)\s+BY\s+(?P<height>\d+(?:\.\d+)?)\s*;")
DEF_UNITS_RE = re.compile(r"UNITS\s+DISTANCE\s+MICRONS\s+(?P<dbu>\d+(?:\.\d+)?)\s*;")
DEF_DIEAREA_RE = re.compile(
    r"DIEAREA\s+\(\s*(?P<min_x>\d+(?:\.\d+)?)\s+(?P<min_y>\d+(?:\.\d+)?)\s*\)\s+\(\s*(?P<max_x>\d+(?:\.\d+)?)\s+(?P<max_y>\d+(?:\.\d+)?)\s*\)\s*;"
)
DEF_ROW_RE = re.compile(
    r"ROW\s+(?P<name>\S+)\s+(?P<site>\S+)\s+(?P<x>\d+(?:\.\d+)?)\s+(?P<y>\d+(?:\.\d+)?)\s+(?P<orientation>\S+)\s+DO\s+(?P<count>\d+)\s+BY\s+\d+\s+STEP\s+(?P<step_x>\d+(?:\.\d+)?)\s+(?P<step_y>\d+(?:\.\d+)?)\s*;"
)
DEF_COMPONENT_HEADER_RE = re.compile(r"-\s+(?P<name>\S+)\s+(?P<kind>\S+)")
DEF_COMPONENT_PLACEMENT_RE = re.compile(
    r"\+\s+(?P<status>PLACED|FIXED)\s+\(\s*(?P<x>\d+(?:\.\d+)?)\s+(?P<y>\d+(?:\.\d+)?)\s*\)\s+(?P<orientation>\S+)"
)
DEF_COMPONENT_UNPLACED_RE = re.compile(r"\+\s+UNPLACED\b")
DEF_NET_RE = re.compile(r"-\s+(?P<name>\S+)\s+(?P<body>.*?);", re.DOTALL)
DEF_CONNECTION_RE = re.compile(r"\(\s*(?P<body>[^()]+?)\s*\)")
DEF_PIN_HEADER_RE = re.compile(r"-\s+(?P<name>\S+)\s+\+\s+NET\s+(?P<net>\S+)")
DEF_PIN_PLACEMENT_RE = re.compile(
    r"\+\s+PLACED\s+\(\s*(?P<x>\d+(?:\.\d+)?)\s+(?P<y>\d+(?:\.\d+)?)\s*\)\s+(?P<orientation>\S+)"
)

def load_chip_problem(lef_path: str | Path, def_path: str | Path) -> tuple[ProblemGraph, ChipProblem]:
    lef_text = Path(lef_path).read_text()
    def_text = Path(def_path).read_text()

    dbu_per_micron = _parse_dbu_per_micron(def_text)
    macro_sizes = _parse_lef(lef_text)
    die_area = _parse_die_area(def_text, dbu_per_micron)
    rows = _parse_rows(def_text, dbu_per_micron)
    macros = _parse_components(def_text, macro_sizes, dbu_per_micron, rows=rows, die_area=die_area)
    macros.update(_parse_pins(def_text, dbu_per_micron))
    nets = _parse_nets(def_text)
    graph = _build_problem_graph(macros, nets, die_area)

    return graph, ChipProblem(die_area=die_area, macros=macros, nets=nets, rows=tuple(rows))


def _parse_lef(lef_text: str) -> dict[str, tuple[float, float]]:
    macro_sizes: dict[str, tuple[float, float]] = {}
    for match in LEF_MACRO_RE.finditer(lef_text):
        size_match = LEF_SIZE_RE.search(match.group("body"))
        if not size_match:
            continue
        macro_sizes[match.group("name")] = (
            float(size_match.group("width")),
            float(size_match.group("height")),
        )
    return macro_sizes


def _parse_dbu_per_micron(def_text: str) -> float:
    match = DEF_UNITS_RE.search(def_text)
    if match is None:
        return 1.0
    return float(match.group("dbu"))


def _parse_die_area(def_text: str, dbu_per_micron: float = 1.0) -> DieArea:
    match = DEF_DIEAREA_RE.search(def_text)
    if match is None:
        raise ValueError("DEF file missing DIEAREA")
    return DieArea(
        min_x=float(match.group("min_x")) / dbu_per_micron,
        min_y=float(match.group("min_y")) / dbu_per_micron,
        max_x=float(match.group("max_x")) / dbu_per_micron,
        max_y=float(match.group("max_y")) / dbu_per_micron,
    )


def _parse_rows(def_text: str, dbu_per_micron: float = 1.0) -> list[PlacementRow]:
    rows = [
        PlacementRow(
            name=match.group("name"),
            origin_x=float(match.group("x")) / dbu_per_micron,
            origin_y=float(match.group("y")) / dbu_per_micron,
            orientation=match.group("orientation"),
            site_count=int(match.group("count")),
            step_x=float(match.group("step_x")) / dbu_per_micron,
            step_y=float(match.group("step_y")) / dbu_per_micron,
        )
        for match in DEF_ROW_RE.finditer(def_text)
    ]
    return sorted(rows, key=lambda row: (row.origin_y, row.origin_x))


def _parse_components(
    def_text: str,
    macro_sizes: dict[str, tuple[float, float]],
    dbu_per_micron: float = 1.0,
    rows: list[PlacementRow] | None = None,
    die_area: DieArea | None = None,
) -> dict[str, Macro]:
    macros: dict[str, Macro] = {}
    pending_unplaced: list[tuple[str, str, float, float]] = []
    in_components = False
    current_record: list[str] = []
    for line in def_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("COMPONENTS"):
            in_components = True
            continue
        if in_components and stripped.startswith("END COMPONENTS"):
            break
        if not in_components:
            continue

        if stripped.startswith("- "):
            current_record = [stripped]
        elif current_record:
            current_record.append(stripped)

        if current_record and stripped.endswith(";"):
            record = " ".join(current_record)
            header_match = DEF_COMPONENT_HEADER_RE.search(record)
            placement_match = DEF_COMPONENT_PLACEMENT_RE.search(record)
            unplaced_match = DEF_COMPONENT_UNPLACED_RE.search(record)
            if header_match and placement_match:
                kind = header_match.group("kind")
                width, height = macro_sizes[kind]
                macro = Macro(
                    name=header_match.group("name"),
                    kind=kind,
                    width=width,
                    height=height,
                    x=float(placement_match.group("x")) / dbu_per_micron,
                    y=float(placement_match.group("y")) / dbu_per_micron,
                    orientation=placement_match.group("orientation"),
                    is_fixed=placement_match.group("status") == "FIXED",
                )
                macros[macro.name] = macro
            elif header_match and unplaced_match:
                kind = header_match.group("kind")
                width, height = macro_sizes[kind]
                pending_unplaced.append((header_match.group("name"), kind, width, height))
            current_record = []
    macros.update(_pack_unplaced_components(pending_unplaced, rows or [], die_area))
    return macros


def _parse_nets(def_text: str) -> list[Net]:
    nets: list[Net] = []
    nets_section = def_text.split("NETS", maxsplit=1)[-1]
    for match in DEF_NET_RE.finditer(nets_section):
        connection_section = match.group("body").split("+", maxsplit=1)[0]
        pins: list[str] = []
        for pin_match in DEF_CONNECTION_RE.finditer(connection_section):
            tokens = pin_match.group("body").split()
            if not tokens:
                continue
            if tokens[0] == "PIN" and len(tokens) > 1:
                pins.append(tokens[1])
            elif not tokens[0][0].isdigit():
                pins.append(tokens[0])
        if pins:
            nets.append(Net(name=match.group("name"), pins=tuple(pins)))
    return nets


def _parse_pins(def_text: str, dbu_per_micron: float = 1.0) -> dict[str, Macro]:
    pins: dict[str, Macro] = {}
    in_pins = False
    current_record: list[str] = []
    for line in def_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("PINS"):
            in_pins = True
            continue
        if in_pins and stripped.startswith("END PINS"):
            break
        if not in_pins:
            continue

        if stripped.startswith("- "):
            current_record = [stripped]
        elif current_record:
            current_record.append(stripped)

        if current_record and stripped.endswith(";"):
            record = " ".join(current_record)
            header_match = DEF_PIN_HEADER_RE.search(record)
            placement_match = DEF_PIN_PLACEMENT_RE.search(record)
            if header_match and placement_match:
                pins[header_match.group("name")] = Macro(
                    name=header_match.group("name"),
                    kind="PIN",
                    width=0.0,
                    height=0.0,
                    x=float(placement_match.group("x")) / dbu_per_micron,
                    y=float(placement_match.group("y")) / dbu_per_micron,
                    orientation=placement_match.group("orientation"),
                    is_fixed=True,
                )
            current_record = []
    return pins


def _pack_unplaced_components(
    pending_unplaced: list[tuple[str, str, float, float]],
    rows: list[PlacementRow],
    die_area: DieArea | None,
) -> dict[str, Macro]:
    macros: dict[str, Macro] = {}
    if not pending_unplaced:
        return macros

    if rows:
        row_states = [{"row": row, "next_x": row.origin_x} for row in rows]
        row_index = 0
        for name, kind, width, height in pending_unplaced:
            placed = False
            for offset in range(len(row_states)):
                state = row_states[(row_index + offset) % len(row_states)]
                row = state["row"]
                site_width = max(row.step_x, width, 1e-9)
                row_capacity = row.origin_x + (row.site_count * row.step_x)
                next_x = max(state["next_x"], row.origin_x)
                snapped_width = math.ceil(width / site_width) * site_width
                if next_x + snapped_width <= row_capacity + 1e-9:
                    macros[name] = Macro(
                        name=name,
                        kind=kind,
                        width=width,
                        height=height,
                        x=next_x,
                        y=row.origin_y,
                        orientation=row.orientation,
                        is_fixed=False,
                    )
                    state["next_x"] = next_x + snapped_width
                    row_index = (row_index + offset) % len(row_states)
                    placed = True
                    break
            if placed:
                continue

    if die_area is None:
        die_area = DieArea(0.0, 0.0, 1000.0, 1000.0)

    cursor_x = die_area.min_x
    cursor_y = die_area.min_y
    row_height = max((height for _, _, _, height in pending_unplaced), default=1.0)
    for name, kind, width, height in pending_unplaced:
        if name in macros:
            continue
        if cursor_x + width > die_area.max_x:
            cursor_x = die_area.min_x
            cursor_y += row_height
        macros[name] = Macro(name=name, kind=kind, width=width, height=height, x=cursor_x, y=cursor_y, orientation="N")
        cursor_x += width
    return macros


def _build_problem_graph(
    macros: dict[str, Macro],
    nets: list[Net],
    die_area: DieArea,
) -> ProblemGraph:
    edge_weights: dict[tuple[str, str], float] = {}
    for net in nets:
        known_pins = [pin for pin in net.pins if pin in macros]
        if len(known_pins) < 2:
            continue
        anchor = known_pins[0]
        for target in known_pins[1:]:
            key = (anchor, target)
            edge_weights[key] = edge_weights.get(key, 0.0) + 1.0

    nodes = {
        name: {
            "kind": macro.kind,
            "width": macro.width,
            "height": macro.height,
            "x": macro.x,
            "y": macro.y,
            "orientation": macro.orientation,
            "is_fixed": macro.is_fixed,
        }
        for name, macro in macros.items()
    }
    edges = [(source, target, {"weight": weight}) for (source, target), weight in edge_weights.items()]
    return ProblemGraph(
        nodes=nodes,
        edges=edges,
        global_features={
            "bounds": (die_area.min_x, die_area.max_x, die_area.min_y, die_area.max_y),
            "fixed_nodes": {name for name, macro in macros.items() if macro.is_fixed},
        },
    )
