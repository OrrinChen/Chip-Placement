from __future__ import annotations

import re
from pathlib import Path

from chip.problem import ChipProblem, DieArea, Macro, Net
from engine.graph import ProblemGraph


LEF_MACRO_RE = re.compile(r"MACRO\s+(?P<name>\S+)(?P<body>.*?)END\s+(?P=name)", re.DOTALL)
LEF_SIZE_RE = re.compile(r"SIZE\s+(?P<width>\d+(?:\.\d+)?)\s+BY\s+(?P<height>\d+(?:\.\d+)?)\s*;")
DEF_DIEAREA_RE = re.compile(
    r"DIEAREA\s+\(\s*(?P<min_x>\d+(?:\.\d+)?)\s+(?P<min_y>\d+(?:\.\d+)?)\s*\)\s+\(\s*(?P<max_x>\d+(?:\.\d+)?)\s+(?P<max_y>\d+(?:\.\d+)?)\s*\)\s*;"
)
DEF_COMPONENT_RE = re.compile(
    r"-\s+(?P<name>\S+)\s+(?P<kind>\S+)\s+\+\s+(?:PLACED|FIXED)\s+\(\s*(?P<x>\d+(?:\.\d+)?)\s+(?P<y>\d+(?:\.\d+)?)\s*\)\s+\S+\s*;"
)
DEF_NET_RE = re.compile(r"-\s+(?P<name>\S+)\s+(?P<body>.*?);", re.DOTALL)
DEF_PIN_RE = re.compile(r"\(\s*(?P<instance>\S+)\s+\S+\s*\)")


def load_chip_problem(lef_path: str | Path, def_path: str | Path) -> tuple[ProblemGraph, ChipProblem]:
    lef_text = Path(lef_path).read_text()
    def_text = Path(def_path).read_text()

    macro_sizes = _parse_lef(lef_text)
    die_area = _parse_die_area(def_text)
    macros = _parse_components(def_text, macro_sizes)
    nets = _parse_nets(def_text)
    graph = _build_problem_graph(macros, nets, die_area)

    return graph, ChipProblem(die_area=die_area, macros=macros, nets=nets)


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


def _parse_die_area(def_text: str) -> DieArea:
    match = DEF_DIEAREA_RE.search(def_text)
    if match is None:
        raise ValueError("DEF file missing DIEAREA")
    return DieArea(
        min_x=float(match.group("min_x")),
        min_y=float(match.group("min_y")),
        max_x=float(match.group("max_x")),
        max_y=float(match.group("max_y")),
    )


def _parse_components(def_text: str, macro_sizes: dict[str, tuple[float, float]]) -> dict[str, Macro]:
    macros: dict[str, Macro] = {}
    for match in DEF_COMPONENT_RE.finditer(def_text):
        kind = match.group("kind")
        width, height = macro_sizes[kind]
        macro = Macro(
            name=match.group("name"),
            kind=kind,
            width=width,
            height=height,
            x=float(match.group("x")),
            y=float(match.group("y")),
        )
        macros[macro.name] = macro
    return macros


def _parse_nets(def_text: str) -> list[Net]:
    nets: list[Net] = []
    nets_section = def_text.split("NETS", maxsplit=1)[-1]
    for match in DEF_NET_RE.finditer(nets_section):
        pins = tuple(pin_match.group("instance") for pin_match in DEF_PIN_RE.finditer(match.group("body")))
        if pins:
            nets.append(Net(name=match.group("name"), pins=pins))
    return nets


def _build_problem_graph(
    macros: dict[str, Macro],
    nets: list[Net],
    die_area: DieArea,
) -> ProblemGraph:
    edge_weights: dict[tuple[str, str], float] = {}
    for net in nets:
        for index, source in enumerate(net.pins):
            for target in net.pins[index + 1 :]:
                key = tuple(sorted((source, target)))
                edge_weights[key] = edge_weights.get(key, 0.0) + 1.0

    nodes = {
        name: {
            "kind": macro.kind,
            "width": macro.width,
            "height": macro.height,
            "x": macro.x,
            "y": macro.y,
        }
        for name, macro in macros.items()
    }
    edges = [(source, target, {"weight": weight}) for (source, target), weight in edge_weights.items()]
    return ProblemGraph(
        nodes=nodes,
        edges=edges,
        global_features={"bounds": (die_area.min_x, die_area.max_x, die_area.min_y, die_area.max_y)},
    )
