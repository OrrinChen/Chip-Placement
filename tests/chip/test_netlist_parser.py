from pathlib import Path

from chip.netlist_parser import _build_problem_graph
from chip.netlist_parser import _parse_components
from chip.netlist_parser import _parse_nets
from chip.netlist_parser import _parse_pins
from chip.netlist_parser import load_chip_problem
from chip.problem import DieArea
from chip.problem import Macro
from chip.problem import Net


def test_parser_loads_sample_benchmark():
    benchmark_dir = Path(__file__).resolve().parents[2] / "chip" / "benchmarks"
    graph, problem = load_chip_problem(benchmark_dir / "sample.lef", benchmark_dir / "sample.def")

    assert set(graph.node_ids) == {"U1", "U2", "U3"}
    assert problem.die_area.width == 100
    assert len(problem.nets) == 2


def test_parse_components_supports_source_and_fixed_attributes():
    def_text = """
COMPONENTS 2 ;
    - TAP0 TAPCELL_X1 + SOURCE DIST + FIXED ( 2280 2800 ) N ;
    - U1 NAND2_X1 + PLACED ( 3040 2800 ) FN ;
END COMPONENTS
"""
    macro_sizes = {"TAPCELL_X1": (1.0, 1.0), "NAND2_X1": (1.0, 1.0)}

    components = _parse_components(def_text, macro_sizes)

    assert set(components) == {"TAP0", "U1"}
    assert components["TAP0"].is_fixed is True
    assert components["U1"].is_fixed is False


def test_parse_nets_ignores_routed_geometry_points():
    def_text = """
NETS 1 ;
    - N1 ( U1 A ) ( PIN clk ) ( U2 Z ) + ROUTED metal2 ( 34280 1280 ) ( 34300 1280 ) ;
END NETS
"""

    nets = _parse_nets(def_text)

    assert len(nets) == 1
    assert nets[0].pins == ("U1", "clk", "U2")


def test_parse_pins_creates_fixed_terminals():
    def_text = """
PINS 1 ;
    - clk + NET clk + DIRECTION INPUT + USE SIGNAL
      + PORT
        + LAYER metal5 ( -140 -140 ) ( 140 140 )
        + PLACED ( 140 62860 ) N ;
END PINS
"""

    pins = _parse_pins(def_text)

    assert set(pins) == {"clk"}
    assert pins["clk"].is_fixed is True
    assert pins["clk"].x == 140.0


def test_build_problem_graph_uses_linear_net_expansion_for_high_fanout_nets():
    macros = {
        name: Macro(name=name, kind="CELL", width=1.0, height=1.0, x=float(index), y=0.0)
        for index, name in enumerate(["A", "B", "C", "D"])
    }
    nets = [Net(name="N1", pins=("A", "B", "C", "D"))]

    graph = _build_problem_graph(macros, nets, DieArea(0.0, 0.0, 10.0, 10.0))

    edge_pairs = {(source, target) for source, target, _ in graph.edges}
    assert edge_pairs == {("A", "B"), ("A", "C"), ("A", "D")}


def test_build_problem_graph_ignores_unknown_net_pins():
    macros = {
        name: Macro(name=name, kind="CELL", width=1.0, height=1.0, x=float(index), y=0.0)
        for index, name in enumerate(["A", "B"])
    }
    nets = [Net(name="N1", pins=("VDD", "A", "B"))]

    graph = _build_problem_graph(macros, nets, DieArea(0.0, 0.0, 10.0, 10.0))

    assert {(source, target) for source, target, _ in graph.edges} == {("A", "B")}
