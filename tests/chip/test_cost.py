from pathlib import Path

from chip.cost import hpwl_cost
from chip.netlist_parser import load_chip_problem
from chip.problem import ChipProblem
from chip.problem import DieArea
from chip.problem import Macro
from chip.problem import Net


def test_hpwl_cost_matches_manual_span():
    benchmark_dir = Path(__file__).resolve().parents[2] / "chip" / "benchmarks"
    _, problem = load_chip_problem(benchmark_dir / "sample.lef", benchmark_dir / "sample.def")

    assert hpwl_cost(problem) == 100.0


def test_hpwl_cost_ignores_unknown_net_pins():
    problem = ChipProblem(
        die_area=DieArea(0.0, 0.0, 20.0, 20.0),
        macros={
            "A": Macro("A", "CELL", 2.0, 2.0, 0.0, 0.0),
            "B": Macro("B", "CELL", 2.0, 2.0, 10.0, 0.0),
        },
        nets=[Net("N1", ("VDD", "A", "B"))],
    )

    assert hpwl_cost(problem) == 10.0
