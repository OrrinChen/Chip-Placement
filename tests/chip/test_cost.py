from pathlib import Path

from chip.cost import hpwl_cost
from chip.netlist_parser import load_chip_problem


def test_hpwl_cost_matches_manual_span():
    benchmark_dir = Path(__file__).resolve().parents[2] / "chip" / "benchmarks"
    _, problem = load_chip_problem(benchmark_dir / "sample.lef", benchmark_dir / "sample.def")

    assert hpwl_cost(problem) == 100.0
