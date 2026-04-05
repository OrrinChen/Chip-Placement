from pathlib import Path

from chip.netlist_parser import load_chip_problem


def test_parser_loads_sample_benchmark():
    benchmark_dir = Path(__file__).resolve().parents[2] / "chip" / "benchmarks"
    graph, problem = load_chip_problem(benchmark_dir / "sample.lef", benchmark_dir / "sample.def")

    assert set(graph.node_ids) == {"U1", "U2", "U3"}
    assert problem.die_area.width == 100
    assert len(problem.nets) == 2
