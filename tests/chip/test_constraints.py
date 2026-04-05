from pathlib import Path

from chip.constraints import build_constraint_registry
from chip.netlist_parser import load_chip_problem


def test_constraint_registry_flags_overlap():
    benchmark_dir = Path(__file__).resolve().parents[2] / "chip" / "benchmarks"
    _, problem = load_chip_problem(benchmark_dir / "sample.lef", benchmark_dir / "sample.def")
    placement = {
        "U1": (10.0, 10.0),
        "U2": (15.0, 15.0),
        "U3": (70.0, 50.0),
    }

    report = build_constraint_registry(problem, min_spacing=0.0).check_all(placement)

    assert report.is_valid is False
    assert any(violation.name == "non_overlap" for violation in report.violations)
