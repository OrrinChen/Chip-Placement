from pathlib import Path

from chip.constraints import build_constraint_registry
from chip.netlist_parser import load_chip_problem
from chip.problem import ChipProblem, DieArea, Macro


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


def test_constraint_registry_ignores_floating_point_boundary_noise():
    problem = ChipProblem(
        die_area=DieArea(0.0, 0.0, 10.0, 10.0),
        macros={
            "U1": Macro(name="U1", kind="M1", width=5.0, height=5.0, x=5.0, y=5.0),
        },
        nets=[],
    )
    placement = {"U1": (5.00000000000001, 5.0)}

    report = build_constraint_registry(problem).check_all(placement)

    assert report.is_valid is True


def test_constraint_registry_ignores_floating_point_overlap_noise():
    problem = ChipProblem(
        die_area=DieArea(0.0, 0.0, 10.0, 10.0),
        macros={
            "U1": Macro(name="U1", kind="M1", width=2.0, height=2.0, x=0.0, y=0.0),
            "U2": Macro(name="U2", kind="M2", width=2.0, height=2.0, x=2.0, y=0.0),
        },
        nets=[],
    )
    placement = {
        "U1": (0.0, 0.0),
        "U2": (1.99999999999999, 0.0),
    }

    report = build_constraint_registry(problem).check_all(placement)

    assert report.is_valid is True
