import subprocess
from pathlib import Path

from chip.experiments import baseline as baseline_module
from chip.experiments.baseline import _neighbor
from chip.experiments.baseline import run_baseline_experiment
from chip.problem import ChipProblem, DieArea, Macro
from chip.problem import PlacementRow


def test_baseline_experiment_returns_sa_and_analytical_metrics():
    benchmark_dir = Path(__file__).resolve().parents[2] / "chip" / "benchmarks"
    result = run_baseline_experiment(
        benchmark_dir / "sample.lef",
        benchmark_dir / "sample.def",
        iterations=15,
        bridge_binary="definitely-missing-openroad",
    )

    assert result["openroad_available"] is False
    assert set(result["placements"]) == {"initial", "analytical", "sa"}
    assert "hpwl" in result["placements"]["initial"]
    assert "is_valid" in result["placements"]["sa"]


def test_baseline_experiment_degrades_when_openroad_execution_fails(monkeypatch):
    class FailingBridge:
        def __init__(self, *args, **kwargs):
            pass

        def is_available(self):
            return True

        def evaluate_placement(self, *args, **kwargs):
            raise subprocess.CalledProcessError(returncode=1, cmd="openroad", stderr="route failed")

    benchmark_dir = Path(__file__).resolve().parents[2] / "chip" / "benchmarks"
    monkeypatch.setattr(baseline_module, "OpenROADBridge", FailingBridge)

    result = run_baseline_experiment(benchmark_dir / "sample.lef", benchmark_dir / "sample.def")

    assert result["openroad_available"] is True
    assert result["openroad"] is None
    assert "route failed" in result["openroad_error"]


def test_neighbor_does_not_move_fixed_instances():
    problem = ChipProblem(
        die_area=DieArea(0, 0, 100, 100),
        macros={
            "fixed": Macro("fixed", "PIN", 0, 0, 10, 10, is_fixed=True),
        },
        nets=[],
    )
    placement = {"fixed": (10.0, 10.0)}

    updated = _neighbor(problem, placement, (0.0, 100.0, 0.0, 100.0), baseline_module.random.Random(7))

    assert updated["fixed"] == (10.0, 10.0)


def test_neighbor_keeps_vertical_moves_on_valid_rows():
    problem = ChipProblem(
        die_area=DieArea(0, 0, 100, 100),
        macros={
            "U1": Macro("U1", "CELL", 0.2, 2.0, 1.0, 0.0, is_fixed=False),
        },
        nets=[],
        rows=(
            PlacementRow(name="row0", origin_x=0.0, origin_y=0.0, orientation="N", site_count=500, step_x=0.2, step_y=0.0),
            PlacementRow(name="row1", origin_x=0.0, origin_y=2.0, orientation="FS", site_count=500, step_x=0.2, step_y=0.0),
            PlacementRow(name="row2", origin_x=0.0, origin_y=4.0, orientation="N", site_count=500, step_x=0.2, step_y=0.0),
        ),
    )
    placement = {"U1": (1.0, 0.0)}
    rng = baseline_module.random.Random(7)

    observed_y = set()
    for _ in range(40):
        placement = _neighbor(problem, placement, (0.0, 100.0, 0.0, 100.0), rng)
        observed_y.add(placement["U1"][1])

    assert observed_y <= {0.0, 2.0, 4.0}


def test_neighbor_can_swap_compatible_cells_in_dense_rows():
    class DeterministicRandom:
        def __init__(self):
            self._choices = ["swap", "U1", "U2"]

        def choice(self, seq):
            choice = self._choices.pop(0)
            assert choice in seq
            return choice

    problem = ChipProblem(
        die_area=DieArea(0, 0, 10, 10),
        macros={
            "U1": Macro("U1", "CELL", 0.2, 2.0, 0.0, 0.0, is_fixed=False),
            "U2": Macro("U2", "CELL", 0.2, 2.0, 0.2, 0.0, is_fixed=False),
        },
        nets=[],
        rows=(
            PlacementRow(name="row0", origin_x=0.0, origin_y=0.0, orientation="N", site_count=50, step_x=0.2, step_y=0.0),
        ),
    )

    updated = _neighbor(
        problem,
        {"U1": (0.0, 0.0), "U2": (0.2, 0.0)},
        (0.0, 10.0, 0.0, 10.0),
        DeterministicRandom(),
    )

    assert updated["U1"] == (0.2, 0.0)
    assert updated["U2"] == (0.0, 0.0)


def test_baseline_experiment_skips_analytical_when_graph_is_too_large():
    benchmark_dir = Path(__file__).resolve().parents[2] / "chip" / "benchmarks"

    result = run_baseline_experiment(
        benchmark_dir / "sample.lef",
        benchmark_dir / "sample.def",
        analytical_node_limit=0,
        bridge_binary="definitely-missing-openroad",
    )

    assert result["placements"]["analytical"]["status"] == "skipped"
    assert result["placements"]["analytical"]["reason"] == "node_limit_exceeded"


def test_baseline_experiment_returns_sa_runtime_seconds():
    benchmark_dir = Path(__file__).resolve().parents[2] / "chip" / "benchmarks"

    result = run_baseline_experiment(
        benchmark_dir / "sample.lef",
        benchmark_dir / "sample.def",
        bridge_binary="definitely-missing-openroad",
        iterations=1,
    )

    assert "runtime_seconds" in result["placements"]["sa"]
