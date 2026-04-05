import subprocess
from pathlib import Path

from chip.experiments import baseline as baseline_module
from chip.experiments.baseline import _neighbor
from chip.experiments.baseline import run_baseline_experiment
from chip.problem import ChipProblem, DieArea, Macro


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
