import subprocess
from pathlib import Path

from chip.experiments import baseline as baseline_module
from chip.experiments.baseline import run_baseline_experiment


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
