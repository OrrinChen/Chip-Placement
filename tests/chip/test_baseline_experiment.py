from pathlib import Path

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
