from __future__ import annotations

from pathlib import Path

from chip.experiments.phase3_macroplacement_benchmarks import resolve_ariane_benchmark
from chip.experiments.phase3_macroplacement_probe import (
    evaluate_reference_placement,
    run_legality_probe,
)


def macroplacement_repo_root() -> Path:
    return Path(__file__).resolve().parents[2] / "tmp" / "MacroPlacement"


def test_evaluate_reference_placement_reports_openroad_metrics() -> None:
    spec = resolve_ariane_benchmark(repo_root=macroplacement_repo_root(), variant="ariane133", platform="NanGate45")

    captured: dict[str, object] = {}

    def fake_evaluate(request, bridge=None, placement=None, orientations=None, timeout_seconds=None):
        del bridge, timeout_seconds
        captured["request"] = request
        captured["placement"] = placement
        captured["orientations"] = orientations
        return {
            "final_hpwl": 123.0,
            "total_routed_wirelength": 456.0,
            "wns": -0.1,
            "tns": -1.0,
            "drc_violations": 0,
            "runtime_seconds": 7.5,
        }

    from chip.experiments import phase3_macroplacement_probe as probe_module

    original = probe_module.evaluate_openroad_request
    probe_module.evaluate_openroad_request = fake_evaluate
    try:
        report = evaluate_reference_placement(spec, timeout_seconds=120)
    finally:
        probe_module.evaluate_openroad_request = original

    assert report["benchmark"] == "ariane133_ng45"
    assert report["placement_mode"] == "reference_def"
    assert report["final_hpwl"] > 0.0
    assert report["total_routed_wirelength"] > 0.0
    assert "wns" in report
    assert "tns" in report
    assert "drc_violations" in report
    assert report["runtime_seconds"] > 0.0
    assert captured["request"].placement_mode == "sa_injected"
    assert captured["request"].benchmark_name == "ariane133_ng45"
    assert captured["placement"]
    assert captured["orientations"]


def test_run_legality_probe_reports_macro_count_and_mask_timing(tmp_path: Path) -> None:
    spec = resolve_ariane_benchmark(repo_root=macroplacement_repo_root(), variant="ariane133", platform="NanGate45")

    report = run_legality_probe(spec, artifact_dir=tmp_path, grid_shape=(32, 32))

    assert report["benchmark"] == "ariane133_ng45"
    assert report["movable_macro_count"] > 0
    assert report["single_legality_check_seconds"] >= 0.0
    assert report["build_action_mask_seconds"] >= 0.0
    assert 0.0 <= report["valid_action_fraction"] <= 1.0
    assert Path(report["combined_lef_path"]).exists()
