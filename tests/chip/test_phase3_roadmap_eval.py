from __future__ import annotations

from types import SimpleNamespace

from chip.experiments import phase3_roadmap_eval
from chip.experiments.phase3_roadmap_eval import build_phase3_roadmap_eval_report


def test_build_phase3_roadmap_eval_report_contains_core_method_rows():
    report = build_phase3_roadmap_eval_report()

    assert "benchmarks" in report
    assert "method_rows" in report
    assert "remaining_gaps" in report
    assert "sa" in report["method_rows"]
    assert "analytical" in report["method_rows"]
    assert "ppo_scratch" in report["method_rows"]
    assert "ppo_warm_start" in report["method_rows"]


def test_build_phase3_roadmap_eval_report_carries_sample_openroad_row():
    report = build_phase3_roadmap_eval_report()

    assert "openroad_default" in report["method_rows"]
    assert "small" in report["method_rows"]["openroad_default"]


def test_build_phase3_real_macroplacement_sa_table_contains_three_benchmarks(monkeypatch):
    spec_by_variant = {
        "ariane133": SimpleNamespace(name="ariane133_ng45"),
        "bp_quad": SimpleNamespace(name="bp_quad_ng45"),
        "nvdla": SimpleNamespace(name="nvdla_ng45"),
    }

    monkeypatch.setattr(
        phase3_roadmap_eval,
        "resolve_macroplacement_benchmark",
        lambda *, variant, repo_root=None, platform="NanGate45": spec_by_variant[variant],
    )
    monkeypatch.setattr(
        phase3_roadmap_eval,
        "_run_real_macroplacement_sa_row",
        lambda spec, grid_shape=(64, 64): {"final_hpwl": float(len(spec.name)), "is_valid": True, "runtime_seconds": 1.0},
    )

    report = phase3_roadmap_eval.build_phase3_real_macroplacement_sa_table()

    assert report["benchmarks"] == ("ariane133_ng45", "bp_quad_ng45", "nvdla_ng45")
    assert "sa" in report["method_rows"]
    assert set(report["method_rows"]["sa"]) == set(report["benchmarks"])
