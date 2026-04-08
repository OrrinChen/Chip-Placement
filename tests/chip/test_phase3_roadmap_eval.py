from __future__ import annotations

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
