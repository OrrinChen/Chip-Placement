from __future__ import annotations

from chip.experiments.phase3_final_report import build_phase3_final_report


def test_build_phase3_final_report_contains_roadmap_sections():
    report = build_phase3_final_report()

    assert "roadmap_phase3_status" in report
    assert "benchmarks" in report
    assert "methods" in report
    assert "remaining_gaps" in report


def test_build_phase3_final_report_mentions_three_benchmarks_and_known_methods():
    report = build_phase3_final_report()

    assert tuple(report["benchmarks"]) == ("tiny", "small", "medium")
    assert "ppo_warm_start" in report["methods"]
    assert "ppo_scratch" in report["methods"]
