from __future__ import annotations

import json
from pathlib import Path


def build_phase3_final_report() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    ladder_summary = json.loads(
        (root / "experiments" / "chip" / "phase3_benchmark_ladder" / "summary.json").read_text(encoding="utf-8")
    )
    teacher_summary = json.loads(
        (root / "experiments" / "chip" / "phase3_teacher_strength_sweep" / "summary.json").read_text(encoding="utf-8")
    )
    benchmark_order = tuple(name for name in ("tiny", "small", "medium") if name in ladder_summary)
    return {
        "roadmap_phase3_status": "partial",
        "benchmarks": benchmark_order,
        "methods": (
            "random",
            "greedy",
            "beam",
            "behavior_cloning",
            "ppo_scratch",
            "ppo_warm_start",
        ),
        "sample_efficiency_summary": ladder_summary,
        "teacher_strength_summary": teacher_summary,
        "remaining_gaps": (
            "proxy_reward_mainline_not_default",
            "roadmap_results_table_missing_sa_analytical_openroad_rows",
            "routed_timing_runtime_report_not_unified_across_three_plus_real_benchmarks",
        ),
    }
