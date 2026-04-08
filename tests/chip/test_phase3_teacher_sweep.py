from __future__ import annotations

from chip.experiments.phase3_benchmarks import build_phase3_benchmark_ladder
from chip.experiments.phase3_teacher_sweep import run_phase3_teacher_strength_sweep


def test_teacher_strength_sweep_reports_requested_beam_widths():
    benchmark = {item["name"]: item for item in build_phase3_benchmark_ladder()}["medium"]

    sweep = run_phase3_teacher_strength_sweep(
        problem=benchmark["problem"],
        grid_shape=benchmark["grid_shape"],
        hidden_dim=16,
        ppo_epochs=1,
        beam_widths=(1, 2),
        seeds=(0,),
    )

    assert sweep["beam_widths"] == [1, 2]
    assert [row["beam_width"] for row in sweep["rows"]] == [1, 2]


def test_teacher_strength_sweep_improves_teacher_quality_with_wider_beam():
    benchmark = {item["name"]: item for item in build_phase3_benchmark_ladder()}["medium"]

    sweep = run_phase3_teacher_strength_sweep(
        problem=benchmark["problem"],
        grid_shape=benchmark["grid_shape"],
        hidden_dim=16,
        ppo_epochs=1,
        beam_widths=(1, 2),
        seeds=(0,),
    )

    assert sweep["rows"][1]["beam_final_hpwl_mean"] <= sweep["rows"][0]["beam_final_hpwl_mean"]
